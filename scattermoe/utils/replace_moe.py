import torch
from .. import parallel_linear, flatten_sort_count
from torch.nn import functional as F

import logging

def replace_function(cls, fun_name):
    def decorator(fun):
        def _fun(*args, **kwargs):
            filename = fun.__code__.co_filename
            name = fun.__name__
            logging.info(f"Replacing `{cls.__name__}.{fun_name}` with {filename}:{name}")
            setattr(cls, fun_name, fun)
            return fun(*args, **kwargs)
        setattr(cls, fun_name, _fun)
    return decorator

try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    @replace_function(cls=GptOssExperts, fun_name='forward')
    def gpt_oss_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        k = router_indices.shape[1]
        selected_weights = torch.gather(routing_weights, dim=1, index=router_indices)
        router_indices = router_indices.flatten()
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(router_indices, num_experts=self.num_experts)

        gate_up = parallel_linear(
            hidden_states, self.gate_up_proj, k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            expert_biases=self.gate_up_proj_bias,
            grouped_in=False, grouped_out=True, 
        )

        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output_ = (up + 1) * glu

        out_scattered = parallel_linear(
            gated_output_, self.down_proj, 1,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            expert_biases=self.down_proj_bias,
            grouped_in=True, grouped_out=False,
            gates=selected_weights,
        )

        next_states = out_scattered.view(batch_size, -1, self.hidden_size)
        return next_states
except Exception:
    logging.info("Failed to replace GptOssExperts")


try: 
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridMoE
    @replace_function(cls=GraniteMoeHybridMoE, fun_name='forward')
    def granite_moe_forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        router_logits = self.router.layer(layer_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.router.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(layer_input.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(selected_experts, num_experts=self.router.num_experts)

        gates, h = parallel_linear(
            layer_input, self.input_linear.weight.transpose(2, 1),
            self.router.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False, grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        layer_output = parallel_linear(
            h, self.output_linear.weight.transpose(2, 1),
            1,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True, grouped_out=False,
            gates=routing_weights
        )
        layer_output = layer_output.view(bsz, length, emb_size)
        return layer_output, router_logits
except Exception:
    logging.info("Failed to replace GraniteMoeHybridMoE")
