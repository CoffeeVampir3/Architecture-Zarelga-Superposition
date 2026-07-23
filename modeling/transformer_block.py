import torch
import torch.nn as nn

from .moe_layer import MoELayer
from .expert_layer import ExpertMLP
from .zRMSNorm import ZeroCenteredRMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_type, input_layernorm=True, is_dense=False):
        super().__init__()
        self.input_layernorm = (
            ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if input_layernorm
            else nn.Identity()
        )
        self.self_attn = layer_type(config)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_dense = is_dense
        # Under AttnRes the block is a value source, so its caller needs the
        # residual branch delta rather than the ordinary ``input + delta``
        # stream output.  This is fixed for the lifetime of the model and is
        # therefore specialized away by the per-block torch.compile graph.
        self.return_residual_delta = bool(getattr(config, "attn_res", False))
        self.mlp = (
            ExpertMLP(config, intermediate_size=config.dense_intermediate_size)
            if is_dense
            else MoELayer(config)
        )

    def forward(
        self,
        hidden_states,
        position_ids=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        block_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]
        attention_delta = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )  # [batch_size, seq_len, hidden_size]
        hidden_states = block_input + attention_delta  # [batch_size, seq_len, hidden_size]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_dense:
            mlp_delta = self.mlp(hidden_states)
            topk_idx = None
        else:
            mlp_delta, topk_idx = self.mlp(hidden_states)

        if self.return_residual_delta:
            # Algebraically, (block_input + attention_delta + mlp_delta)
            # - block_input == attention_delta + mlp_delta.  Accumulate in the
            # fp32 residual-stream dtype; Inductor fuses the cast and add into
            # one pointwise output kernel.
            hidden_states = attention_delta.to(block_input.dtype) + mlp_delta
        else:
            hidden_states = residual + mlp_delta

        return hidden_states, topk_idx
