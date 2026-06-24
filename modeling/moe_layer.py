import torch
import torch.nn as nn

from scattermoe.mlp import GLUMLP

from .moe_gate import MoEGate
from .expert_layer import ExpertMLP

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts_per_token = config.n_experts_per_token
        self.gate = MoEGate(config)

        shared_intermediate = config.intermediate_size * config.n_shared_experts
        self.shared_experts = ExpertMLP(config, intermediate_size=shared_intermediate)

        self.routed_experts = GLUMLP(
            input_size=config.hidden_size,
            hidden_size=config.intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.n_experts_per_token,
            activation=nn.SiLU(),
        )
        self.reset_routed_expert_parameters()

    def reset_routed_expert_parameters(self):
        std = self.config.initializer_range
        nn.init.normal_(self.routed_experts.experts.weight, mean=0.0, std=std)
        nn.init.normal_(self.routed_experts.output_experts.weight, mean=0.0, std=std)

    def forward(self, hidden_states):
        x = hidden_states

        topk_idx, topk_weight = self.gate(x)
        shared_output = self.shared_experts(x)

        flat_topk_idx = topk_idx.reshape(-1, self.n_experts_per_token)
        flat_topk_weight = topk_weight.reshape(-1, self.n_experts_per_token)
        routed_output = self.routed_experts(x, flat_topk_weight, flat_topk_idx)

        out = shared_output + routed_output

        return out, topk_idx
