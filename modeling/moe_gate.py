import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.n_experts_per_token
        self.n_routed_experts = config.n_routed_experts
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.register_buffer('expert_biases', torch.zeros(self.n_routed_experts))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        gate_output = F.linear(hidden_states, self.weight, None)
        gate_probs = torch.sigmoid(gate_output)

        biased_scores = gate_probs + self.expert_biases
        _, topk_idx = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = gate_probs.gather(-1, topk_idx)
        topk_weight = F.normalize(topk_weight, p=1, dim=-1)

        return topk_idx.view(bsz, seq_len, -1), topk_weight.view(bsz, seq_len, -1)
