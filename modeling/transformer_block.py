import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )  # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states  # [batch_size, seq_len, hidden_size]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_dense:
            hidden_states = checkpoint(self.mlp, hidden_states, use_reentrant=False)
            topk_idx = None
        else:
            hidden_states, topk_idx = checkpoint(self.mlp, hidden_states, use_reentrant=False)
        hidden_states = residual + hidden_states

        return hidden_states, topk_idx
