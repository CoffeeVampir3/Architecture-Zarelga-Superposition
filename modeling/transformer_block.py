import torch
import torch.nn as nn

from .moe_layer import MoELayer
from .zRMSNorm import ZeroCenteredRMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_type, input_layernorm=True):
        super().__init__()
        self.input_layernorm = (
            ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if input_layernorm
            else nn.Identity()
        )
        self.self_attn = layer_type(config)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MoELayer(config)

    def forward(
        self,
        hidden_states,
        position_ids=None,
        cu_seqlens=None,
        max_seqlen=None,
        unpad_indices=None,
    ):
        # Input shape: [batch_size, seq_len, hidden_size]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            unpad_indices=unpad_indices,
        )  # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states  # [batch_size, seq_len, hidden_size]

        # MoE sublayer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, topk_idx = self.mlp(hidden_states, unpad_indices=unpad_indices)
        hidden_states = residual + hidden_states

        return hidden_states, topk_idx
