import torch
import torch.nn as nn

from .transformer_block import TransformerBlock

from .attention import GatedAttention
from .zRMSNorm import ZeroCenteredRMSNorm

class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, GatedAttention, True)
                for _ in range(config.transformer_depth)
            ]
        )

        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=self.config.initializer_range)

    def _run_layers(self, x, position_ids, cu_seqlens, unpad_indices, max_seqlen):
        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(
                x,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                unpad_indices=unpad_indices,
            )
            all_topk_indices.append(topk_idx)
        return x, all_topk_indices

    def forward(self, x, position_ids=None, cu_seqlens=None, unpad_indices=None, max_seqlen=None):
        x = self.embedding(x)
        x, all_topk_indices = self._run_layers(x, position_ids, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        x = self.output_layer(x)
        return x, all_topk_indices

    def headless_forward(self, x, position_ids=None, cu_seqlens=None, unpad_indices=None, max_seqlen=None):
        x = self.embedding(x)
        x, all_topk_indices = self._run_layers(x, position_ids, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
