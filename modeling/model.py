from functools import partial

import torch
import torch.nn as nn

from .transformer_block import TransformerBlock

from .attention import GatedAttention
from .zRMSNorm import ZeroCenteredRMSNorm

from engram.EngramEmbeddingSparse import EngramConfig, EngramEmbeddingSparse

class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, sparse=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    partial(GatedAttention, window_size=window),
                    True,
                    is_dense=(layer_idx < config.first_k_dense_replace),
                )
                for layer_idx, window in enumerate(config.attention_window_pattern)
            ]
        )

        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = config.tie_word_embeddings

        # config.engram.layers maps engram id -> layers it's injected at; one module
        # per id, weight-tied across its layer group. ModuleDict is keyed by str(id).
        self.engram_map = {
            int(eid): tuple(int(i) for i in group)
            for eid, group in config.engram.layers.items()
        }
        if self.engram_map:
            self.engrams = nn.ModuleDict({
                str(eid): self._build_engram(config, eid)
                for eid in sorted(self.engram_map)
            })
            self._engrams_at = {}  # layer_idx -> ModuleDict keys, in id order
            for eid in sorted(self.engram_map):
                for layer_idx in self.engram_map[eid]:
                    self._engrams_at.setdefault(layer_idx, []).append(str(eid))
        else:
            self.engrams = None
            self._engrams_at = {}

        self.reset_parameters()
        if self.tie_word_embeddings:
            self.tie_weights()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.config.initializer_range)
        with torch.no_grad():
            self.embedding.weight.div_(
                self.embedding.weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
            )
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=self.config.initializer_range)

    def tie_weights(self):
        self.output_layer.weight = self.embedding.weight

    def _embed_tokens(self, x):
        emb = self.embedding(x)
        if not self.tie_word_embeddings:
            emb = emb * (self.config.hidden_size ** 0.5)
        return emb

    def _build_engram(self, config, engram_id):
        engram = EngramEmbeddingSparse(EngramConfig(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            orders=tuple(config.engram.orders),
            n_heads=config.engram.n_heads,
            rows_per_head=config.engram.rows_per_head,
            dim_per_head=config.engram.dim_per_head,
            layer_id=engram_id,
            pad_id=config.pad_token_id,
            alpha_init=config.engram.alpha_init,
            importance_weighting=config.engram.importance_weighting,
            head_norm=config.engram.head_norm,
            gate_mode=config.engram.gate_mode,
        ))
        nn.init.normal_(engram.value_proj.weight, mean=0.0, std=config.initializer_range)
        if getattr(engram, "key_proj", None) is not None:
            nn.init.normal_(engram.key_proj.weight, mean=0.0, std=config.initializer_range)
        return engram

    def _run_layers(self, x, token_ids, position_ids, cu_seqlens, max_seqlen):
        all_topk_indices = []
        apply_engram = self.engrams is not None and token_ids is not None and token_ids.dim() == 2
        # Memory readouts depend only on the token stream, so an engram tied across
        # several layers reads its table once; only the gate runs per layer.
        engram_reads = {}
        for layer_idx, layer in enumerate(self.layers):
            if apply_engram:
                for key in self._engrams_at.get(layer_idx, ()):
                    engram = self.engrams[key]
                    if key not in engram_reads:
                        engram_reads[key] = engram.read(token_ids, position_ids)
                    x = engram.inject(x, *engram_reads[key])
            x, topk_idx = layer(
                x,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            all_topk_indices.append(topk_idx)
        return x, all_topk_indices

    def forward(self, x, position_ids=None, cu_seqlens=None, max_seqlen=None):
        token_ids = x
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, token_ids, position_ids, cu_seqlens, max_seqlen)
        x = self.norm(x)
        x = self.output_layer(x)
        return x, all_topk_indices

    def headless_forward(self, x, position_ids=None, cu_seqlens=None, max_seqlen=None):
        token_ids = x
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, token_ids, position_ids, cu_seqlens, max_seqlen)
        x = self.norm(x)
        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
