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
        # sparse=True: backward produces a COO gradient over the touched rows only, so
        # the table rides GramReaperSparse (per-row RMSProp + unit-sphere projection)
        # instead of the dense MuonMD/Adam optimizer. See build_muonmd_optimizer.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, sparse=True)

        # Per-layer sliding window from config.attention_window_pattern (one entry
        # per layer; None => full causal/global). partial binds each layer's window
        # into the GatedAttention factory that TransformerBlock instantiates.
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, partial(GatedAttention, window_size=window), True)
                for window in config.attention_window_pattern
            ]
        )

        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = config.tie_word_embeddings

        # Engram conditional-memory branches at selected layers (pre-attention residual;
        # applied in _run_layers before each chosen block). The hashed n-gram table is
        # sparse-gradient and zero-init, so each branch is exactly identity at init.
        self.engram_layers = tuple(int(i) for i in config.engram.layers)
        if self.engram_layers:
            self.engrams = nn.ModuleDict({
                str(layer_idx): self._build_engram(config, layer_idx)
                for layer_idx in self.engram_layers
            })
        else:
            self.engrams = None

        self.reset_parameters()
        if self.tie_word_embeddings:
            self.tie_weights()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.config.initializer_range)
        # Seed the embedding rows onto the unit L2 sphere. GramReaperSparse only ever
        # renormalizes the rows it touches, so a never-hit (rare) token would otherwise
        # keep its slightly-off init norm forever; projecting here makes the unit-norm
        # constraint hold table-wide from step 0 (Hägele et al. 2026: keep embedding /
        # LM-head rows at unit L2 norm throughout).
        with torch.no_grad():
            self.embedding.weight.div_(
                self.embedding.weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
            )
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=self.config.initializer_range)

    def tie_weights(self):
        # Alias the classifier weight onto the embedding so both share one
        # tensor. get_classifier_weights() then returns the embedding matrix.
        self.output_layer.weight = self.embedding.weight

    def _embed_tokens(self, x):
        if x.ndim == 3:
            bag = self.embedding(x)
            emb = bag.float().mean(dim=-2).to(bag.dtype)
        else:
            emb = self.embedding(x)
        # Paper's sqrt(d) upscale gives an RMS of 1 going into the model -- but it is
        # only safe when embeddings are UNTIED. With tying (the shared input/output
        # weight) it makes the residual stream dominated by the current token, so the
        # tied head self-predicts the current token and is confidently wrong on the
        # next-token objective at init. We therefore apply it only when untied; the
        # 1/sqrt(d) matrix init (which sets the MuonMD sphere radii) is kept either
        # way, and the first RMSNorm renormalizes activations regardless.
        if not self.tie_word_embeddings:
            emb = emb * (self.config.hidden_size ** 0.5)
        return emb

    def _build_engram(self, config, layer_idx):
        engram = EngramEmbeddingSparse(EngramConfig(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            orders=tuple(config.engram.orders),
            n_heads=config.engram.n_heads,
            rows_per_head=config.engram.rows_per_head,
            dim_per_head=config.engram.dim_per_head,
            layer_id=layer_idx,
            pad_id=config.pad_token_id,
            alpha_init=config.engram.alpha_init,
            importance_weighting=config.engram.importance_weighting,
            head_norm=config.engram.head_norm,
            learned_gate=config.engram.learned_gate,
        ))
        # Match the repo's 1/sqrt(d) matrix init so value_proj's MuonMD sphere radius is
        # correct; the table stays zero-init (set in EngramEmbeddingSparse), so the
        # branch is still exactly identity at step 0.
        nn.init.normal_(engram.value_proj.weight, mean=0.0, std=config.initializer_range)
        return engram

    def _run_layers(self, x, token_ids, position_ids, cu_seqlens, unpad_indices, max_seqlen):
        all_topk_indices = []
        # Engram needs the raw token IDs; only the 2-D (non-TST) path supplies them.
        apply_engram = self.engrams is not None and token_ids is not None and token_ids.dim() == 2
        for layer_idx, layer in enumerate(self.layers):
            if apply_engram and str(layer_idx) in self.engrams:
                x = self.engrams[str(layer_idx)](x, token_ids, position_ids)
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
        token_ids = x
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, token_ids, position_ids, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        x = self.output_layer(x)
        return x, all_topk_indices

    def headless_forward(self, x, position_ids=None, cu_seqlens=None, unpad_indices=None, max_seqlen=None):
        token_ids = x
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, token_ids, position_ids, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
