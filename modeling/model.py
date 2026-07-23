from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .transformer_block import TransformerBlock

from .attention import GatedAttention
from .attn_res import DepthAttnRes
from .zRMSNorm import ZeroCenteredRMSNorm

from engram.EngramEmbeddingSparse import EngramConfig, EngramEmbeddingSparse

class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_res = bool(getattr(config, "attn_res", False))
        # The token table is also the LM classifier when weight tying is enabled.
        # Classifier loss produces a dense [vocab, hidden] gradient, so this table
        # must use the ordinary dense optimizer path. Engram tables remain sparse.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, sparse=False)

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

        if self.attn_res:
            self.depth_mixer = DepthAttnRes(
                config.hidden_size,
                n_sites=len(self.layers) + 1,  # one per block + the final head mix
                max_sources=1 + len(self.engram_map) + len(self.layers),
                eps=config.rms_norm_eps,
            )
        else:
            self.depth_mixer = None

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

    def compile_blockwise(self, **compile_kwargs):
        """Compile each transformer block and engram read/inject as separate
        graphs rather than the whole forward. A single joint graph puts every
        checkpointed block's backward recompute into one inductor schedule,
        whose buffer live-ranges overlap and multiply backward peak memory
        (~2.5x measured); per-block graphs keep autograd's sequential
        recompute-then-free, at equal or better step time.
        """
        for layer in self.layers:
            layer.compile(**compile_kwargs)
        if self.engrams is not None:
            for engram in self.engrams.values():
                engram.read = torch.compile(engram.read, **compile_kwargs)
                engram.inject = torch.compile(engram.inject, **compile_kwargs)
        if self.depth_mixer is not None:
            # Site index and source count are compile-time constants per call
            # site, so each of the depth+1 mixes gets its own small graph.
            self.depth_mixer.compile(**compile_kwargs)
            self.depth_mixer.source_logits = torch.compile(
                self.depth_mixer.source_logits, **compile_kwargs
            )

    def _embed_tokens(self, x):
        # The raw shared rows are free to learn useful classifier magnitudes.
        # Normalize only their input-facing view, preserving the previous unit-row
        # embedding geometry and the sqrt(d) AttnRes value scale. For large packed
        # batches it is cheaper to normalize V rows once than the same rows once per
        # token; autoregressive inference takes the gathered-row path instead.
        weight = self.embedding.weight
        if x.numel() >= weight.shape[0]:
            normalized_weight = F.normalize(weight.float(), dim=-1, eps=1e-12)
            emb = F.embedding(x, normalized_weight.to(weight.dtype))
        else:
            emb = self.embedding(x)
            emb = F.normalize(emb.float(), dim=-1, eps=1e-12).to(emb.dtype)
        return emb * (self.config.hidden_size ** 0.5)

    def _build_engram(self, config, engram_id):
        # Under AttnRes the readout is a depth-attention value source; the
        # context gate (and its key projection) is never used, so build the
        # module in the parameter-free fixed_alpha mode. read() then returns
        # (delta, None) without computing a key.
        gate_mode = "fixed_alpha" if self.attn_res else config.engram.gate_mode
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
            gate_mode=gate_mode,
        ))
        nn.init.normal_(engram.value_proj.weight, mean=0.0, std=config.initializer_range)
        if getattr(engram, "key_proj", None) is not None:
            nn.init.normal_(engram.key_proj.weight, mean=0.0, std=config.initializer_range)
        return engram

    def _mixed_block_step(self, layer_idx, values, logits, position_ids, cu_seqlens, max_seqlen):
        """One AttnRes site: mix the sources, run the block, emit the block delta.

        Checkpointed as a unit so the mixed input h is recomputed in backward —
        only the shared source list (and its tiny precomputed logits) stays
        live across the network.
        """
        h = self.depth_mixer(layer_idx, values, logits)
        delta, topk_idx = self.layers[layer_idx](
            h,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return delta, topk_idx

    def _run_layers_attn_res(self, x, token_ids, position_ids, cu_seqlens, max_seqlen):
        all_topk_indices = []
        apply_engram = self.engrams is not None and token_ids is not None and token_ids.dim() == 2
        # Engram deltas are depth-attention value sources, not stream
        # injections: the tied readout is computed once and every later site
        # retrieves it through its softmax weight. Site 0 sees only the
        # embedding (the ablation-accepted "layer 0 is embedding-only").
        # Each source is normalized and scored against every site's query once
        # when it enters the pool (checkpointed: only the [B, T, n_sites]
        # logits are retained).
        sources = [x]
        if apply_engram:
            for key in sorted(self.engrams.keys(), key=int):
                delta, _ = checkpoint(self.engrams[key].read, token_ids, position_ids,
                                      use_reentrant=False)
                sources.append(delta.to(x.dtype))
        logits = [checkpoint(self.depth_mixer.source_logits, v, use_reentrant=False)
                  for v in sources]
        n_static = len(sources)  # embedding + engram sources
        for layer_idx in range(len(self.layers)):
            if layer_idx == 0:
                n_site = 1
            else:
                n_site = n_static + layer_idx
            delta, topk_idx = checkpoint(
                self._mixed_block_step, layer_idx,
                sources[:n_site], logits[:n_site],
                position_ids, cu_seqlens, max_seqlen,
                use_reentrant=False,
            )
            sources.append(delta)
            logits.append(checkpoint(self.depth_mixer.source_logits, delta,
                                     use_reentrant=False))
            all_topk_indices.append(topk_idx)
        x = checkpoint(self.depth_mixer, len(self.layers), sources, logits,
                       use_reentrant=False)
        return x, all_topk_indices

    def _run_layers(self, x, token_ids, position_ids, cu_seqlens, max_seqlen):
        if self.attn_res:
            return self._run_layers_attn_res(
                x, token_ids, position_ids, cu_seqlens, max_seqlen
            )
        all_topk_indices = []
        apply_engram = self.engrams is not None and token_ids is not None and token_ids.dim() == 2
        # Memory readouts depend only on the token stream, so an engram tied across
        # several layers reads its table once; only the gate runs per layer. Both
        # halves are checkpointed: the read's gather/norm/projection intermediates
        # and the gate's fp32 intermediates are large but cheap to recompute, so
        # only the read outputs (delta, normed_key) stay live across the network,
        # shared by every injection site.
        engram_reads = {}
        for layer_idx, layer in enumerate(self.layers):
            if apply_engram:
                for key in self._engrams_at.get(layer_idx, ()):
                    engram = self.engrams[key]
                    if key not in engram_reads:
                        engram_reads[key] = checkpoint(
                            engram.read, token_ids, position_ids,
                            use_reentrant=False,
                        )
                    x = checkpoint(engram.inject, x, *engram_reads[key],
                                   use_reentrant=False)
            # The whole block (attention included) is checkpointed — only the
            # block input stays live per layer; the block itself must not nest
            # its own checkpoint calls.
            x, topk_idx = checkpoint(
                layer,
                x,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                use_reentrant=False,
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
