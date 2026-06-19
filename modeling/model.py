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
        self.tie_word_embeddings = config.tie_word_embeddings

        self.reset_parameters()
        if self.tie_word_embeddings:
            self.tie_weights()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.config.initializer_range)
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

    def _run_layers(self, x, position_ids, s_value, cu_seqlens, unpad_indices, max_seqlen):
        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(
                x,
                position_ids=position_ids,
                s_value=s_value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                unpad_indices=unpad_indices,
            )
            all_topk_indices.append(topk_idx)
        return x, all_topk_indices

    def forward(self, x, position_ids=None, s_value=1, cu_seqlens=None, unpad_indices=None, max_seqlen=None):
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, position_ids, s_value, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        x = self.output_layer(x)
        return x, all_topk_indices

    def headless_forward(self, x, position_ids=None, s_value=1, cu_seqlens=None, unpad_indices=None, max_seqlen=None):
        x = self._embed_tokens(x)
        x, all_topk_indices = self._run_layers(x, position_ids, s_value, cu_seqlens, unpad_indices, max_seqlen)
        x = self.norm(x)
        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
