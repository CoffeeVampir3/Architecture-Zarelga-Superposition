"""Depth-wise attention residuals (AttnRes, arXiv:2603.15031)."""

import torch
import torch.nn as nn

from .zRMSNorm import ZeroCenteredRMSNorm


class DepthAttnRes(nn.Module):
    """Softmax attention over depth-wise residual sources.

    One site per transformer block plus a final site feeding the head. Site l
    mixes its source list — [embedding, engram delta(s), block deltas < l] —
    with per-token weights softmax_i(w_l · RMSNorm(v_i)). Pseudo-queries w_l
    are zero-initialized so every site starts as a uniform average of its
    sources (the paper's stability requirement). Keys are RMS-normed in fp32
    so no source dominates the softmax by magnitude; values enter raw, so a
    source's effective scale is carried by the source itself (e.g. the engram
    value_proj), not the mixing weight.
    """

    def __init__(self, hidden_size, n_sites, max_sources, eps=1e-6):
        super().__init__()
        self.queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_size)) for _ in range(n_sites)
        )
        self.key_norms = nn.ModuleList(
            ZeroCenteredRMSNorm(hidden_size, eps=eps) for _ in range(n_sites)
        )
        # Mean mixing weight per (site, source) from the last training step;
        # slots beyond a site's source count stay zero. Telemetry only.
        self.register_buffer(
            "last_alpha_mean", torch.zeros(n_sites, max_sources), persistent=False
        )

    def forward(self, site: int, values: list):
        if len(values) == 1:
            return values[0]
        query = self.queries[site]
        key_norm = self.key_norms[site]
        logits = torch.stack(
            [(key_norm(v.float()) * query).sum(-1) for v in values]
        )                                          # [n_sources, B, T] fp32
        alpha = logits.softmax(0)
        if self.training:
            with torch.no_grad():
                self.last_alpha_mean[site, : len(values)] = alpha.mean(dim=(1, 2))
        out_dtype = values[0].dtype
        mixed = values[0] * alpha[0].unsqueeze(-1).to(out_dtype)
        for i in range(1, len(values)):
            mixed = mixed + values[i] * alpha[i].unsqueeze(-1).to(out_dtype)
        return mixed
