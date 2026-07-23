"""Depth-wise attention residuals (AttnRes, arXiv:2603.15031)."""

import torch
import torch.nn as nn


class DepthAttnRes(nn.Module):
    """Softmax attention over depth-wise residual sources.

    One site per transformer block plus a final site feeding the head. Site l
    mixes its source list — [embedding, engram delta(s), block deltas < l] —
    with per-token weights softmax_i(w_l · v̂_i), where v̂ is the RMS-normalized
    source. Pseudo-queries w_l are zero-initialized so every site starts as a
    uniform average of its sources (the paper's stability requirement); keys
    are normalized in fp32 so no source dominates by magnitude; values enter
    raw, so a source's effective scale is carried by the source itself (e.g.
    the engram value_proj).

    The paper's per-site RMSNorm gain is folded into the query — for any gain
    g, w·(g ⊙ v̂) ≡ (w ⊙ g)·v̂, so a learned per-site gain is a redundant
    reparameterization of w. That makes the logits site-independent up to a
    matvec: ``source_logits`` normalizes each source once, when it enters the
    pool, and scores it against every site's query in one pass ([B,T,n_sites],
    tiny, retained). ``forward`` then only pays the weighted-sum read.
    """

    def __init__(self, hidden_size, n_sites, max_sources, eps=1e-6):
        super().__init__()
        self.eps = eps
        # One 1-D parameter per site (kept unstacked so the optimizer routes
        # them to the Adam scalar group, not Muon).
        self.queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_size)) for _ in range(n_sites)
        )
        # Mean mixing weight per (site, source) from the last training step;
        # slots beyond a site's source count stay zero. Telemetry only.
        self.register_buffer(
            "last_alpha_mean", torch.zeros(n_sites, max_sources), persistent=False
        )

    def source_logits(self, v):
        """[B, T, d] source -> [B, T, n_sites] logits against every site query.

        One fp32 normalization + matvec per source for the whole network;
        checkpointed by the caller so only the [B, T, n_sites] result is
        retained.
        """
        vf = v.float()
        vhat = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + self.eps)
        queries = torch.stack(list(self.queries))          # [n_sites, d] fp32
        return vhat @ queries.t()                          # [B, T, n_sites]

    def forward(self, site: int, values: list, logits: list):
        """Mix `values` with the site's softmax weights.

        `logits[i]` is `source_logits(values[i])`; only column `site` is used.
        """
        if len(values) == 1:
            return values[0]
        site_logits = torch.stack([lg[..., site] for lg in logits])  # [n, B, T]
        alpha = site_logits.softmax(0)
        if self.training:
            with torch.no_grad():
                self.last_alpha_mean[site, : len(values)] = alpha.mean(dim=(1, 2))
        out_dtype = values[0].dtype
        mixed = values[0] * alpha[0].unsqueeze(-1).to(out_dtype)
        for i in range(1, len(values)):
            mixed = mixed + values[i] * alpha[i].unsqueeze(-1).to(out_dtype)
        return mixed
