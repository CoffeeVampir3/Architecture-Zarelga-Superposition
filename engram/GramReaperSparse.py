"""Per-row RMSProp for sparse embedding-table gradients."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class GramReaperSparse(Optimizer):
    """Per-row RMSProp consuming a sparse (COO) embedding-table gradient."""

    def __init__(self, params, lr: float = 1e-2, beta: float = 0.9,
                 eps: float = 1e-10, unit_norm: bool = False):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps, unit_norm=unit_norm))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta, eps = group["lr"], group["beta"], group["eps"]
            unit_norm = group["unit_norm"]
            noise_std = group.get("noise_std", 0.0)  # annealed memory corruption (0 = off)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError("GramReaperSparse supports 2-D params [n_rows, dim] only")
                grad = p.grad
                if not grad.is_sparse:
                    raise ValueError(
                        "GramReaperSparse expects a sparse gradient; build the table with "
                        "nn.Embedding(..., sparse=True) (see EngramEmbeddingSparse). For a "
                        "dense gradient use GramReaperDense."
                    )
                grad = grad.coalesce()
                idx = grad.indices()[0]                  # [nnz_unique]
                g = grad.values()                        # [nnz_unique, dim]
                if idx.numel() == 0:
                    continue

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                    state["t"] = 0
                state["t"] += 1
                t = state["t"]
                v = state["v"]

                g = g.float()
                v[idx] = beta * v[idx] + (1 - beta) * (g * g).sum(dim=1)
                v_hat = v[idx] / (1 - beta ** t)         # bias correction
                step = lr / (v_hat.sqrt() + eps)         # per-row scalar step size
                row = p.data[idx].float() - step.unsqueeze(1) * g
                if unit_norm:
                    row = row / row.norm(dim=1, keepdim=True).clamp_min(eps)
                if noise_std > 0.0:
                    # Relative corruption: per-row std is `noise_std` *as a fraction
                    # of that row's norm*, so the noise/signal ratio is scale-invariant
                    # instead of an absolute kick that an un-normalized row could dilute
                    # or amplify. Touched rows only; resampled each step.
                    row_norm = row.norm(dim=1, keepdim=True)
                    row = row + (noise_std * row_norm) * torch.randn_like(row)
                    if unit_norm:  # keep the stored row on the unit sphere
                        row = row / row.norm(dim=1, keepdim=True).clamp_min(eps)
                p.data[idx] = row.to(p.dtype)
        return loss
