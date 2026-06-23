"""GramReaperSparse — per-row RMSProp over a SPARSE embedding-table gradient.

Sparse counterpart of :class:`engram.prod.GramReaperDense`. It consumes the COO
sparse gradient that ``nn.Embedding(..., sparse=True)`` produces on backward (see
:class:`engram.prod.EngramEmbeddingSparse`), so the touched rows arrive *as data*
— there is no full-table scan and no ``abs()`` copy needed to find them. The cost
(time and transient memory) is therefore proportional to the rows touched, not to
the table size, which is the whole point of sparse-row optimisation.

The per-row math is identical to the dense path, so results are directly
comparable; only the route to ``(idx, g)`` differs::

    grad = p.grad.coalesce()       # sum duplicate row hits -> one entry per unique row
    idx  = grad.indices()[0]       # [nnz_unique]
    g    = grad.values()           # [nnz_unique, dim]

Coalescing sums the contributions of rows hit by multiple tokens, which is exactly
what autograd accumulates into a dense row — so ``(idx, g)`` here equals
``(active_rows, grad[active_rows])`` in the dense variant, element for element.

Persistent state is still one scalar per row (``[n_rows]``, allocated once); only
``v[idx]`` is read or written each step::

    v     <- beta * v + (1 - beta) * sum(g**2)
    v_hat  = v / (1 - beta**t)                      # bias correction (global t)
    w     <- w - lr * g / (sqrt(v_hat) + eps)

With ``unit_norm=True`` each touched row is projected back onto the unit L2 sphere
after its update (``w <- w / ||w||``). This implements the magnitude-direction
recipe for the embedding / LM-head (Hägele et al. 2026): keep those rows at unit
L2 norm throughout, with no weight decay -- the per-row RMSProp moves the direction
and the projection fixes the magnitude. Because only the rows in the sparse gradient
are renormalised, a never-touched row keeps its norm, so the table must be seeded on
the sphere at init (unit-norm rows) for the constraint to hold table-wide. Off by
default, so zero-init tables (e.g. Engram's) keep their zero rows untouched.
"""

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
                # Coalesce -> one entry per unique row (duplicate hits summed).
                grad = grad.coalesce()
                idx = grad.indices()[0]                  # [nnz_unique]
                g = grad.values()                        # [nnz_unique, dim]
                if idx.numel() == 0:
                    continue

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros(p.shape[0], dtype=p.dtype, device=p.device)
                    state["t"] = 0
                state["t"] += 1
                t = state["t"]
                v = state["v"]

                # Touched rows only; advanced indexing returns a copy, so assign back.
                v[idx] = beta * v[idx] + (1 - beta) * (g * g).sum(dim=1)
                v_hat = v[idx] / (1 - beta ** t)         # bias correction
                step = lr / (v_hat.sqrt() + eps)         # per-row scalar step size
                row = p.data[idx] - step.unsqueeze(1) * g
                if unit_norm:
                    # Project each updated row back onto the unit L2 sphere (paper's
                    # "keep embeddings / LM-head rows at unit L2 norm throughout").
                    # Only the touched rows are renormalised -- an untouched row did
                    # not move, so it is still on the sphere from a previous step (or
                    # from a unit-norm init). Magnitude-direction decoupling: the per-
                    # row RMSProp sets the direction, the sphere fixes the magnitude,
                    # which is why these groups carry no weight decay.
                    row = row / row.norm(dim=1, keepdim=True).clamp_min(eps)
                p.data[idx] = row
        return loss
