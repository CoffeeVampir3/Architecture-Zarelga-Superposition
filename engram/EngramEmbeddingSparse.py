"""Sparse-gradient hashed n-gram memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_I64_MAX = (1 << 63) - 1
_PRIME_1 = 10007  # per-layer multiplier seed offset (reference constant)

# Deterministic Miller-Rabin witnesses: exact for the whole 64-bit range.
_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _WITNESSES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _distinct_primes(target: int, count: int, seen: set[int]) -> list[int]:
    """``count`` distinct primes, each the next prime at/above the previous, >= target."""
    out: list[int] = []
    cursor = target - 1
    while len(out) < count:
        cursor += 1
        if _is_prime(cursor) and cursor not in seen:
            seen.add(cursor)
            out.append(cursor)
    return out


@dataclass
class EngramConfig:
    """Configuration for one per-layer Engram embedding."""

    vocab_size: int                    # token vocabulary the addresses are formed from
    d_model: int                       # backbone hidden size (the residual stream width)
    orders: tuple[int, ...] = (2, 3)   # suffix n-gram orders
    n_heads: int = 4                   # independent hash heads per order
    rows_per_head: int = 65537         # target rows/head; a distinct prime is found at/above
    dim_per_head: int = 64             # embedding width per head
    layer_id: int = 0                  # per-layer seed (base_seed = seed + 10007*layer_id)
    seed: int = 0
    pad_id: int = 0                    # left-pad token for suffix n-grams
    alpha_init: float = 0.1
    importance_weighting: bool = False
    head_norm: bool = False
    learned_gate: bool = True
    forward_noise: bool = False                # Song-Ermon mode: perturb looked-up rows in-forward
    table_dtype: torch.dtype = torch.bfloat16  # storage dtype for the hashed n-gram tables

    @property
    def heads_total(self) -> int:
        return len(self.orders) * self.n_heads

    @property
    def engram_dim(self) -> int:
        return self.heads_total * self.dim_per_head


class EngramEmbeddingSparse(nn.Module):
    """Per-layer hashed n-gram memory; sparse-gradient table (``sparse=True``)."""

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        self.orders = tuple(int(o) for o in config.orders)
        self.n_heads = int(config.n_heads)
        self.max_order = max(self.orders)
        self.pad_id = int(config.pad_id)

        base_seed = config.seed + _PRIME_1 * config.layer_id
        g = torch.Generator().manual_seed(int(base_seed))
        m_max = _I64_MAX // int(config.vocab_size)
        half = max(1, m_max // 2)
        r = torch.randint(0, half, (self.max_order,), generator=g, dtype=torch.int64)
        self.register_buffer("multipliers", r * 2 + 1)

        seen: set[int] = set()
        head_sizes: list[int] = []
        for _ in self.orders:
            head_sizes.extend(_distinct_primes(int(config.rows_per_head), self.n_heads, seen))
        self.register_buffer("primes", torch.tensor(head_sizes, dtype=torch.int64))

        offsets = [0]
        for n in head_sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64))
        self.embedding = nn.Embedding(sum(head_sizes), int(config.dim_per_head),
                                      sparse=True, dtype=config.table_dtype)

        self.value_proj = nn.Linear(config.engram_dim, config.d_model)
        self.learned_gate = bool(config.learned_gate)
        alpha = torch.full((config.d_model,), float(config.alpha_init))
        if self.learned_gate:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

        nn.init.zeros_(self.embedding.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

        self.head_norm = bool(config.head_norm)
        self.importance_weighting = bool(config.importance_weighting)
        if self.importance_weighting:
            gi = torch.Generator().manual_seed(int(base_seed) + 1)
            ri = torch.randint(0, half, (self.max_order,), generator=gi, dtype=torch.int64)
            self.register_buffer("imp_multipliers", ri * 2 + 1)
            self.imp_table = nn.Embedding(sum(head_sizes), 1, sparse=True, dtype=config.table_dtype)
            nn.init.ones_(self.imp_table.weight)

    @torch.no_grad()
    def _addresses(self, token_ids: torch.Tensor, mult: torch.Tensor,
                   position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T]`` token stream -> ``[B, T, heads_total]`` head-local indices."""
        token_ids = token_ids.to(torch.int64)
        b, t = token_ids.shape
        primes = self.primes

        shifts = [token_ids]
        for k in range(1, self.max_order):
            pad = torch.full((b, k), self.pad_id, dtype=torch.int64, device=token_ids.device)
            shifted = torch.cat([pad, token_ids[:, :t - k]], dim=1)
            if position_ids is not None:
                shifted = shifted.masked_fill(position_ids < k, self.pad_id)
            shifts.append(shifted)

        cols, col = [], 0
        for order in self.orders:
            mixed = shifts[0] * mult[0]
            for k in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifts[k] * mult[k])
            for _ in range(self.n_heads):
                cols.append(torch.remainder(mixed, primes[col]))
                col += 1
        return torch.stack(cols, dim=-1)

    @torch.no_grad()
    def addresses(self, token_ids: torch.Tensor,
                  position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T]`` token stream -> ``[B, T, heads_total]`` head-local row indices."""
        return self._addresses(token_ids, self.multipliers, position_ids)

    def forward(self, hidden_states: torch.Tensor,
                token_ids: torch.Tensor,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """hidden_states: [B, T, d_model]; token_ids: [B, T] -> [B, T, d_model]."""
        addr = self.addresses(token_ids, position_ids) + self.offsets   # [B, T, H_total]
        e = self.embedding(addr)                                        # [B, T, H_total, dim]
        if self.importance_weighting:                                   # per-head scalar reweight
            iaddr = self._addresses(token_ids, self.imp_multipliers, position_ids) + self.offsets
            e = e * self.imp_table(iaddr)                               # [B, T, H_total, dim]
        if self.head_norm:                                       # shrink-only: cap per-head L2 norm at 1
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1.0)  # bounds value_proj input; zeros stay zero
        delta = self.value_proj(e.flatten(start_dim=-2))         # [B, T, d_model]
        return hidden_states + self.alpha * delta.to(hidden_states.dtype)
