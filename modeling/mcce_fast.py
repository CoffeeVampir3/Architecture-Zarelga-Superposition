"""
Fast Multi-target Cut Cross Entropy (MCCE), raw-token-mean specialization.

This file contains two implementations:

1. A Triton implementation for CUDA tensors, specialized for the production
   invariants:
      - hidden:     [T, D], contiguous
      - embeddings: [V, D], contiguous
      - labels:     [T, S_MAX], integer, first s[i] entries are valid labels
      - s:          [T], integer, 1 <= s[i] <= S_MAX
      - all active labels are in [0, V)
      - reduction is raw-token mean

   It never materializes [T, V] logits or [T, S_MAX, D] target embeddings.
   It saves lse[T] for backward and recomputes vocab tiles in backward.

2. A chunked PyTorch fallback with the same math. This is useful for CPU,
   debugging, and correctness tests when Triton is not installed.

The optimized raw-token-mean objective is:

    N    = sum_i s[i]
    lse_i = logsumexp_v(hidden[i] @ embeddings[v])
    loss = (sum_i s[i] * lse_i - sum_i sum_{j < s[i]} z[i, labels[i, j]]) / N

The logit gradient is:

    d loss / d z[i, v] = (s[i] / N) * softmax_i[v] - count_i(v) / N

Duplicate labels are counted multiple times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - used on non-Triton installs
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@dataclass(frozen=True)
class MCCEKernelConfig:
    block_t: int = 16
    block_v: int = 128
    block_d: int = 32
    reduce_block_v: int = 256
    num_warps_lse: int = 4
    num_warps_target: int = 4
    num_warps_finalize: int = 4
    num_warps_backward: int = 8


# -----------------------------------------------------------------------------
# Debug validation. Keep this out of the hot path unless debugging.
# -----------------------------------------------------------------------------


def _validate_inputs(
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    s: torch.Tensor,
    *,
    check_label_values: bool,
) -> None:
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [T, D], got {tuple(hidden.shape)}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [V, D], got {tuple(embeddings.shape)}")
    if hidden.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"hidden D and embeddings D differ: {hidden.shape[1]} vs {embeddings.shape[1]}"
        )
    if labels.ndim != 2:
        raise ValueError(f"labels must be [T, S_MAX], got {tuple(labels.shape)}")
    if labels.shape[0] != hidden.shape[0]:
        raise ValueError(f"labels T and hidden T differ: {labels.shape[0]} vs {hidden.shape[0]}")
    if s.shape != (hidden.shape[0],):
        raise ValueError(f"s must be [T], got {tuple(s.shape)}")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be int32 or int64, got {labels.dtype}")
    if s.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"s must be an integer tensor, got {s.dtype}")
    if hidden.device != embeddings.device or hidden.device != labels.device or hidden.device != s.device:
        raise ValueError("hidden, embeddings, labels, and s must be on the same device")
    if not torch.all(s > 0):
        raise ValueError("all sequence granularities s[i] must be positive")
    if not torch.all(s <= labels.shape[1]):
        raise ValueError("all s[i] must be <= S_MAX")
    if check_label_values:
        # Check only active label slots. This is debug-only because it syncs/does extra work.
        T, S_MAX = labels.shape
        cols = torch.arange(S_MAX, device=labels.device)[None, :]
        active = cols < s[:, None]
        active_labels = labels[active]
        if active_labels.numel() > 0:
            V = embeddings.shape[0]
            if not torch.all((active_labels >= 0) & (active_labels < V)):
                raise ValueError("active labels must all be in [0, V)")


# -----------------------------------------------------------------------------
# Reference implementation for correctness tests.
# -----------------------------------------------------------------------------


def naive_mcce_raw_token_mean(
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Naive reference. Materializes [T, V], so use only for tests."""
    T, D = hidden.shape
    V = embeddings.shape[0]
    S_MAX = labels.shape[1]

    logits = hidden.float() @ embeddings.float().T
    lse = torch.logsumexp(logits, dim=-1)

    cols = torch.arange(S_MAX, device=labels.device)[None, :]
    active = cols < s[:, None]
    safe_labels = torch.where(active, labels, torch.zeros_like(labels))
    target_logits = logits.gather(1, safe_labels)
    target_sum = (target_logits * active.float()).sum(dim=-1)

    s_f = s.float()
    total_raw = s_f.sum()
    return (s_f.mul(lse).sum() - target_sum.sum()) / total_raw


# -----------------------------------------------------------------------------
# PyTorch chunked fallback. Exact, no [T, V] materialization, but not fused.
# -----------------------------------------------------------------------------


class _MCCERawTokenMeanChunked(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        s: torch.Tensor,
        v_chunk_size: int,
    ) -> torch.Tensor:
        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        h = hidden.float()

        running_max = torch.full((T,), -float("inf"), device=hidden.device, dtype=torch.float32)
        running_sum = torch.zeros((T,), device=hidden.device, dtype=torch.float32)

        for v0 in range(0, V, v_chunk_size):
            v1 = min(v0 + v_chunk_size, V)
            z = h @ embeddings[v0:v1].float().T
            chunk_max = z.max(dim=-1).values
            new_max = torch.maximum(running_max, chunk_max)
            running_sum = running_sum * torch.exp(running_max - new_max)
            running_sum = running_sum + torch.exp(z - new_max[:, None]).sum(dim=-1)
            running_max = new_max

        lse = running_max + torch.log(running_sum)

        # Target logits: loop over S_MAX so we never build [T, S_MAX, D].
        target_sum = torch.zeros((T,), device=hidden.device, dtype=torch.float32)
        s_long = s.long()
        for j in range(S_MAX):
            active = j < s_long
            y = torch.where(active, labels[:, j], torch.zeros_like(labels[:, j]))
            e_y = embeddings[y].float()
            z_y = (h * e_y).sum(dim=-1)
            target_sum = target_sum + torch.where(active, z_y, torch.zeros_like(z_y))

        s_f = s.float()
        total_raw = s_f.sum()
        loss = (s_f.mul(lse).sum() - target_sum.sum()) / total_raw

        ctx.save_for_backward(hidden, embeddings, labels, s, lse, total_raw)
        ctx.v_chunk_size = int(v_chunk_size)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):  # type: ignore[override]
        hidden, embeddings, labels, s, lse, total_raw = ctx.saved_tensors
        v_chunk_size = ctx.v_chunk_size

        need_hidden, need_embeddings = ctx.needs_input_grad[:2]
        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        h = hidden.float()
        scale = grad_loss.float() / total_raw.float()
        s_f = s.float()
        s_long = s.long()

        dhidden = torch.zeros_like(hidden, dtype=torch.float32) if need_hidden else None
        dembeddings = torch.zeros_like(embeddings, dtype=torch.float32) if need_embeddings else None

        # Target side:
        #   dH_i -= (grad_loss / N) * sum_j E[y_ij]
        #   dE_y -= (grad_loss / N) * H_i per active target occurrence
        if need_hidden or need_embeddings:
            target_embed_sum = torch.zeros_like(hidden, dtype=torch.float32) if need_hidden else None

            for j in range(S_MAX):
                active = j < s_long
                y = torch.where(active, labels[:, j], torch.zeros_like(labels[:, j]))

                if need_hidden:
                    e_y = embeddings[y].float()
                    target_embed_sum = target_embed_sum + e_y * active[:, None].float()

                if need_embeddings:
                    src = (-scale * h) * active[:, None].float()
                    dembeddings.index_add_(0, y.reshape(-1), src.reshape(-1, D))

            if need_hidden:
                dhidden.add_(-scale * target_embed_sum)

        # Softmax side:
        #   g_soft[i, v] = (grad_loss / N) * s[i] * softmax_i[v]
        alpha = scale * s_f
        for v0 in range(0, V, v_chunk_size):
            v1 = min(v0 + v_chunk_size, V)
            e = embeddings[v0:v1].float()
            z = h @ e.T
            p = torch.exp(z - lse[:, None])
            wp = alpha[:, None] * p

            if need_hidden:
                dhidden.add_(wp @ e)
            if need_embeddings:
                dembeddings[v0:v1].add_(wp.T @ h)

        return dhidden, dembeddings, None, None, None


# -----------------------------------------------------------------------------
# Triton kernels. Defined only when Triton is importable.
# -----------------------------------------------------------------------------


if _TRITON_AVAILABLE:

    @triton.jit
    def _mcce_lse_tiles_kernel(
        H,
        E,
        PARTIAL_M,
        PARTIAL_S,
        T: tl.constexpr,
        V: tl.constexpr,
        D: tl.constexpr,
        NVB: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_v = tl.program_id(1)

        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        d_offsets = tl.arange(0, BLOCK_D)

        acc = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_idx = d0 + d_offsets
            h = tl.load(
                H + rows[:, None] * D + d_idx[None, :],
                mask=(rows[:, None] < T) & (d_idx[None, :] < D),
                other=0.0,
            )
            e = tl.load(
                E + d_idx[:, None] + vocab[None, :] * D,
                mask=(d_idx[:, None] < D) & (vocab[None, :] < V),
                other=0.0,
            )
            acc += tl.dot(h, e)

        valid = (rows[:, None] < T) & (vocab[None, :] < V)
        acc = tl.where(valid, acc, -float("inf"))

        row_valid = rows < T
        m = tl.max(acc, axis=1)
        m_safe = tl.where(row_valid, m, 0.0)
        exp_terms = tl.where(valid, tl.exp(acc - m_safe[:, None]), 0.0)
        ss = tl.sum(exp_terms, axis=1)

        out = rows * NVB + pid_v
        tl.store(PARTIAL_M + out, m_safe, mask=row_valid)
        tl.store(PARTIAL_S + out, ss, mask=row_valid)


    @triton.jit
    def _mcce_target_sum_kernel(
        H,
        E,
        LABELS,
        S,
        TARGET_SUM,
        T: tl.constexpr,
        D: tl.constexpr,
        S_MAX: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        row_valid = rows < T
        d_offsets = tl.arange(0, BLOCK_D)
        s_i = tl.load(S + rows, mask=row_valid, other=0)

        acc_targets = tl.zeros((BLOCK_T,), tl.float32)

        for j in range(0, S_MAX):
            active = row_valid & (j < s_i)
            y = tl.load(LABELS + rows * S_MAX + j, mask=active, other=0)
            dot = tl.zeros((BLOCK_T,), tl.float32)

            for d0 in range(0, D, BLOCK_D):
                d_idx = d0 + d_offsets
                h = tl.load(
                    H + rows[:, None] * D + d_idx[None, :],
                    mask=(rows[:, None] < T) & (d_idx[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
                e = tl.load(
                    E + y[:, None] * D + d_idx[None, :],
                    mask=active[:, None] & (d_idx[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
                dot += tl.sum(h * e, axis=1)

            acc_targets += tl.where(active, dot, 0.0)

        tl.store(TARGET_SUM + rows, acc_targets, mask=row_valid)


    @triton.jit
    def _mcce_finalize_kernel(
        PARTIAL_M,
        PARTIAL_S,
        TARGET_SUM,
        S,
        LSE,
        LOSS_NUM,
        T: tl.constexpr,
        NVB: tl.constexpr,
        BLOCK_T: tl.constexpr,
        REDUCE_BLOCK_V: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        row_valid = rows < T
        nb_offsets = tl.arange(0, REDUCE_BLOCK_V)

        running_m = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        running_s = tl.zeros((BLOCK_T,), tl.float32)

        for nb0 in range(0, NVB, REDUCE_BLOCK_V):
            nb = nb0 + nb_offsets
            valid = row_valid[:, None] & (nb[None, :] < NVB)
            pm = tl.load(
                PARTIAL_M + rows[:, None] * NVB + nb[None, :],
                mask=valid,
                other=-float("inf"),
            )
            ps = tl.load(
                PARTIAL_S + rows[:, None] * NVB + nb[None, :],
                mask=valid,
                other=0.0,
            )

            chunk_m = tl.max(pm, axis=1)
            new_m = tl.maximum(running_m, chunk_m)
            running_s = running_s * tl.exp(running_m - new_m) + tl.sum(
                ps * tl.exp(pm - new_m[:, None]), axis=1
            )
            running_m = new_m

        lse = running_m + tl.log(running_s)
        s_i = tl.load(S + rows, mask=row_valid, other=0).to(tl.float32)
        target = tl.load(TARGET_SUM + rows, mask=row_valid, other=0.0)
        contrib = s_i * lse - target

        tl.store(LSE + rows, lse, mask=row_valid)
        block_sum = tl.sum(tl.where(row_valid, contrib, 0.0), axis=0)
        tl.atomic_add(LOSS_NUM, block_sum)


    @triton.jit
    def _mcce_backward_exact_kernel(
        H,
        E,
        LABELS,
        S,
        LSE,
        TOTAL_RAW,
        GRAD_LOSS,
        DH,
        DE,
        T: tl.constexpr,
        V: tl.constexpr,
        D: tl.constexpr,
        S_MAX: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_v = tl.program_id(1)

        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        vocab = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        d_offsets = tl.arange(0, BLOCK_D)

        row_valid = rows < T
        vocab_valid = vocab < V

        # Recompute z tile.
        z = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_idx = d0 + d_offsets
            h = tl.load(
                H + rows[:, None] * D + d_idx[None, :],
                mask=(rows[:, None] < T) & (d_idx[None, :] < D),
                other=0.0,
            )
            e = tl.load(
                E + d_idx[:, None] + vocab[None, :] * D,
                mask=(d_idx[:, None] < D) & (vocab[None, :] < V),
                other=0.0,
            )
            z += tl.dot(h, e)

        valid = row_valid[:, None] & vocab_valid[None, :]
        z = tl.where(valid, z, -float("inf"))

        lse = tl.load(LSE + rows, mask=row_valid, other=0.0)
        s_i = tl.load(S + rows, mask=row_valid, other=0).to(tl.float32)
        scale = tl.load(GRAD_LOSS).to(tl.float32) / tl.load(TOTAL_RAW).to(tl.float32)

        # Positive softmax side: scale * s_i * p_i[v]
        p = tl.exp(z - lse[:, None])
        g = (scale * s_i)[:, None] * p
        g = tl.where(valid, g, 0.0)

        # Negative target side: subtract scale once per active target occurrence.
        for j in range(0, S_MAX):
            active = row_valid & (j < tl.load(S + rows, mask=row_valid, other=0))
            y = tl.load(LABELS + rows * S_MAX + j, mask=active, other=0)
            hit = active[:, None] & (y[:, None] == vocab[None, :]) & vocab_valid[None, :]
            g -= tl.where(hit, scale, 0.0)

        # Accumulate gradients over D blocks. g is built in fp32 for softmax
        # precision; cast it down to the native dtype of H/E at the tl.dot
        # boundary so the matmul uses the right tensor cores (bf16/fp16/fp32).
        for d0 in range(0, D, BLOCK_D):
            d_idx = d0 + d_offsets
            d_valid = d_idx < D

            h_block = tl.load(
                H + rows[:, None] * D + d_idx[None, :],
                mask=row_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            e_block = tl.load(
                E + vocab[:, None] * D + d_idx[None, :],
                mask=vocab_valid[:, None] & d_valid[None, :],
                other=0.0,
            )

            g_cast = g.to(h_block.dtype)
            dh = tl.dot(g_cast, e_block)
            de = tl.dot(tl.trans(g_cast), h_block)

            tl.atomic_add(
                DH + rows[:, None] * D + d_idx[None, :],
                dh,
                sem="relaxed",
                mask=row_valid[:, None] & d_valid[None, :],
            )
            tl.atomic_add(
                DE + vocab[:, None] * D + d_idx[None, :],
                de,
                sem="relaxed",
                mask=vocab_valid[:, None] & d_valid[None, :],
            )


class _MCCERawTokenMeanTriton(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        s: torch.Tensor,
        config: MCCEKernelConfig,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not hidden.is_cuda:
            raise RuntimeError("Triton MCCE requires CUDA tensors")

        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        BT = int(config.block_t)
        BV = int(config.block_v)
        BD = int(config.block_d)
        RBV = int(config.reduce_block_v)
        NVB = triton.cdiv(V, BV)

        partial_m = torch.empty((T, NVB), device=hidden.device, dtype=torch.float32)
        partial_s = torch.empty((T, NVB), device=hidden.device, dtype=torch.float32)
        target_sum = torch.empty((T,), device=hidden.device, dtype=torch.float32)
        lse = torch.empty((T,), device=hidden.device, dtype=torch.float32)
        loss_num = torch.zeros((), device=hidden.device, dtype=torch.float32)
        total_raw = s.float().sum()

        grid_lse = (triton.cdiv(T, BT), NVB)
        _mcce_lse_tiles_kernel[grid_lse](
            hidden,
            embeddings,
            partial_m,
            partial_s,
            T,
            V,
            D,
            NVB,
            BLOCK_T=BT,
            BLOCK_V=BV,
            BLOCK_D=BD,
            num_warps=config.num_warps_lse,
        )

        grid_rows = (triton.cdiv(T, BT),)
        _mcce_target_sum_kernel[grid_rows](
            hidden,
            embeddings,
            labels,
            s,
            target_sum,
            T,
            D,
            S_MAX,
            BLOCK_T=BT,
            BLOCK_D=BD,
            num_warps=config.num_warps_target,
        )

        _mcce_finalize_kernel[grid_rows](
            partial_m,
            partial_s,
            target_sum,
            s,
            lse,
            loss_num,
            T,
            NVB,
            BLOCK_T=BT,
            REDUCE_BLOCK_V=RBV,
            num_warps=config.num_warps_finalize,
        )

        loss = loss_num / total_raw

        ctx.save_for_backward(hidden, embeddings, labels, s, lse, total_raw)
        ctx.config = config
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):  # type: ignore[override]
        hidden, embeddings, labels, s, lse, total_raw = ctx.saved_tensors
        config: MCCEKernelConfig = ctx.config

        need_hidden, need_embeddings = ctx.needs_input_grad[:2]
        # Current exact Triton kernel computes both gradients. Return None if not needed.
        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        BT = int(config.block_t)
        BV = int(config.block_v)
        BD = int(config.block_d)

        dhidden = torch.zeros_like(hidden, dtype=torch.float32)
        dembeddings = torch.zeros_like(embeddings, dtype=torch.float32)

        grid = (triton.cdiv(T, BT), triton.cdiv(V, BV))
        _mcce_backward_exact_kernel[grid](
            hidden,
            embeddings,
            labels,
            s,
            lse,
            total_raw,
            grad_loss,
            dhidden,
            dembeddings,
            T,
            V,
            D,
            S_MAX,
            BLOCK_T=BT,
            BLOCK_V=BV,
            BLOCK_D=BD,
            num_warps=config.num_warps_backward,
        )

        return (
            dhidden if need_hidden else None,
            dembeddings if need_embeddings else None,
            None,
            None,
            None,
        )


# -----------------------------------------------------------------------------
# Public API.
# -----------------------------------------------------------------------------


def mcce_raw_token_mean(
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    s: torch.Tensor,
    *,
    use_triton: bool = True,
    config: Optional[MCCEKernelConfig] = None,
    v_chunk_size: int = 8192,
    debug: bool = False,
    check_label_values: bool = False,
) -> torch.Tensor:
    """Exact MCCE loss specialized to raw-token mean.

    Active labels are labels[i, :s[i]]. Slots labels[i, s[i]:] are ignored and
    may contain any integer value.

    For production, pass contiguous CUDA tensors and keep debug=False.
    """
    if config is None:
        config = MCCEKernelConfig()

    if debug:
        _validate_inputs(hidden, embeddings, labels, s, check_label_values=check_label_values)

    # Production path expects row-major contiguous tensors. Making them contiguous
    # here keeps the API safe; if copies matter, enforce contiguity at the caller.
    hidden = hidden.contiguous()
    embeddings = embeddings.contiguous()
    labels = labels.contiguous()
    s = s.contiguous()

    if use_triton and _TRITON_AVAILABLE and hidden.is_cuda:
        return _MCCERawTokenMeanTriton.apply(hidden, embeddings, labels, s, config)

    return _MCCERawTokenMeanChunked.apply(hidden, embeddings, labels, s, int(v_chunk_size))


# -----------------------------------------------------------------------------
# Correctness check.
# -----------------------------------------------------------------------------


def self_test(
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    use_triton: bool = True,
    seed: int = 123,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    T, D, V, S_MAX = 17, 64, 1009, 4

    hidden = torch.randn(T, D, device=device, dtype=dtype, requires_grad=True)
    embeddings = torch.randn(V, D, device=device, dtype=dtype, requires_grad=True)
    s = torch.randint(1, S_MAX + 1, (T,), device=device, dtype=torch.int32)

    labels = torch.randint(0, V, (T, S_MAX), device=device, dtype=torch.int64)
    # Deliberately make some duplicate labels to test multiset behavior.
    if S_MAX >= 2:
        labels[:, 1] = labels[:, 0]

    h_ref = hidden.detach().clone().requires_grad_(True)
    e_ref = embeddings.detach().clone().requires_grad_(True)
    loss_ref = naive_mcce_raw_token_mean(h_ref, e_ref, labels, s)
    loss_ref.backward()

    h_fast = hidden.detach().clone().requires_grad_(True)
    e_fast = embeddings.detach().clone().requires_grad_(True)
    loss_fast = mcce_raw_token_mean(
        h_fast,
        e_fast,
        labels,
        s,
        use_triton=use_triton,
        v_chunk_size=128,
        debug=True,
        check_label_values=True,
    )
    loss_fast.backward()

    print(f"device={device} dtype={dtype} triton_requested={use_triton} triton_available={_TRITON_AVAILABLE}")
    print(f"loss_ref  = {loss_ref.item():.8f}")
    print(f"loss_fast = {loss_fast.item():.8f}")
    print(f"loss_abs_diff = {(loss_ref - loss_fast).abs().item():.3e}")
    print(f"dH_max_abs_diff = {(h_ref.grad.float() - h_fast.grad.float()).abs().max().item():.3e}")
    print(f"dE_max_abs_diff = {(e_ref.grad.float() - e_fast.grad.float()).abs().max().item():.3e}")

    atol = 3e-4 if dtype in (torch.float16, torch.bfloat16) else 2e-5
    assert torch.allclose(loss_ref, loss_fast, atol=atol, rtol=atol)
    assert torch.allclose(h_ref.grad.float(), h_fast.grad.float(), atol=5e-4, rtol=5e-4)
    assert torch.allclose(e_ref.grad.float(), e_fast.grad.float(), atol=5e-4, rtol=5e-4)


if __name__ == "__main__":
    self_test(use_triton=True)
