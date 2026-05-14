"""Fast Multi-target Cut Cross Entropy (MCCE), v2 — performance-oriented rewrite.

Same math as mcce_fast.py (raw-token-mean reduction), same public API shape.
The changes are all kernel-engineering, not algorithmic:

  1. Fused single-pass forward.
     v1 launches three kernels: LSE-tiles (writes [T, V/BV] partial buffers),
     target-sum (per-row dot products), and finalize (reduces partials → lse,
     atomic-adds to scalar loss). v2 collapses these into one kernel with grid
     (T/BT,) that does online softmax in registers, then the target-sum loop
     against the same H tile already cached. The [T, V/BV] HBM intermediates
     vanish.

  2. Atomic-contention-free backward.
     v1's backward has grid (T/BT, V/BV) and atomic_adds into both DH and DE
     from every block. At production scales that's ~256k blocks all contending
     on shared rows of two fp32 buffers. v2 splits backward into two kernels:
       - DH kernel: grid (T/BT,), loops V internally. Each block fully owns its
         DH rows. Cross-block contention is eliminated; the only atomics left
         are within-block accumulations across the inner V-loop, which the L1
         atomic units coalesce efficiently.
       - DE kernel: grid (V/BV,), loops T internally. Symmetric.
     Total matmul FLOPs are unchanged (still ~3 GEMM-equivalents) — only the
     synchronization pattern changes.

  3. Triton autotune.
     v1 used a fixed MCCEKernelConfig with conservative defaults (BT=16, BV=128,
     BD=32). v2 autotunes over a grid of (BLOCK_T, BLOCK_V, BLOCK_D, num_warps,
     num_stages) keyed on (T, V, D, S_MAX). First call per shape pays a tuning
     cost (~seconds); subsequent calls use the cached best.

Falls back to mcce_fast._MCCERawTokenMeanChunked for non-CUDA or non-Triton.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import mcce_fast  # reuse naive ref, chunked fallback, _validate_inputs

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


# -----------------------------------------------------------------------------
# Autotune configurations.
#
# All block sizes are >= 16 (Triton tl.dot minimum on Ampere). The list is
# deliberately small to keep autotune time reasonable; if you have very unusual
# shapes you may want to widen it.
# -----------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_T": 16,  "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 16,  "BLOCK_V": 256, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 256, "BLOCK_D": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_T": 64,  "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_T": 64,  "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_T": 64,  "BLOCK_V": 256, "BLOCK_D": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_T": 128, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
    ]

    _BWD_CONFIGS = [
        triton.Config({"BLOCK_T": 16,  "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 16,  "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_T": 32,  "BLOCK_V": 256, "BLOCK_D": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_T": 64,  "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_T": 64,  "BLOCK_V": 256, "BLOCK_D": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_T": 128, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
    ]


    # -------------------------------------------------------------------------
    # Forward: fused single-pass kernel.
    # -------------------------------------------------------------------------

    @triton.autotune(configs=_FWD_CONFIGS, key=["V", "D", "S_MAX"])
    @triton.jit
    def _mcce_fwd_fused_kernel(
        H, E, LABELS, S, LSE, TARGET_SUM,
        T,
        V: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # NOTE: we intentionally do NOT atomic_add into a shared scalar loss buffer
        # here. Triton's autotuner reruns the same kernel multiple times per config
        # to measure latency, and accumulator state would carry over between those
        # runs and produce wrong values during autotune. Instead we write lse and
        # target_sum per row and reduce on the host.
        pid_t = tl.program_id(0)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        row_valid = rows < T
        d_offsets = tl.arange(0, BLOCK_D)
        v_offsets = tl.arange(0, BLOCK_V)

        # Online softmax over V.
        running_m = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        running_s = tl.zeros((BLOCK_T,), tl.float32)

        for v0 in range(0, V, BLOCK_V):
            v = v0 + v_offsets
            v_valid = v < V

            z = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                h = tl.load(
                    H + rows[:, None] * D + d[None, :],
                    mask=row_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                e = tl.load(
                    E + d[:, None] + v[None, :] * D,
                    mask=d_valid[:, None] & v_valid[None, :],
                    other=0.0,
                )
                z += tl.dot(h, e)

            z = tl.where(row_valid[:, None] & v_valid[None, :], z, -float("inf"))

            chunk_m = tl.max(z, axis=1)
            new_m = tl.maximum(running_m, chunk_m)
            # If running_m is -inf (first tile), exp(running_m - new_m) is 0/NaN;
            # guard explicitly so running_s stays at 0 on the first tile.
            alpha = tl.where(running_m == -float("inf"), 0.0, tl.exp(running_m - new_m))
            running_s = running_s * alpha + tl.sum(tl.exp(z - new_m[:, None]), axis=1)
            running_m = new_m

        lse = running_m + tl.log(running_s)

        # Target sum: per-row dot product with E[labels[i, j]] for each j < s[i].
        s_i_int = tl.load(S + rows, mask=row_valid, other=0).to(tl.int32)
        target_sum = tl.zeros((BLOCK_T,), tl.float32)

        for j in range(0, S_MAX):
            active = row_valid & (j < s_i_int)
            y = tl.load(LABELS + rows * S_MAX + j, mask=active, other=0)
            dot = tl.zeros((BLOCK_T,), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                h = tl.load(
                    H + rows[:, None] * D + d[None, :],
                    mask=row_valid[:, None] & d_valid[None, :],
                    other=0.0,
                ).to(tl.float32)
                e_y = tl.load(
                    E + y[:, None] * D + d[None, :],
                    mask=active[:, None] & d_valid[None, :],
                    other=0.0,
                ).to(tl.float32)
                dot += tl.sum(h * e_y, axis=1)
            target_sum += tl.where(active, dot, 0.0)

        tl.store(LSE + rows, lse, mask=row_valid)
        tl.store(TARGET_SUM + rows, target_sum, mask=row_valid)


    # -------------------------------------------------------------------------
    # Backward: dH kernel. Grid (T/BT,). Each block fully owns DH[t_block, :].
    # -------------------------------------------------------------------------

    @triton.autotune(configs=_BWD_CONFIGS, key=["V", "D", "S_MAX"], reset_to_zero=["DH"])
    @triton.jit
    def _mcce_bwd_dh_kernel(
        H, E, LABELS, S, LSE, TOTAL_RAW, GRAD_LOSS, DH,
        T,
        V: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        row_valid = rows < T
        d_offsets = tl.arange(0, BLOCK_D)
        v_offsets = tl.arange(0, BLOCK_V)

        lse = tl.load(LSE + rows, mask=row_valid, other=0.0)
        s_i_int = tl.load(S + rows, mask=row_valid, other=0).to(tl.int32)
        s_i_f = s_i_int.to(tl.float32)
        grad_loss = tl.load(GRAD_LOSS).to(tl.float32)
        total_raw = tl.load(TOTAL_RAW).to(tl.float32)
        scale = grad_loss / total_raw

        for v0 in range(0, V, BLOCK_V):
            v = v0 + v_offsets
            v_valid = v < V

            # Recompute z[BT, BV] over the full D dimension.
            z = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                h = tl.load(
                    H + rows[:, None] * D + d[None, :],
                    mask=row_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                e = tl.load(
                    E + d[:, None] + v[None, :] * D,
                    mask=d_valid[:, None] & v_valid[None, :],
                    other=0.0,
                )
                z += tl.dot(h, e)

            valid = row_valid[:, None] & v_valid[None, :]
            z = tl.where(valid, z, -float("inf"))

            # g[BT, BV] = scale * s_i * softmax_i(v)  -  target_indicator * scale
            p = tl.exp(z - lse[:, None])
            g = (scale * s_i_f)[:, None] * p
            g = tl.where(valid, g, 0.0)
            for j in range(0, S_MAX):
                active = row_valid & (j < s_i_int)
                y = tl.load(LABELS + rows * S_MAX + j, mask=active, other=0)
                hit = active[:, None] & (y[:, None] == v[None, :]) & v_valid[None, :]
                g -= tl.where(hit, scale, 0.0)

            # dh[t_block, :] += g[BT, BV] @ E[v_block, :].
            # No cross-block contention since grid is 1D over t_block. Within-block
            # atomics across v-iterations are uncontended at the L1 level.
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                e_d = tl.load(
                    E + v[:, None] * D + d[None, :],
                    mask=v_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                g_cast = g.to(e_d.dtype)
                dh = tl.dot(g_cast, e_d)
                tl.atomic_add(
                    DH + rows[:, None] * D + d[None, :],
                    dh,
                    mask=row_valid[:, None] & d_valid[None, :],
                    sem="relaxed",
                )


    # -------------------------------------------------------------------------
    # Backward: dE kernel. Grid (V/BV,). Each block fully owns DE[v_block, :].
    # -------------------------------------------------------------------------

    @triton.autotune(configs=_BWD_CONFIGS, key=["V", "D", "S_MAX"], reset_to_zero=["DE"])
    @triton.jit
    def _mcce_bwd_de_kernel(
        H, E, LABELS, S, LSE, TOTAL_RAW, GRAD_LOSS, DE,
        T,
        V: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        v_valid = v < V
        d_offsets = tl.arange(0, BLOCK_D)
        t_offsets = tl.arange(0, BLOCK_T)

        grad_loss = tl.load(GRAD_LOSS).to(tl.float32)
        total_raw = tl.load(TOTAL_RAW).to(tl.float32)
        scale = grad_loss / total_raw

        for t0 in range(0, T, BLOCK_T):
            rows = t0 + t_offsets
            row_valid = rows < T

            lse = tl.load(LSE + rows, mask=row_valid, other=0.0)
            s_i_int = tl.load(S + rows, mask=row_valid, other=0).to(tl.int32)
            s_i_f = s_i_int.to(tl.float32)

            z = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                h = tl.load(
                    H + rows[:, None] * D + d[None, :],
                    mask=row_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                e = tl.load(
                    E + d[:, None] + v[None, :] * D,
                    mask=d_valid[:, None] & v_valid[None, :],
                    other=0.0,
                )
                z += tl.dot(h, e)

            valid = row_valid[:, None] & v_valid[None, :]
            z = tl.where(valid, z, -float("inf"))

            p = tl.exp(z - lse[:, None])
            g = (scale * s_i_f)[:, None] * p
            g = tl.where(valid, g, 0.0)
            for j in range(0, S_MAX):
                active = row_valid & (j < s_i_int)
                y = tl.load(LABELS + rows * S_MAX + j, mask=active, other=0)
                hit = active[:, None] & (y[:, None] == v[None, :]) & v_valid[None, :]
                g -= tl.where(hit, scale, 0.0)

            # de[v_block, :] += g[BT, BV].T @ H[t_block, :].
            for d0 in range(0, D, BLOCK_D):
                d = d0 + d_offsets
                d_valid = d < D
                h_d = tl.load(
                    H + rows[:, None] * D + d[None, :],
                    mask=row_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                g_cast = g.to(h_d.dtype)
                de = tl.dot(tl.trans(g_cast), h_d)
                tl.atomic_add(
                    DE + v[:, None] * D + d[None, :],
                    de,
                    mask=v_valid[:, None] & d_valid[None, :],
                    sem="relaxed",
                )


class _MCCERawTokenMeanTritonV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, embeddings, labels, s):  # type: ignore[override]
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if not hidden.is_cuda:
            raise RuntimeError("v2 requires CUDA tensors")

        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        lse = torch.empty((T,), device=hidden.device, dtype=torch.float32)
        target_sum = torch.empty((T,), device=hidden.device, dtype=torch.float32)
        total_raw = s.float().sum()

        grid_fwd = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
        _mcce_fwd_fused_kernel[grid_fwd](
            hidden, embeddings, labels, s, lse, target_sum,
            T, V, D, S_MAX,
        )

        s_f = s.float()
        loss = (s_f * lse - target_sum).sum() / total_raw

        ctx.save_for_backward(hidden, embeddings, labels, s, lse, total_raw)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):  # type: ignore[override]
        hidden, embeddings, labels, s, lse, total_raw = ctx.saved_tensors
        need_h, need_e = ctx.needs_input_grad[:2]
        T, D = hidden.shape
        V = embeddings.shape[0]
        S_MAX = labels.shape[1]

        dhidden = torch.zeros_like(hidden, dtype=torch.float32) if need_h else None
        dembeddings = torch.zeros_like(embeddings, dtype=torch.float32) if need_e else None

        if need_h:
            grid_dh = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
            _mcce_bwd_dh_kernel[grid_dh](
                hidden, embeddings, labels, s, lse, total_raw, grad_loss, dhidden,
                T, V, D, S_MAX,
            )
        if need_e:
            grid_de = lambda meta: (triton.cdiv(V, meta["BLOCK_V"]),)
            _mcce_bwd_de_kernel[grid_de](
                hidden, embeddings, labels, s, lse, total_raw, grad_loss, dembeddings,
                T, V, D, S_MAX,
            )

        return dhidden, dembeddings, None, None


def mcce_raw_token_mean_v2(
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    s: torch.Tensor,
    *,
    use_triton: bool = True,
    v_chunk_size: int = 8192,
    debug: bool = False,
    check_label_values: bool = False,
) -> torch.Tensor:
    """v2 MCCE. Same semantics as mcce_fast.mcce_raw_token_mean, faster kernels."""
    if debug:
        mcce_fast._validate_inputs(hidden, embeddings, labels, s, check_label_values=check_label_values)

    hidden = hidden.contiguous()
    embeddings = embeddings.contiguous()
    labels = labels.contiguous()
    s = s.contiguous()

    if use_triton and _TRITON_AVAILABLE and hidden.is_cuda:
        return _MCCERawTokenMeanTritonV2.apply(hidden, embeddings, labels, s)

    # Fall back to v1's chunked path for non-CUDA / non-Triton.
    return mcce_fast._MCCERawTokenMeanChunked.apply(hidden, embeddings, labels, s, int(v_chunk_size))
