#!/usr/bin/env python3
"""
Validation harness for mcce_fast.py's Triton path.

Run from the directory containing mcce_fast.py:

    python validate_mcce_triton.py

Useful variants:

    CUDA_LAUNCH_BLOCKING=1 python validate_mcce_triton.py
    python validate_mcce_triton.py --dtypes fp16 bf16 fp32
    python validate_mcce_triton.py --quick
    python validate_mcce_triton.py --benchmark

The test compares the Triton implementation against the naive full-logits
reference for forward loss, d(hidden), and d(embeddings). It also compares the
chunked fallback against the same reference so failures are easier to localize.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from typing import Iterable

import torch

try:
    import triton  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"ERROR: Triton import failed: {exc}", file=sys.stderr)
    raise

import mcce_fast
from mcce_fast import MCCEKernelConfig, mcce_raw_token_mean, naive_mcce_raw_token_mean


def _dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def _tol(dtype: torch.dtype) -> tuple[float, float, float, float]:
    """Return loss_atol, loss_rtol, grad_atol, grad_rtol."""
    # Triton tl.dot may use Tensor Core math / TF32 depending on dtype and GPU.
    # These tolerances are intended to catch real implementation bugs without
    # failing on expected matmul-order and Tensor Core roundoff.
    if dtype is torch.float16:
        return 2.0e-2, 2.0e-2, 2.0e-2, 2.0e-2
    if dtype is torch.bfloat16:
        return 6.0e-2, 6.0e-2, 6.0e-2, 6.0e-2
    if dtype is torch.float32:
        return 2.0e-2, 2.0e-2, 2.0e-2, 2.0e-2
    raise TypeError(dtype)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _max_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-8) -> float:
    denom = b.float().abs().clamp_min(eps)
    return float(((a.float() - b.float()).abs() / denom).max().item())


def _assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> None:
    if not torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: "
            f"max_abs={_max_abs(got, ref):.4e}, max_rel={_max_rel(got, ref):.4e}, "
            f"atol={atol:.1e}, rtol={rtol:.1e}"
        )


def _make_case(
    *,
    T: int,
    D: int,
    V: int,
    S_MAX: int,
    dtype: torch.dtype,
    label_dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Scale inputs so logits are not pathologically huge. This makes validation
    # sensitive to algorithmic mistakes rather than overflow/extreme-softmax cases.
    scale = 1.0 / math.sqrt(D)
    hidden = (torch.randn((T, D), device=device, dtype=dtype, generator=g) * scale).requires_grad_(True)
    embeddings = (torch.randn((V, D), device=device, dtype=dtype, generator=g) * scale).requires_grad_(True)

    s = torch.randint(1, S_MAX + 1, (T,), device=device, dtype=torch.int32, generator=g)
    labels = torch.randint(0, V, (T, S_MAX), device=device, dtype=label_dtype, generator=g)

    # Test duplicate/multiset semantics. Repeated labels should be counted twice.
    if S_MAX >= 2:
        labels[:, 1] = labels[:, 0]

    # Inactive slots should be ignored completely. Make them deliberately invalid
    # to verify that kernels do not read/use labels[i, j] where j >= s[i].
    cols = torch.arange(S_MAX, device=device)[None, :]
    inactive = cols >= s[:, None]
    labels = torch.where(inactive, torch.full_like(labels, V + 777), labels)

    return hidden, embeddings, labels, s


def _run_one(
    *,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    label_dtype: torch.dtype,
    config: MCCEKernelConfig,
    seed: int,
    verbose: bool,
) -> None:
    T, D, V, S_MAX = shape
    device = torch.device("cuda")
    loss_atol, loss_rtol, grad_atol, grad_rtol = _tol(dtype)

    hidden, embeddings, labels, s = _make_case(
        T=T,
        D=D,
        V=V,
        S_MAX=S_MAX,
        dtype=dtype,
        label_dtype=label_dtype,
        device=device,
        seed=seed,
    )

    # Naive full-logits reference.
    h_ref = hidden.detach().clone().requires_grad_(True)
    e_ref = embeddings.detach().clone().requires_grad_(True)
    loss_ref = naive_mcce_raw_token_mean(h_ref, e_ref, labels, s)
    loss_ref.backward()

    # Chunked fallback reference. This helps distinguish Triton issues from math issues.
    h_chunk = hidden.detach().clone().requires_grad_(True)
    e_chunk = embeddings.detach().clone().requires_grad_(True)
    loss_chunk = mcce_raw_token_mean(
        h_chunk,
        e_chunk,
        labels,
        s,
        use_triton=False,
        v_chunk_size=max(17, min(V, 257)),
        debug=True,
        check_label_values=True,
    )
    loss_chunk.backward()

    # Triton path under test.
    h_tri = hidden.detach().clone().requires_grad_(True)
    e_tri = embeddings.detach().clone().requires_grad_(True)
    loss_tri = mcce_raw_token_mean(
        h_tri,
        e_tri,
        labels,
        s,
        use_triton=True,
        config=config,
        debug=True,
        check_label_values=True,
    )
    loss_tri.backward()
    torch.cuda.synchronize()

    # Chunked fallback should be quite close to naive. Keep same tolerance family
    # because matmul decomposition/order can still differ.
    _assert_close("chunked loss", loss_chunk, loss_ref, atol=loss_atol, rtol=loss_rtol)
    _assert_close("chunked dH", h_chunk.grad, h_ref.grad, atol=grad_atol, rtol=grad_rtol)
    _assert_close("chunked dE", e_chunk.grad, e_ref.grad, atol=grad_atol, rtol=grad_rtol)

    # Triton correctness check.
    _assert_close("triton loss", loss_tri, loss_ref, atol=loss_atol, rtol=loss_rtol)
    _assert_close("triton dH", h_tri.grad, h_ref.grad, atol=grad_atol, rtol=grad_rtol)
    _assert_close("triton dE", e_tri.grad, e_ref.grad, atol=grad_atol, rtol=grad_rtol)

    if verbose:
        print(
            "PASS "
            f"shape=(T={T},D={D},V={V},S={S_MAX}) "
            f"dtype={str(dtype).replace('torch.', '')} labels={str(label_dtype).replace('torch.', '')} "
            f"loss_ref={loss_ref.item():.8f} loss_tri={loss_tri.item():.8f} "
            f"loss_abs={abs(loss_ref.item() - loss_tri.item()):.3e} "
            f"dH_abs={_max_abs(h_tri.grad, h_ref.grad):.3e} "
            f"dE_abs={_max_abs(e_tri.grad, e_ref.grad):.3e}"
        )


def _benchmark_one(config: MCCEKernelConfig) -> None:
    T, D, V, S_MAX = 256, 128, 32768, 4
    dtype = torch.float16
    hidden, embeddings, labels, s = _make_case(
        T=T,
        D=D,
        V=V,
        S_MAX=S_MAX,
        dtype=dtype,
        label_dtype=torch.int32,
        device=torch.device("cuda"),
        seed=999,
    )

    def step() -> torch.Tensor:
        h = hidden.detach().clone().requires_grad_(True)
        e = embeddings.detach().clone().requires_grad_(True)
        loss = mcce_raw_token_mean(h, e, labels, s, use_triton=True, config=config)
        loss.backward()
        return loss

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    iters = 20
    last_loss = None
    for _ in range(iters):
        last_loss = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"benchmark smoke: shape=(T={T},D={D},V={V},S={S_MAX}) dtype=fp16 "
        f"iters={iters} avg_ms={elapsed * 1000.0 / iters:.3f} loss={float(last_loss.item()):.6f}"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], help="fp16 bf16 fp32")
    parser.add_argument("--quick", action="store_true", help="run fewer shapes")
    parser.add_argument("--benchmark", action="store_true", help="run a small speed smoke test after validation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Triton validation requires a CUDA GPU")
    if not mcce_fast._TRITON_AVAILABLE:
        raise RuntimeError("mcce_fast reports Triton unavailable; install/import Triton first")

    device_name = torch.cuda.get_device_name()
    print(f"CUDA device: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton

        print(f"Triton: {triton.__version__}")
    except Exception:
        pass

    config = MCCEKernelConfig(
        block_t=16,
        block_v=128,
        block_d=32,
        reduce_block_v=256,
        num_warps_lse=4,
        num_warps_target=4,
        num_warps_finalize=4,
        num_warps_backward=8,
    )
    print(f"Config: {asdict(config)}")

    shapes = [
        (1, 16, 33, 1),        # CE-equivalent degenerate case
        (7, 32, 127, 2),       # tiny odd T/V
        (17, 65, 1009, 4),     # D not divisible by BLOCK_D, V not divisible by BLOCK_V
        (33, 128, 4099, 8),    # larger S_MAX and odd V
    ]
    if args.quick:
        shapes = shapes[:2]

    dtypes = [_dtype(x) for x in args.dtypes]
    label_dtypes = [torch.int64, torch.int32]

    failures: list[str] = []
    case_id = 0
    for dtype in dtypes:
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("SKIP bf16: CUDA device does not report bf16 support")
            continue
        for label_dtype in label_dtypes:
            for shape in shapes:
                case_id += 1
                try:
                    _run_one(
                        shape=shape,
                        dtype=dtype,
                        label_dtype=label_dtype,
                        config=config,
                        seed=args.seed + case_id,
                        verbose=not args.quiet,
                    )
                except Exception as exc:
                    msg = (
                        f"FAIL shape={shape} dtype={dtype} label_dtype={label_dtype}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(msg, file=sys.stderr)
                    failures.append(msg)

    if failures:
        print("\nValidation failed:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1

    print("\nAll Triton MCCE validation cases passed.")

    if args.benchmark:
        _benchmark_one(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
