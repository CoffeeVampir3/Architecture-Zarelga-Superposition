"""Memory and speed benchmarks for mcce_fast.

Validates two strategic claims:
  1. Memory: the Triton path does NOT materialize [T, V] or [T, S_MAX, D] tensors,
     so peak CUDA memory should be drastically lower than the naive reference at
     production-scale V.
  2. Speed: forward+backward wall-clock should be at least competitive with the
     naive reference at moderate V, and remain feasible at larger V where naive OOMs.

Run with:
    uv run python bench_mcce.py
    uv run python bench_mcce.py --triton-only-shape T=16384 D=2048 V=262144 S=4
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcce_fast
import mcce_fast_v2
from mcce_fast import mcce_raw_token_mean, naive_mcce_raw_token_mean
from mcce_fast_v2 import mcce_raw_token_mean_v2


@dataclass
class Result:
    name: str
    peak_mb: float
    fwd_ms: float
    fwd_bwd_ms: float
    loss: float
    ok: bool = True
    err: str = ""


def _make_inputs(T, D, V, S_MAX, dtype, device, seed=0):
    g = torch.Generator(device=device); g.manual_seed(seed)
    scale = 1.0 / math.sqrt(D)
    hidden = (torch.randn((T, D), device=device, dtype=dtype, generator=g) * scale).requires_grad_(True)
    embeddings = (torch.randn((V, D), device=device, dtype=dtype, generator=g) * scale).requires_grad_(True)
    s = torch.randint(1, S_MAX + 1, (T,), device=device, dtype=torch.int32, generator=g)
    labels = torch.randint(0, V, (T, S_MAX), device=device, dtype=torch.int64, generator=g)
    return hidden, embeddings, labels, s


def _time_callable(fn, *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return elapsed_ms, last


def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def _measure(name: str, build_fn, fwd_only=False, *, warmup=2, iters=5) -> Result:
    """Measure forward and forward+backward time and peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # 1) Peak memory: just one fwd+bwd. Allocate fresh tensors inside, run, measure.
        loss_t = None
        def one_step_fwd_bwd():
            nonlocal loss_t
            h, e, labels, s = build_fn()
            if name.startswith("naive"):
                loss = naive_mcce_raw_token_mean(h, e, labels, s)
            elif name.startswith("triton_v2"):
                loss = mcce_raw_token_mean_v2(h, e, labels, s)
            elif name.startswith("triton"):
                loss = mcce_raw_token_mean(h, e, labels, s, use_triton=True)
            else:  # chunked
                loss = mcce_raw_token_mean(h, e, labels, s, use_triton=False, v_chunk_size=8192)
            loss.backward()
            loss_t = loss.detach()
            return loss_t

        # Warmup first — this triggers Triton autotuning on the first call so the
        # peak-memory measurement reflects the chosen config, not the worst trial.
        for _ in range(warmup):
            one_step_fwd_bwd()
        torch.cuda.synchronize()

        # Memory measurement (single iter on cached/best config)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        one_step_fwd_bwd()
        torch.cuda.synchronize()
        peak = _peak_mb()

        # Speed measurement (fwd+bwd)
        fwd_bwd_ms, _ = _time_callable(one_step_fwd_bwd, warmup=warmup, iters=iters)

        # Speed measurement (fwd only)
        def one_step_fwd():
            h, e, labels, s = build_fn()
            with torch.no_grad():
                if name.startswith("naive"):
                    return naive_mcce_raw_token_mean(h, e, labels, s)
                elif name.startswith("triton_v2"):
                    return mcce_raw_token_mean_v2(h, e, labels, s)
                elif name.startswith("triton"):
                    return mcce_raw_token_mean(h, e, labels, s, use_triton=True)
                else:
                    return mcce_raw_token_mean(h, e, labels, s, use_triton=False, v_chunk_size=8192)
        if fwd_only:
            fwd_ms, _ = _time_callable(one_step_fwd, warmup=warmup, iters=iters)
        else:
            fwd_ms = float("nan")

        return Result(name=name, peak_mb=peak, fwd_ms=fwd_ms, fwd_bwd_ms=fwd_bwd_ms,
                      loss=float(loss_t.item()))
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return Result(name=name, peak_mb=float("nan"), fwd_ms=float("nan"),
                      fwd_bwd_ms=float("nan"), loss=float("nan"), ok=False,
                      err=f"OOM: {e}")
    except Exception as e:
        torch.cuda.empty_cache()
        return Result(name=name, peak_mb=float("nan"), fwd_ms=float("nan"),
                      fwd_bwd_ms=float("nan"), loss=float("nan"), ok=False,
                      err=f"{type(e).__name__}: {e}")


def _bench_shape(T: int, D: int, V: int, S_MAX: int, dtype: torch.dtype, device, *,
                 do_naive: bool, fwd_only: bool):
    print(f"\n== shape T={T} D={D} V={V} S_MAX={S_MAX} dtype={dtype} ==")
    print(f"   approx logits memory if materialized: T*V*4 = {T*V*4/1024**2:.1f} MB (fp32)")

    def builder():
        return _make_inputs(T, D, V, S_MAX, dtype, device)

    results: list[Result] = []
    if do_naive:
        results.append(_measure("naive (reference)", builder, fwd_only=fwd_only))
    results.append(_measure("chunked (no-Triton)", builder, fwd_only=fwd_only))
    results.append(_measure("triton (v1)",         builder, fwd_only=fwd_only))
    # autotune warmup so v2's first timed step doesn't pay the autotune cost
    print("   (autotuning v2 — may take ~5-20s on first shape)")
    results.append(_measure("triton_v2",           builder, fwd_only=fwd_only,
                            warmup=3, iters=10))

    print(f"   {'path':<22}  {'peak MB':>10}  {'fwd ms':>10}  {'fwd+bwd ms':>13}  {'loss':>10}")
    for r in results:
        if not r.ok:
            print(f"   {r.name:<22}  {'-':>10}  {'-':>10}  {'-':>13}  {'-':>10}   {r.err}")
            continue
        print(f"   {r.name:<22}  {r.peak_mb:>10.1f}  {r.fwd_ms:>10.3f}  {r.fwd_bwd_ms:>13.3f}  {r.loss:>10.4f}")

    # Strategic memory comparison
    by_name = {r.name: r for r in results if r.ok}
    if "naive (reference)" in by_name and "triton_v2" in by_name:
        print(f"   memory ratio triton_v2/naive = {by_name['triton_v2'].peak_mb / by_name['naive (reference)'].peak_mb:.3f}")
    if "triton (v1)" in by_name and "triton_v2" in by_name:
        v1 = by_name["triton (v1)"].fwd_bwd_ms
        v2 = by_name["triton_v2"].fwd_bwd_ms
        print(f"   speedup triton_v2/triton(v1) = {v1 / v2:.2f}x")
    if "naive (reference)" in by_name and "triton_v2" in by_name:
        n = by_name["naive (reference)"].fwd_bwd_ms
        v2 = by_name["triton_v2"].fwd_bwd_ms
        print(f"   speedup triton_v2 vs naive = {n / v2:.2f}x  ({'WIN' if v2 < n else 'LOSS'})")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fwd-only-too", action="store_true",
                    help="also time forward only (otherwise just fwd+bwd, which is faster overall)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return 1
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}, total mem {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"PyTorch {torch.__version__}, Triton {__import__('triton').__version__}")

    # Moderate vocab: naive fits, all three paths comparable.
    _bench_shape(T=512,  D=512,  V=32_000,   S_MAX=4, dtype=torch.bfloat16, device=device,
                 do_naive=True, fwd_only=args.fwd_only_too)
    if args.quick:
        return 0

    # Production-ish vocab: naive starts pinching memory.
    _bench_shape(T=2048, D=1024, V=128_000,  S_MAX=4, dtype=torch.bfloat16, device=device,
                 do_naive=True, fwd_only=args.fwd_only_too)

    # Aggressive: T=4k, V=128k. Probably naive OOMs on 24GB.
    _bench_shape(T=4096, D=2048, V=128_000,  S_MAX=4, dtype=torch.bfloat16, device=device,
                 do_naive=True, fwd_only=args.fwd_only_too)

    # Stretch: T=8k, V=200k.
    _bench_shape(T=8192, D=2048, V=200_000,  S_MAX=4, dtype=torch.bfloat16, device=device,
                 do_naive=True, fwd_only=args.fwd_only_too)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
