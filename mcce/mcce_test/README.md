# MCCE deep validation scratch repo

Scratch repo for factually validating `mcce_fast.py` (Multi-target Cut Cross Entropy,
raw-token-mean specialization) — both correctness (math, dtypes, edge shapes) and
strategic claims (memory savings, speed).

## Layout

- `mcce_fast.py` — the v1 implementation under test (Triton kernels + chunked fallback + naive reference). Contains a one-line dtype-cast fix applied during this validation; otherwise unchanged.
- `mcce_fast_v2.py` — performance-oriented rewrite (fused single-pass forward + atomic-contention-free backward + Triton autotune). 2.5-2.6× faster than v1 at every tested shape, preserves the 5.6× memory savings vs naive.
- `validate_mcce_triton.py` — original validation harness shipped with the v1 implementation.
- `tests/` — pytest suite anchored to `torch.nn.functional.cross_entropy` rather than only to `mcce_fast`'s own naive reference.
  - `test_anchor_cross_entropy.py` — `S_MAX=1` vs standard CE; general `s` vs expanded virtual-rows CE; duplicate-label multiset semantics; inactive-slot insensitivity. Runs both v1 (`triton` and `chunked`) paths.
  - `test_finite_difference.py` — central finite-difference grad check for both `hidden` and `embeddings`.
  - `test_edge_shapes.py` — boundary conditions for v1 (V at/just-over `BLOCK_V`, T=1, large V forcing multi-reduce blocks, non-default `MCCEKernelConfig`, etc.).
  - `test_v2_against_ground_truth.py` — same anchor + edge-case suite applied to v2.
- `bench_mcce.py` — memory + speed benchmark comparing naive, chunked, v1, v2 at production-scale shapes.

## Setup

```
uv sync
```

(Requires CUDA, recent Triton — installs torch 2.12 + triton 3.7.)

## Run

```
# Shipped harness (24 dtype × shape cases)
uv run python validate_mcce_triton.py --dtypes fp16 bf16 fp32

# Deep correctness suite (~70s, 60 cases)
uv run pytest

# Memory + speed benchmarks (~3 min on RTX 3090 Ti)
uv run python bench_mcce.py
```

## Findings on this hardware (RTX 3090 Ti, torch 2.12, triton 3.7)

See `VALIDATION_REPORT.md`.
