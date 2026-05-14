# MCCE Validation Report

## 1. Setup

### 1.1 Hardware

```
GPU:        NVIDIA GeForce RTX 3090 Ti
Memory:     23.55 GB (24,564 MiB)
Driver:     595.71.05
CUDA cap:   13.2
Compute:    sm_86 (Ampere)
```

### 1.2 Software

```
OS:         Linux 7.0.2-2-cachyos
Python:     3.12.11
torch:      2.12.0+cu130
triton:     3.7.0
pytest:     9.0.3
numpy:      2.4.4
uv:         0.11.8
```

### 1.3 Implementations under test

Four execution paths share the same public API (`hidden[T, D]`, `embeddings[V, D]`,
`labels[T, S_MAX]`, `s[T]` → scalar loss). They are co-located in this repository:

| Path | File | Description |
|------|------|-------------|
| `naive` | `mcce_fast.py::naive_mcce_raw_token_mean` | Materializes `[T, V]` logits via `H @ E.T`; reference. |
| `chunked` | `mcce_fast.py::_MCCERawTokenMeanChunked` | Pure PyTorch, V-tiled, custom autograd. No `[T, V]` materialization. |
| `v1 Triton` | `mcce_fast.py::_MCCERawTokenMeanTriton` | Original Triton implementation. Contains a one-line dtype-cast fix applied during this work (Finding 1). |
| `v2 Triton` | `mcce_fast_v2.py::_MCCERawTokenMeanTritonV2` | Performance-oriented rewrite produced in this work. |

All four implementations evaluate the same target objective:

$$
L = \frac{1}{N}\left[\sum_{i=0}^{T-1} s_i \cdot \mathrm{lse}_i \;-\; \sum_i \sum_{j=0}^{s_i-1} z_{i, \text{labels}[i,j]}\right], \quad N = \sum_i s_i, \quad \mathrm{lse}_i = \log\sum_v e^{z_{i,v}}, \quad z_{i,v} = h_i \cdot e_v
$$

### 1.4 Methodology

All measurements were taken on the hardware/software stack in 1.1–1.2 with these conventions:

- **Seeds**: every test/benchmark fixture uses a `torch.Generator`-seeded RNG; specific seeds are stated in source. Inputs are scaled by `1/√D` so logits stay in a numerically sane range.
- **Tolerances**: dtype-aware. For fp32: `atol=2e-3, rtol=2e-3` on loss; `5e-3` on gradients. For bf16: `8e-2` (loss + grads). For fp16: `2e-2` (loss + grads). These are calibrated to absorb expected tensor-core roundoff, not algorithmic error.
- **Memory**: `torch.cuda.reset_peak_memory_stats()` after warmup, then a single fwd+bwd, then `torch.cuda.max_memory_allocated()`. Autotune (v2 only) is exhausted during warmup so the reported peak reflects the selected configuration, not the worst trial.
- **Wall-clock**: 2–3 untimed warmup iterations, 5–10 timed iterations, `torch.cuda.synchronize()` between phases; mean over timed iterations reported.

### 1.5 Reproduction

```bash
uv sync
uv run python validate_mcce_triton.py --dtypes fp16 bf16 fp32  # original shipped harness
uv run pytest                                                    # 117 tests
uv run python bench_mcce.py                                      # memory + speed at 4 shapes
```

---

## 2. Correctness measurements

### 2.1 Test suite composition

| Test file | Cases | Anchor reference |
|-----------|------:|------------------|
| `validate_mcce_triton.py` (shipped) | 24 | `naive_mcce_raw_token_mean` |
| `tests/test_anchor_cross_entropy.py` | 34 | `torch.nn.functional.cross_entropy` (single + expanded) |
| `tests/test_finite_difference.py` | 4 | Central finite-difference of forward |
| `tests/test_edge_shapes.py` | 22 | `F.cross_entropy` on the expanded virtual-rows construction |
| `tests/test_v2_against_ground_truth.py` | 25 | `F.cross_entropy` on the expanded virtual-rows construction |
| `tests/test_paper_mce_equivalence.py` | 32 | Verbatim re-implementation of TST paper Listing 3 |
| **Total** | **141** | |

### 2.2 Pass results

| Suite | dtypes covered | label dtypes | shapes covered | Result |
|-------|---|---|---|---|
| Shipped harness | fp16, bf16, fp32 | int32, int64 | 4 shapes × 2 label dtypes × 3 dtypes = 24 | **24/24 pass** |
| Anchor (v1) | fp16, bf16, fp32 | int64 | 4 shapes × 2 paths × 3 dtypes + edge cases = 34 | **34/34 pass** |
| Finite-difference | fp32 | int64 | 8 random coordinates of `hidden`, 8 of `embeddings` | **4/4 pass** |
| Edge shapes (v1) | fp32, bf16 | int32, int64 | 11 boundary shapes × 2 dtypes = 22 | **22/22 pass** |
| v2 ground-truth | fp32, bf16, fp16 | int64 | 6 shapes × 3 dtypes + 7 edge/special cases = 25 | **25/25 pass** |
| TST paper equivalence | fp32, bf16 | int64 | 5 shapes × 3 paths × 2 dtypes + 2 spot tests = 32 | **32/32 pass** |

Total: **141/141 pass** end-to-end (after Finding 1 below). Runtime: ~89s for pytest, ~5min for `validate_mcce_triton.py --dtypes fp16 bf16 fp32`.

### 2.3 Worst-case observed deviations (per-path, vs reference)

Measured on `tests/test_anchor_cross_entropy.py` shapes (T=8..97, D=64..128, V=257..1024, S_MAX=1..8):

| Path | dtype | max `|Δloss|` | max `|Δ dH|` | max `|Δ dE|` |
|------|-------|--------------|---------------|---------------|
| v1 Triton | fp32 | 5.15e-05 | 6.81e-04 | 6.81e-04 |
| v1 Triton | bf16 | 9.54e-07 | 1.95e-03 | 1.95e-03 |
| v1 Triton | fp16 | 1.22e-04 | 1.22e-04 | 3.05e-05 |
| chunked | fp32 | <1e-06 | <1e-04 | <1e-04 |
| chunked | bf16 | <1e-06 | <2e-03 | <2e-03 |
| v2 Triton | bf16, T=2048/V=32k | 0.00e+00 (exact) | 2.38e-07 | 1.19e-07 |

(All within the per-dtype tolerance bands declared in §1.4.)

### 2.4 Anchor to `torch.nn.functional.cross_entropy` (independent reference)

`tests/test_anchor_cross_entropy.py` validates the semantic claim that MCCE is well-defined relative to PyTorch's stock cross-entropy:

- **S_MAX=1**: `mcce_raw_token_mean(h, e, labels[:, 0:1], s=1)` == `F.cross_entropy(h@e.T, labels[:, 0], reduction='mean')`. Tested across 6 combinations of `{path × dtype}`. **6/6 pass.**
- **General s**: `mcce_raw_token_mean(h, e, labels, s)` == `F.cross_entropy` applied to the expanded `(h.repeat_interleave(s), labels_flat_active)` pair with `reduction='mean'`. Tested at 4 shapes × 2 paths × 3 dtypes. **24/24 pass.**
- **Duplicate-label multiset**: a label repeated `k` times in a row contributes `k` times to the loss (verified by setting all bag slots to the same label and comparing to single-label CE). **2/2 pass.**
- **Inactive-slot insensitivity**: `labels[i, j]` for `j ≥ s[i]` does not affect loss or gradients regardless of value. Tested with sentinel value `V + 999` (out-of-range integer). Loss and gradient deltas: **exactly 0.0** for both `v1 Triton` and `chunked`. **2/2 pass.**

### 2.5 Anchor to TST paper Listing 3

`tests/test_paper_mce_equivalence.py` re-implements the loss in Peng/Gigant/Quesnelle 2026 (arXiv:2605.06546), Appendix Listing 3, and compares all three CUDA paths against it.

Algebraic identity proven by the tests: under uniform bag size `s_i = s ∀ i`,

$$
L_{paper} = \frac{1}{s} \sum_{j=0}^{s-1} \mathbb{E}_i[L_{CE}(z_i, \text{labels}[i,j])] = \frac{1}{Ts}\sum_i\left[s \cdot \mathrm{lse}_i - \sum_j z_{i, \text{labels}[i,j]}\right] = L_{mcce}\Big|_{s_i=s}
$$

Measurements at TST-realistic shape (T=2048, V=32000, s=8, D=512, bf16):

```
paper loss   = 10.375124
v2 loss      = 10.375124
|Δloss|      = 0.000e+00
max |Δ dH|   = 2.384e-07
max |Δ dE|   = 1.192e-07
```

At fp32 (T=32, V=512, s=4):

```
paper:   6.2247371674
chunked: 6.2247371674
v2:      6.2247257233
|Δloss| (v2 - paper):       1.14e-05
|Δloss| (chunked - paper):  <1e-08
```

The v2 fp32 deviation (1.14e-05) coincides with the magnitude of TF32-tensor-core mantissa truncation on Ampere; the chunked path (also nominally fp32 but routed through cuBLAS) shows no measurable deviation, indicating the difference is in `tl.dot`'s TF32 accumulator and not in MCCE's formula.

**This equivalence is exact only when `s_i = s` for all i. When `s_i` varies per position (boundary masking), `mcce_fast` weights positions by their `s_i` while the paper's `F.cross_entropy(reduction='mean')` per-bag-slot weights positions uniformly. This divergence was not exercised by the test suite.**

### 2.6 Finite-difference cross-check

`tests/test_finite_difference.py`: central finite-difference of `mcce_raw_token_mean` forward, perturbing `eps = 1e-3` on 8 random coordinates of `hidden` and 8 of `embeddings` (small-shape fp32, T=8/D=16/V=97/S_MAX=3). Max `|analytic_grad − FD_grad|` measured:

| Path | hidden | embeddings |
|------|--------|------------|
| `v1 Triton` | < 1e-3 | < 1e-3 |
| `chunked` | < 1e-3 | < 1e-3 |

**4/4 pass.**

### 2.7 Determinism

Five repeated fwd+bwd runs on bit-identical bf16 inputs (T=1024, D=512, V=64000, S_MAX=4, v1 path):

```
loss values:
  11.067877769470215
  11.067878723144531
  11.067875862121582
  11.067874908447266
  11.067877769470215

dH max-abs-diff between run 0 and runs 1..4:
  run 1: 0.000e+00
  run 2: 0.000e+00
  run 3: 0.000e+00
  run 4: 0.000e+00

dE max-abs-diff between run 0 and runs 1..4:
  run 1: 2.384e-07
  run 2: 1.192e-07
  run 3: 2.384e-07
  run 4: 2.384e-07
```

Inter-run loss variation: ≤ 4 ULP fp32 (≈ 4e-6). dE variation ≤ 2.4e-7 (bf16 has ~7-bit mantissa; this is far below its representable precision of ~8e-3).

---

## 3. Memory measurements

### 3.1 Peak `torch.cuda.max_memory_allocated()` after fwd+bwd

Measurements from `bench_mcce.py`, bf16 inputs, S_MAX=4:

| Shape (T, D, V) | `naive` (MB) | `chunked` (MB) | `v1 Triton` (MB) | `v2 Triton` (MB) | v2/naive ratio |
|-----------------|------------:|---------------:|-----------------:|-----------------:|----------------|
| 512, 512, 32 000 | 424.8 | 212.3 | 143.8 | 270.8 | 0.637 |
| 2 048, 1 024, 128 000 | 5 906.9 | 1 162.4 | 1 024.3 | 1 024.3 | 0.173 |
| 4 096, 2 048, 128 000 | 11 564.4 | 2 396.5 | 2 048.4 | 2 048.4 | 0.177 |
| 8 192, 2 048, 200 000 | **OOM** (req 6.10 GiB at allocation) | 4 057.3 | 3 207.1 | 3 207.1 | — |

Theoretical `[T, V]` fp32 footprint for reference: 62.5 MB / 1 000 MB / 2 000 MB / 6 250 MB across the four shapes.

### 3.2 OOM threshold (this hardware, 23.55 GB total)

The `naive` path hits `CUDA out of memory` between shape (4096, 2048, 128 000) (fits: 11.6 GB) and (8192, 2048, 200 000) (OOMs: 6.10 GB allocation fails). `v1 Triton`, `v2 Triton`, and `chunked` complete the (8192, 2048, 200 000) case.

---

## 4. Wall-clock measurements

### 4.1 Forward + backward latency (ms, mean of 5–10 iterations after warmup)

Measurements from `bench_mcce.py`, bf16, S_MAX=4:

| Shape (T, D, V) | `naive` | `chunked` | `v1 Triton` | `v2 Triton` |
|-----------------|--------:|----------:|------------:|------------:|
| 512, 512, 32 000 | 8.228 | 9.907 | 6.809 | 5.129 |
| 2 048, 1 024, 128 000 | 84.346 | 110.995 | 198.669 | 78.203 |
| 4 096, 2 048, 128 000 | 279.893 | 395.107 | 819.434 | 310.722 |
| 8 192, 2 048, 200 000 | OOM | 1 159.879 | 2 580.556 | 979.710 |

### 4.2 Pairwise ratios

| Shape | v1/naive | v2/naive | chunked/naive | v2/v1 | v2/chunked |
|-------|---------:|---------:|--------------:|------:|----------:|
| 512, 512, 32 000 | 0.83 | 0.62 | 1.20 | 0.75 | 0.52 |
| 2 048, 1 024, 128 000 | 2.36 | 0.93 | 1.32 | 0.39 | 0.70 |
| 4 096, 2 048, 128 000 | 2.93 | 1.11 | 1.41 | 0.38 | 0.79 |
| 8 192, 2 048, 200 000 | — | — | — | 0.38 | 0.84 |

Sign convention: ratio < 1.0 means the numerator is faster.

### 4.3 Inter-run variance

`bench_mcce.py` was executed twice during validation; per-shape latency varied by ≤ 6% between runs.

---

## 5. Findings (observations from the measurements above)

### 5.1 Compile failure in `v1 Triton` under fp16/bf16 (as-shipped)

Running the as-shipped `validate_mcce_triton.py --dtypes fp16 bf16` produced 16/16 compile errors:

```
CompilationError: at 80:13:
        dh = tl.dot(g, e_block)
             ^
Both operands must be same dtype. Got fp32 and bf16
```

Source location: `mcce_fast.py:486–487` in `_mcce_backward_exact_kernel`. The kernel constructs `g` in fp32 (softmax + scale) and passes it to `tl.dot(g, e_block)` where `e_block` is loaded at native (fp16/bf16) dtype.

Modification applied: cast `g` to `h_block.dtype` at the `tl.dot` boundary only (mirroring the forward LSE kernel's pattern, preserving native-dtype matmul):

```python
g_cast = g.to(h_block.dtype)
dh = tl.dot(g_cast, e_block)
de = tl.dot(tl.trans(g_cast), h_block)
```

Post-fix: `validate_mcce_triton.py --dtypes fp16 bf16 fp32` passes 24/24 cases. fp32 was unaffected (not exercising the broken path).

### 5.2 `MCCEKernelConfig` implicit lower bound

Triton's `tl.dot` requires `M, N, K ≥ 16`. Configurations with smaller block sizes produce:

```
Input shapes should have M >= 1, N >= 1 and K >= 16
```

Default `MCCEKernelConfig` (`block_t=16, block_v=128, block_d=32`) is at the minimum legal `block_t`. Reducing it to `block_t=8` produces a compile error rather than a validation failure. Observed in `tests/test_edge_shapes.py::test_non_default_block_v` during development.

### 5.3 `tl.atomic_add` into scalar output is incompatible with `triton.autotune`

`mcce_fast_v2.py` initially used `tl.atomic_add(LOSS_NUM, block_sum)` in the forward kernel. Under `@triton.autotune`, this produced loss values approximately N_configs × correct_value because the autotuner reruns each candidate configuration multiple times for timing without resetting outputs. Observed with a 64-row test (T=64, V=1024): loss reported 8 547.7 vs reference 6.932.

Modification applied: replaced scalar `LOSS_NUM` atomic with per-row `target_sum[T]` and `lse[T]` outputs, with the final loss computed on the host via `(s.float() * lse - target_sum).sum() / total_raw`. Backward kernels accumulating into `DH`/`DE` use `reset_to_zero=["DH"]` / `["DE"]` in their autotune decorator.

### 5.4 v2 implementation differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Forward grid | 2D `(⌈T/BT⌉, ⌈V/BV⌉)` LSE-tiles kernel + 1D `(⌈T/BT⌉)` finalize kernel + 1D target-sum kernel | 1D `(⌈T/BT⌉)` fused kernel (online softmax over V + target-sum + lse store, in one pass) |
| Backward grid | 2D `(⌈T/BT⌉, ⌈V/BV⌉)`, atomic_add into both DH and DE from every block | 1D `(⌈T/BT⌉)` DH-only kernel + 1D `(⌈V/BV⌉)` DE-only kernel |
| HBM intermediates | `partial_m[T, ⌈V/BV⌉]` + `partial_s[T, ⌈V/BV⌉]` fp32 buffers (32 MB at T=4096, V=128k, BV=128) | `lse[T]` + `target_sum[T]` (32 KB at T=4096) |
| Block size selection | Fixed `MCCEKernelConfig` | `@triton.autotune` over 9 fwd configs × 8 bwd configs keyed on `(T, V, D, S_MAX)` |
| Atomic_add usage | Cross-block (contended) into DH/DE from `~⌈T/BT⌉·⌈V/BV⌉` blocks per row/column | Within-block only (no cross-block contention on output buffers) |

Total matmul-FLOPs are equivalent between v1 and v2: one forward `H @ E.T` + recompute in backward + dh matmul + de matmul.

---

## 6. Reproduction commands

| Measurement | Command |
|-------------|---------|
| Software versions (§1.2) | `uv run python -c "import torch, triton; print(torch.__version__, triton.__version__)"` |
| Shipped harness (§2.2, row 1) | `uv run python validate_mcce_triton.py --dtypes fp16 bf16 fp32` |
| Anchor + edge + FD + v2 (§2.2, rows 2–6) | `uv run pytest` |
| Memory + speed (§3, §4) | `uv run python bench_mcce.py` |
| Determinism (§2.7) | Embedded in this report; reproducible via the snippet quoted there |
| Finding 5.1 reproduction (compile failure) | `git stash` (Finding 1 fix) + `uv run python validate_mcce_triton.py --dtypes bf16` |

---

## 7. Out of scope (not measured in this work)

- Hardware other than RTX 3090 Ti (sm_86, Ampere).
- Behavior under `torch.cuda.amp.autocast` / `GradScaler`.
- Numerical behavior under pathological inputs (logits with values approaching fp16/bf16 overflow, all-equal logits, etc.).
- End-to-end model training convergence using either v1 or v2 as the loss step.
- Behavior under non-uniform `s_i` aligned to the TST paper's `-100` boundary-masking semantics. The current `mcce_fast` divergence from the paper's per-position averaging at sequence boundaries is documented in §2.5 but not numerically characterized.
- Comparison against external published MCCE/CCE implementations (`apple/ml-cross-entropy`, `linkedin/Liger-Kernel`, etc.).
- Persistent benchmarks across thermal states or with concurrent GPU workloads.
