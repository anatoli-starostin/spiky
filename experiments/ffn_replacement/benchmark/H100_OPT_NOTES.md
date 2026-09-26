# H100 FFN-slot optimization sweep — findings + kernels

Measured on an NVIDIA H100 80GB (SM 9.0, CUDA 13.0, torch 2.13), batch 48 × seq 512,
warmed, `no_grad`, trained checkpoints (0126/0127/0128). These are the H100 counterpart
to the RTX-5090 work in `README.md`. gpustar's committed files (`gather.py`,
`gather_cuda.cu/.py`, `run_bench.py`, `README.md`) are unchanged; the files here are
additive H100 prototypes.

Config shorthand (CompressionMHL, H4/inner 48, per-head block-diagonal anchors):
`0126` nap7/tph64 · `0127` nap7/tph128 · `0128` nap8/tph64. Vanilla dense FFN slot on
H100 ≈ **0.155 ms** (bf16) — the target; it is compute-bound and very fast on tensor cores.

## The winner: `gather_fused_v2_h100.cu` (fused routing + gather)

Fuses the routing (anchor-compare → index) and the gather+sum into ONE kernel, keeping
the index in **shared memory** so the ~50 MB int64 index array is NEVER written to HBM.
Routing uses the v2 layout (below): z-slice staged column-major in shared, token-inner
warp mapping (conflict-free). Bit-exact vs the reference (native index + `embedding_bag`).

Routing+gather stage, ms/call (all **bit-exact** fp32):

| model | native+Triton | v2 routing + Triton | **fused_v2** | fused vs best non-fused |
|---|---|---|---|---|
| 0126 | 0.194 | 0.233 | **0.141** | 0.61× |
| 0127 | 0.387 | 0.486 | **0.255** | 0.53× |
| 0128 | 0.345 | 0.236 | **0.150** | 0.64× |

Full FFN slot (compress + fused_v2 + decompress) vs vanilla dense (0.155 ms):
**0126 ≈ 1.09× · 0127 ≈ 1.82× · 0128 ≈ 1.14×** (was 1.60× / 2.85× / 2.57× with native+Triton).
ncu on fused_v2/0128: L2 dropped 83.6% → 75.7% (the index-write pressure is gone); kernel
now balanced (L1/TEX 74%, L2 76%, compute 58%). **This is the best H100 path.** It does not
quite overtake vanilla dense, but closes the gap to ~9–14% for the tph64 configs.

## `route_v2_h100.cu` — bank-conflict-fixed routing (standalone)

ncu root-caused the native anchor-compare kernel as **L1/TEX-throughput bound** at nap8
(0128: 99% L1, 27% compute, 6% DRAM — everything cache-resident; 225 µs) vs compute-bound
at nap7 (0126: 82 µs). The native kernel is 1 thread per (token,table), re-reading z
through L1 with data-dependent shared-bank collisions. route_v2 stores the z-slice
**column-major** in shared + a **token-inner** warp map ⇒ per-(table,pair) reads are
conflict-free and anchor reads become broadcasts. Bit-exact. Standalone routing stage:

| model | native | route_shared (v1) | **route_v2** |
|---|---|---|---|
| 0126 nap7 | 0.082 | **0.062 (1.33×)** | 0.121 (0.68×, regresses) |
| 0127 nap7 | 0.160 | **0.111 (1.43×)** | 0.258 (0.62×, regresses) |
| 0128 nap8 | 0.225 | 0.137 (1.64×) | **0.114 (1.97×)** |

**Regime-dispatched:** route_v2 wins the L1-bound high-nap case (nap≥8); route_shared (v1)
wins the compute-bound low-nap case (nap≤7). ncu on route_v2/0128 confirms L1/TEX 96%→63%.
(In practice, prefer `gather_fused_v2` over standalone routing + separate gather.)

## `route_shared_h100.cu` — v1 shared-staged routing

The first fix: stage each token's 48-col z-slice in shared once and reuse across tables
(cuts redundant L1 traffic). Row-major z, per-(token,table) thread map. Best for nap7
(1.33–1.43× over native). Superseded by fused_v2 for the fused path; kept as the nap7
routing reference.

## DEAD levers (measured, ruled out on H100)

- **L2 residency pinning** (`cudaAccessPolicyWindow`): 1.07–1.12× slower than plain
  cuda-fp32, 1.8× slower than Triton — the 6–12 MB table is already implicitly L2-resident,
  so pinning only steals L2 from the streaming index/output.
- **cp.async double-buffered gather**: 1.37–1.53× slower than Triton — the gather is pure
  memory latency with no compute to overlap and no row reuse.
- **Tensor-core GEMM** (one-hot selection @ table, dense bf16): **14–22× slower**; sparse
  CSR 2–3× worse still. The M=48 output is too skinny for tensor cores and the
  selection-matrix build (402–805 MB/head) dominates. cuSPARSELt absent; 2:4 sparsity is
  inapplicable (sparsity is 1-of-table_dim, not 2:4).
- **`F.embedding_bag(mode='sum')`**: 5.4–5.8× slower than Triton (it is the original path
  Triton replaced). Under `torch.compile` it is UNCHANGED — inductor captures it (0 graph
  breaks) but lowers to the same ATen kernel; no speedup.
- **`torch.compile` on the whole routed FFN**: 4–5× slower than fused_v2, ~= eager — it
  graph-breaks (6×) at the custom `lutorch_cuda` routing op and can't fuse/improve it; the
  compress/decompress GEMMs it CAN touch are already cuBLAS-optimal.

## Building / using these kernels (JIT via torch)

```python
from torch.utils.cpp_extension import load
ext = load(name='gather_fused_v2_h100',
           sources=['gather_fused_v2_h100.cu'],
           extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
           extra_cflags=['-O3', '-std=c++20'])
# NOTE: -std=c++20 is REQUIRED on this torch build (ATen headers fail at the default std).
```

APIs (all fp32, bit-exact; index int64 from the native MSB bit-pack kernel):
- `gather_fused_v2_h100.fused_v2(Z[N,H*48], Aa, Ab, W[n_tables,table_dim,48|64], n_heads, tph, nap, block_n)` → `[N, H, 48]`. `Z` is the compressed input (post compress-Linear); `Aa/Ab` are `lut.soft_anchor_a_long/b_long`; `W` is `lut.weights`. block_n ∈ {32,64,85}; 64 was best.
- `route_v2_h100.route_v2(...)` / `route_shared_h100.route_shared(...)` → index `[N, n_tables]` int64 (feed to the Triton gather).

Assumes the multi_head_input block-diagonal anchor layout (head h reads z cols
[h·48, h·48+48)), which holds for the 01xx CompressionMHL grid. Not yet wired into
`run_bench.py` — these are prototypes to preserve; integration TBD.
