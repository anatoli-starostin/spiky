---
name: fastmhl-hybrid-smooth-fwd-dispatch
description: "FastMHL hybrid_smooth forward dispatches on per-head n_outputs >= 128: bmm+sparse-S above the threshold, gather below. n_heads itself is NOT the right criterion."
metadata:
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# FastMHL hybrid_smooth forward: n_outputs-dispatched (2026-06-06)

`_FastMHLutHybridSmooth.forward` in `src/spiky/lutorch/fast_multi_head_lut.py`
dispatches between two compiled bodies based on **per-head N = `weights.shape[2]`
(the constructor's `n_outputs` arg)**:

- **n_outputs >= 128 → `_hybrid_smooth_fwd_bmm`**: build sparse selection mass
  `S[B, n_tables, K]` (2 nonzeros per (b, t), in bf16 under autocast), then
  one bmm against `weights[n_heads, tph*K, n_out]`. Replaces two random-access
  F.embedding gathers (~9.7 GB bf16 HBM at random offsets at NAP=7 out_proj
  shape) with one streaming bf16 matmul reading ~1.6 GB.
- **n_outputs < 128 → `_hybrid_smooth_fwd_gather`**: gather+blend+sum body.
  At small N, tensor cores can't amortise tile overhead — bmm pads N up to
  its tile size and wastes the register file. Gather's random payload per
  (b, t) is also tiny (~n_out·2 bytes), so random-HBM cost stays low.

## Why n_outputs, not n_heads?

The bmm decomposes into `n_heads` independent matmuls of shape
`[B, M=tph*K] @ [M, N=n_outputs]`. **N is the dimension that determines tensor
core efficiency** (tile sizes, register reuse, SM occupancy). M and n_heads
scale total work but don't change per-matmul efficiency.

A naive "dispatch on n_heads==1" criterion happens to work for LUTGPT modules
(all n_heads=1 modules have n_outputs=384, all n_heads=6 have n_outputs=64),
but it would WRONGLY send (n_heads=4, n_outputs=96) or (n_heads=6,
n_outputs=128) to gather — losing 12–118 ms per call.

## Crossover sweep (B=6144, NAP=6, tph=256)

|       | n_out=32 | 64    | 128   | 192   | 384   |
|-------|----------|-------|-------|-------|-------|
| n_heads=1 gather | 1.52 | 1.94 | 2.75  | 3.52  | 5.91  |
| n_heads=1 bmm    | 1.78 | 2.03 | **2.47** | **2.90**  | **4.22**  |
| n_heads=6 gather | 27.1 | 29.4 | 53.1  | 73.2  | 188.7 |
| n_heads=6 bmm    | 29.7 | 31.4 | **25.8** | **30.4**  | **70.8**  |
| n_heads=2 gather | —    | —    | —     | 28.6  | —     |
| n_heads=2 bmm    | —    | —    | —     | **14.9** | —  |
| n_heads=4, n_out=96 | gather 33.5 / **bmm 20.7**             |

Crossover at n_outputs ≈ 64–128 regardless of n_heads. **Threshold 128 is
safely on the bmm-winning side everywhere.** Higher n_heads compounds the
bmm win because gather scales linearly with n_tables=n_heads·tph, while bmm
batches heads via tensor cores.

## LUTGPT modules at B=12288

| Module | n_heads | n_outputs | OLD gather | NEW dispatch | Δ |
|---|---|---|---|---|---|
| out_proj | 1 | 384 | 29.4 ms | **24.0 ms** | **−5.4 ms** |
| residual_lut | 1 | 384 | 19.0 ms | **15.4 ms** | **−3.6 ms** |
| emb_resid_lut | 1 | 384 | 19.0 ms | **15.4 ms** | **−3.6 ms** |
| qkv_lut | 6 | 64 | 41.4 ms | 42.4 ms | +1.0 ms |
| v_lut | 6 | 64 | 41.3 ms | 42.3 ms | +1.0 ms |

Net per LUTGPT-style step (6 layers × 4 LUTs/layer + 1 emb_resid):
**806 → 761 ms = −5.6% wall-clock.** H mode (hard/ball) unchanged.

## Implementation notes

- Tried sharing the front half (d, main_index, alt_index, u) in a separate
  `@torch.compile`'d helper — adds ~1 ms because each compile boundary breaks
  Inductor's pointwise fusion. Front half inlined into both bodies instead.
- `s_dtype = torch.bfloat16 if use_bf16 and x.is_cuda else weights.dtype` so
  S is built directly in bf16 under autocast — skips the fp32→bf16 cast
  inside the matmul (saves 1.5–2.5 ms and ~5 GB peak at qkv shape, but qkv
  goes to gather anyway).
- Correctness verified at fp32 (diff 1.5e-8 ULP) and bf16 (out 1.5e-4
  bf16-precision, grads at fp32 precision because backward uses
  main_index/alt_index/u which are bit-identical between paths).
- The dispatch is precisely the optimisation Inductor cannot do
  automatically — it requires shape-conditional algorithm choice, not just
  shape-specialised codegen.

Cross-refs: [[fastmhl-hard-ball-deployment-sota]],
[[hybrid-smooth-wgrad-compile-ceiling]].
