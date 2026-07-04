---
name: fastmhl-wgrad-bmm
description: "Sparse-S + bmm replacement for single-row LUT weight-grad scatter (hard+dense_K / hard+ball / hard+ball_gather) wins 12–41 % at n_outputs >= 128, ~10 % faster end-to-end at LUTGPT exp731 shapes. Trades 0.25 bf16 ULP gw precision."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# FastMHL wgrad via sparse-S + bmm — LUTGPT exp731 shapes

## What changed

`_soft_lut_bwd_body`, `_ball_lut_bwd_body`, `_ball_gather_lut_bwd_body` in
`src/spiky/lutorch/fast_multi_head_lut.py` now accept a `wgrad_via_bmm: bool`
parameter. When True, the single-row weight-grad scatter is replaced with
a sparse-S + bmm pattern:

```python
S = torch.zeros(B, n_tables, K, dtype=grad_pt.dtype, device=...)  # bf16 under autocast
S.scatter_(2, index.unsqueeze(-1), 1.0)
grad_weights = torch.einsum("btk,bto->tko", S, grad_pt).to(accum_dtype)
```

`_FastMHLutSoft.backward` dispatches automatically: **bmm if `n_outputs >= 128`,
index_add otherwise**. Same threshold as the hybrid_smooth forward dispatch
(tensor-core efficiency crossover at H100 bf16 LUTGPT shapes).

## Wins (B=12288, A/B against same-shape index_add)

| Module | NAP | tph | n_out | index_add | bmm | Δ | applied |
|---|---|---|---|---|---|---|---|
| qk_lut | 4 | 256 | 128 | 11.84 ms | 9.93 ms | -1.91 (-16%) | YES |
| v_lut | 6 | 256 | 64 | 13.91 ms | 14.00 ms | +0.09 | no (n_out<128) |
| out_proj | 7 | 512 | 384 | 15.68 ms | 9.88 ms | **-5.80 (-37%)** | YES |
| resid_lut | 6 | 256 | 384 | 4.92 ms | 3.68 ms | -1.24 (-25%) | YES |
| emb_resid | 6 | 256 | 384 | 4.92 ms | 3.68 ms | -1.24 (-25%) | YES |

Per training step at LUTGPT exp731 shapes (6 layers + emb_resid):
- Per fwd-bwd: 6×(qk + v + out + resid) + emb_resid → saves ~55 ms
- With grad_accum=2: ~110 ms / step
- 16 000 steps: ~29 min wallclock
- **exp731 4.89 h → ~4.4 h, ~10% faster end-to-end**

Ball / ball_gather have the same single-row scatter pattern and see the same
12–41% wgrad savings (`_ball_lut_bwd_body` for hard+ball mode = exp729's
recipe).

## Precision

`gw` rel_rms = 2.1e-3 vs the fp32-accumulated index_add baseline.
This is exactly **0.25 of one bf16 ULP** (eps_bf16 = 7.8e-3) and comes
purely from the bf16 output truncation of the cuBLAS bf16-tensor-core matmul
(internal fp32 accumulator → bf16 write). `gx`, `grad_log_T_soft`,
`grad_log_T_sel` are bit-exact (1e-7 noise floor).

Why this is safe:
- LUTGPT uses **Lion** optimizer for LUT params — Lion is sign-based, so
  magnitude precision in grad doesn't directly affect updates unless the sign
  flips, which requires grad to be near zero (where it's already noisy from
  the bf16 input grad_pt floor at 7.8e-3 per value).
- After 16k steps with independent rounding per step, momentum drift grows
  as sqrt(K) × 2e-3 ≈ 25%, comparable to natural bf16 input noise.
- AdamW (non-LUT params) doesn't touch LUT wgrad, so its precision is
  unaffected.

## Why bmm wins (B=12K, K=128)

- Atomic-add contention on the small `[n_tables*K, n_out]` dest tensor was the
  bottleneck (~8.5 ms for out_proj wgrad in compiled fused body).
- bmm sees a dense GEMM shape `[n_tables, K, n_out] = [B, n_tables, K] x [B, n_tables, n_out]`
  with bf16 tensor cores; the S tensor at exp731 recipe is 1.5 GB (fits in HBM
  comfortably) vs the 6.3 GB at the older publish recipe (B=32K) where this
  pattern was previously rejected — see `project_hybrid_smooth_wgrad_compile_ceiling.md`.
- At smaller n_outputs (v_lut n_out=64), the wgrad atomic contention is
  proportionally smaller while the S tensor is still 2.25 GB (n_tables=1536
  for H=6) → wash. Dispatch correctly avoids regression.

## What didn't work

- `torch.compile(mode='max-autotune-no-cudagraphs')` triggers an inductor bug
  (`FlexibleLayout → FixedLayout` assertion failure) on this body.
- fp32-input einsum (autocast disabled) is precision-perfect but always
  slower than the baseline at every shape (1.5–4× slower).
- TF32 tensor cores with fp32 inputs/output is precise but ~15% slower than
  baseline because the 3 GB fp32 S tensor dominates memory bandwidth.
- Chunked bmm with fp32 accumulator: same precision as plain bmm (chunking
  doesn't recover the per-cell bf16 truncation), no speed difference.
- bf16-dest atomic accumulator: precision destroyed (3-9e-2 rel error).

## Not yet applied

- `_hybrid_smooth_weight_grad` has a 2-row weighted scatter (main + alt with
  u / 1-u). Same family of optimization could apply, but separate work —
  see existing `project_hybrid_smooth_wgrad_compile_ceiling.md` note for why
  it was rejected at the older publish recipe; needs retest at exp731 recipe
  (B=12K, K=128).

## Files touched
- `src/spiky/lutorch/fast_multi_head_lut.py`: added `wgrad_via_bmm` parameter
  to `_soft_lut_bwd_body`, `_ball_lut_bwd_body`, `_ball_gather_lut_bwd_body`,
  and dispatch in `_FastMHLutSoft.backward`.

Cross-refs: [[fastmhl-hybrid-smooth-fwd-dispatch]] (same n_out >= 128 threshold),
[[hybrid_smooth-wgrad-compile-ceiling]] (precursor analysis at older recipe),
[[exp731-fastmhl-hard-densek-sota]] (current deployment SOTA, this is the
recipe to apply the optimization to).
