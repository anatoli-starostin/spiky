---
name: mhlut-smooth-beats-tinymhlut
description: "Legacy MultiHeadLut(n_alternatives=3, smooth_mode=True) drop-in beats TinyMHLut(soft) by −0.005 bpb at bs=16 nanochat LM scale; new bs=16 LUT-LM SOTA exp386 = 1.6164 vs exp365 = 1.6215. Same params, same hyperparams."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Legacy MultiHeadLut(n_alt=3, smooth) beats TinyMHLut(soft) at bs=16 (2026-05-17, exp386)

**Fork:** exp365 (TinyMHLut(soft), bs=16, lut_lr=1e-3, noise=0, β1=0.9, 8000 steps). Single change: every TinyMultiHeadLut(soft) replaced with the legacy `MultiHeadLut(n_alternatives=3, smooth_mode=True)`. Same anchor sampling (CANONICAL_FULL_COVERAGE), same shapes (qkv NAP=6/tph=16, v NAP=8/tph=32, out_proj NAP=8/tph=128, residual NAP=6/tph=64), same init std, same RoPE, same optimizer.

**Result:** **exp386 = 1.6164 @ 43.06M, 8000 steps** vs exp365 = 1.6215 → **−0.0051 bpb (new bs=16 LUT-LM SOTA)**.

| step | exp386 | exp365 | Δ |
|------|--------|--------|---|
| 200  | 2.2903 | 2.2975 | −0.007 |
| 800  | 1.9000 | 1.9154 | −0.015 |
| 2000 | 1.7647 | 1.7804 | −0.016 |
| 4000 | 1.6811 | 1.6904 | −0.009 |
| 6000 | 1.6336 | 1.6418 | −0.008 |
| 7000 | 1.6220 | 1.6286 | −0.007 |
| 8000 | **1.6164** | 1.6215 | **−0.005** |

Lead of −7 to −16 millibits maintained across **all 40 evals** (warmup through end).

## Why this matters

- CIFAR-era Phase-2 canonical settings (`n_alt=3, smooth_mode=True`) are the historical "always fixed (critical for gradients)" recipe. They translate cleanly to nanochat-scale LM and beat the TinyMHLut(soft) drop-in we've been using since exp257.
- The TinyMHLut(soft) class was designed as a fast/scalable approximation of SoftMHLut(hard=True) (per `project_soft_lut_noise_regularization.md`). Inferred from this result: the *legacy* MultiHeadLut with `n_alternatives=3, smooth_mode=True` is producing *better gradients* than that approximation at bs=16 — likely because the multi-alt mechanism (3 alternative argmaxes per table) provides a built-in form of row exploration that the soft pipeline lacks.
- This is the *first* bs=16 experiment in the exp366-exp385 sweep that actually beat exp365 by a clear margin. None of the optimizer-side, β1-sweep, sparse-aware-Adam, load-balance, input-noise, or windowed-grad tricks matched this.

## How to apply

- **DO NOT adopt this as the new default.** The −0.005 bpb gain has a 3-4× inference cost and is NOT a training-only mechanism.
- Verified via post-hoc "hardened" eval on exp386's checkpoint:
  - as-trained (n_alt=3, smooth=True): val bpb = 1.6110 (50-batch eval)
  - smooth_mode=False at inference: 1.8606 (Δ=+0.2497, much WORSE than exp365)
  - n_alt=1, smooth=False at inference: 1.8606 (same)
- The trained weights *fundamentally depend* on smooth interpolation across 3 alternatives at inference. Dropping it destroys the model by 0.25 bpb.
- For a model meant to be bandwidth-friendly and matmul-free (the whole point of LUT-based models), the +5 millibit gain doesn't justify the 3× LUT bandwidth + multiplications at inference.
- **TinyMHLut(soft) (exp365 = 1.6215) remains the practical bs=16 reference.** It uses single-row lookup with no multiplications — true to the LUT-based-model design goal.
- Potential follow-up: distill exp386 → single-alt model (use exp386 as teacher for a TinyMHLut(soft) student). Untested.
