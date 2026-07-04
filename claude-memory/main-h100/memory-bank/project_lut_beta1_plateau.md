---
name: lut-beta1-plateau
description: "β1 sweep on LUT param group at bs=16 — β1 ∈ [0.80, 0.92] is a flat plateau indistinguishable from β1=0.9 (Δ ≤ ±0.005 across full trajectories); β1=0.95 is a mild regression (~+0.005); β1=0.99 broke (warmup-lag instability, +0.30 bpb deficit that never closes). Stick to β1=0.9 default."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# β1 sweep on LUT params at bs=16 (2026-05-16, exp371–exp374)

Motivation: bs=128 → bs=16 introduces ~0.21 bpb gap (exp367 proved it's gradient-quality). β1=0.9 at bs=128 integrates ~10 × 128 = 1280 sequences of per-row history; at bs=16 only ~160. Tested whether raising β1 at bs=16 recovers history → closes gap.

Setup: all forks of exp365 (1.6215, β1=0.9). Single change per fork: `lut_beta1` value. Other hyperparams identical (lut_lr=1e-3, noise=0, bs=16). 8000 steps, ~0.6h each. Reference at all steps is exp365 (β1=0.9).

| exp | β1 | window (×16=seq) | result |
|-----|-----|---|--------|
| **exp365** | **0.90** | 10 × 16 = **160** | **1.6215 final** (reference) |
| exp371 | 0.99 | 100 × 16 = 1600 | **broken** — warmup lag: Δ=+0.30 at step 800, never recovers. Killed @ 2000 |
| exp372 | 0.95 | 20 × 16 = 320 | mild regression: Δ=+0.005 steady, sweep ended at 3800 with +0.005 |
| exp373 | 0.92 | 12.5 × 16 = 200 | neutral: Δ ∈ [+0.002, +0.006], sweep ended at 1000 |
| exp374 | 0.80 | 5 × 16 = 80 | neutral: Δ ∈ [−0.003, +0.005], sweep ended at 6400 with −0.001 |

## Findings

**β1 ∈ [0.80, 0.92] is a flat plateau** indistinguishable from β1=0.9 — ±0.005 noise.
- Going much lower than 0.8 likely follows the same trend (β1=0 was tested earlier at exp356: ~+0.01–0.03 worse but not catastrophic).
- Going to 0.95: small but consistent regression. The lag starts to bite.
- Going to 0.99: catastrophic warmup-lag instability — the long EMA can't track the rising warmup LR, accumulates +0.30 bpb of deficit during the 800-step warmup that never closes.

## Why "increasing β1 to recover history" failed

The hypothesis was that matching bs=128's per-row history (~1280 seq) by setting β1≈0.9875 would close the bs=16↔bs=128 gap. It doesn't, for two reasons:

1. **Warmup-lag**: β1 high → optimizer state lags the LR schedule. During 800-step warmup the m_t built up against tiny LRs is wrong for the post-warmup peak LR. Permanent ~0.30 bpb deficit.

2. **Even if you fix warmup**: β1 averages *across* rows indiscriminately. The actual per-row issue is sparsity within a batch — a row touched 16× per batch vs touched 0× in 5 batches need different treatment. A blind exponential filter can't distinguish these. **The history bs=128 has is *uniformly-sampled* per row** because 128 sequences hit more rows per batch. β1 can't fake uniform sampling.

## How to apply

- **Use β1=0.9 as default for LUT params at bs=16** (and likely all batch sizes).
- Don't try to recover the bs=16↔bs=128 gradient-quality gap via β1 alone. It's not the right knob.
- The actual fix is *per-row* sparse-aware: see [project_beta1_load_bearing_at_bs128.md](project_beta1_load_bearing_at_bs128.md) for the math intuition, and the planned SparseRowAdamW implementation which uses per-row visit counts `c_r` and `β1^{c_r}` weighting for the correct generalization.
