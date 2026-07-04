---
name: lut-convergence-levers-dead-ends
description: "bs=16 LUT-LM bottleneck is gradient quality, not capacity/conditioning/decoder. Dead ends (2026-05-21): Gauss-Newton optimizer, parallel main-effect/Linear branches, frozen decoder. LION lr=2e-4 + β2∈[0.93,0.95] confirmed optimal."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# bs=16 LUT-LM: convergence is gradient-quality-limited; cheap levers exhausted (2026-05-21)

All fork exp453 (LION 0.9/0.95, lut_lr=2e-4, bs=16, 8K, 1.4967). Confirms the standing conclusion: only effective-batch (exp364 bs=192) or (untried) distillation lowers the floor; everything cheap is ±noise.

## Optimizer/lr CONFIRMED (no win)
- **lut_lr=2e-4 is optimal** (exp467/468/469 = 1e-4/4e-4/8e-4): all monotonically worse — 1e-4 too small, 4e-4/8e-4 overshoot. LION is non-adaptive (step=±lr per coord, only sign/direction varies), so lr is load-bearing; 2e-4 (≈ AdamW's realized lr) is right.
- **β₂ plateau** (exp454 β=(0.9,0.93)=1.4969 ≈ exp453 0.95=1.4967): sweet spot spans [0.93,0.95].

## DEAD ENDS
1. **Gauss-Newton optimizer** (exp455): `ΔW=−lr·m/(count+λ)`, count=per-row visit count = exact diagonal Hessian of the linear-in-weights LUT (exposed via new `enable_visit_count_stash` in tiny_multi_head_lut.py, default off). Landed ~+0.013, BETWEEN SGD (+0.034) and LION/Adam = "denoised SGD." Lesson: GN keeps gradient *magnitude*; the winners discard it (sign/√v̂). **The bottleneck is gradient NOISE, not conditioning** — count is right curvature for a noiseless problem; the right statistic for a noisy one is variance/sign. Adam/sign already handle conditioning.
2. **Parallel "main-effect" branches** (exp456 NAP=1 tables on all pairs; exp457/458 VectorToDominance→Linear): zero-init dominance branch = +0.027 drag. The base NAP=6 LUT already encodes pairwise dominance (its anchors ARE sign comparisons) → a first-order term is redundant. (exp457 init bug: default nn.Linear init made the branch dominate at init, +0.083 — always zero-init additive residual branches.)
3. **LUT(x) + Linear(x)** (exp463): only ~−0.001 even with a full dense matmul per module → **the LUT body is NOT linearly deficient** (good negative; the gap to vanilla isn't "LUTs can't do linear maps"). Confirms the matmul-free thesis.
4. **Frozen decoder** (exp459): froze unembedder at converged value (verified all 96 anchor buffers byte-identical, so the body could reproduce the teacher rep). Huge early lead (−0.215@200) but final 1.5138 (+0.017) — the body converges to a slightly WORSE floor than full co-training. → the gap lives in the **LUT body's convergence (gradient quality)**, not the decoder.

## Net
Cheap optimizer/capacity/conditioning/decoder levers are exhausted (±noise). The one untried lever that adds INFORMATION (not just reweights noisy gradients): **distillation** (denser per-token supervision). See [[lut-optimizer-sweep]], [[lut-prenorm-is-magnitude-calibration]].
