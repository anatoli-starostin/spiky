---
name: hard-mining-tiny-lut
description: "Sequence-level hard-example mining (forward bs=48, backward on top-16 hardest) is a consistent regression vs no-mining at the same backward batch on tiny LUT-LM. Doesn't capture the bs-scaling benefit."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Sequence-level hard-example mining: regression (2026-05-15, exp358, killed at step 1800)

exp358 = exp353 fork. Each step pulls bs=48 sequences, scores per-sequence mean CE via no-grad forward, selects top-16 hardest, then forward+backward on those 16. Otherwise identical to exp353 (β1=0.9, lut_lr=1e-3, noise=0, all-soft).

| step | exp358 (hard-mine 48→16) | exp353 (no mine, bs=16) | exp352 (no mine, bs=16, lr=3e-4) | exp348 (TRUE bs=48) | Δ vs exp353 |
|---|---|---|---|---|---|
| 200  | 2.3204 | 2.2945 | 2.3181 | 2.2980 | +0.0259 |
| 400  | 2.0788 | 2.0588 | 2.0833 | 2.0137 | +0.0200 |
| 800  | 1.9320 | 1.9174 | 1.9293 | 1.8532 | +0.0146 |
| 1200 | 1.8643 | 1.8502 | 1.8655 | 1.7737 | +0.0141 |
| 1400 | 1.8455 | 1.8285 | 1.8409 | 1.7424 | +0.0170 |

**Findings**:
1. Hard-mining is **consistently +0.014–0.026 bpb worse** than same-bs=16 no-mining (exp353).
2. Nowhere near closing the bs=48 gap (exp348). The "bs=48 forward → bs=16 backward" recipe captures **none** of the bs-scaling win.
3. Compute is wasted: 3× forward + 1× backward per step, for negative bpb effect.

**Why the recipe failed**:
- "Hard tokens" is a biased selection. Unbiased gradient = average over a uniform sample of tokens; picking only the hardest distorts the gradient direction toward worst-case examples rather than typical ones. Adam's gradient/Hessian estimates are based on the WRONG distribution.
- Big-batch wins because the gradient is a more accurate estimate of the TRUE mean gradient — not because it sees harder examples. Hard mining gives a worse estimate by sampling the tail of the distribution.
- In early training especially, "hardest" is dominated by random initialization noise, not informative signal.

**How to apply**: don't use hard-example mining as a variance-reduction substitute for bigger batch. The bs-scaling benefit is from **unbiased gradient averaging**, not from concentrating on hard examples. If we want bs=48 quality at bs=16 backward, the answer is more likely something like:
- True grad-accumulation (`total_batch_size=24576, device_batch_size=16`) — backward 3× per opt step, average grads → mathematically identical to bs=48 in expectation.
- Stochastic momentum / SWA on top of bs=16 — averages weight trajectories.
- Sparse-aware Adam that uses per-row counts.

Conclusion of optimizer/sampling sweep (exp353–exp358): **all the cheap optimizer-side levers (LR, β1, Lookahead, hard mining) max out at ±0.01 bpb**. The bs-scaling effect is fundamentally about *more gradient samples per row*, not about which samples or how Adam treats them. To beat bs=16 by more than 0.01, we likely need either (a) actual bigger batch, (b) actual grad_accum, or (c) sparse-LUT-aware optimizer rewrite.
