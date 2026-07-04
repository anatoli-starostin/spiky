---
name: lut-beta1-ablation
description: "AdamW β1=0 on the LUT param group is consistently worse than β1=0.9, even with 3.3× higher LR to compensate. Momentum is load-bearing for sparse-row LUT training."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# LUT-group β1 ablation: momentum is load-bearing (2026-05-15, exp356, killed at step 3000)

exp356 = exp353 fork. Single change: LUT param group `betas[0]` 0.9 → 0.0 (`lut_beta1=0.0`). Tests whether Adam's first-moment EMA helps or just adds stale signal on sparse rows.

| step | exp356 (β1=0, lr=1e-3) | exp353 (β1=0.9, lr=1e-3) | exp352 (β1=0.9, lr=3e-4) | Δ vs exp353 | Δ vs exp352 |
|---|---|---|---|---|---|
| 200  | 2.3240 | 2.2945 | 2.3181 | +0.0295 | +0.0059 |
| 400  | 2.0945 | 2.0588 | 2.0833 | +0.0357 | +0.0112 |
| 800  | 1.9460 | 1.9174 | 1.9293 | +0.0286 | +0.0167 |
| 1400 | 1.8492 | 1.8285 | 1.8409 | +0.0207 | +0.0083 |
| 1800 | 1.8075 | 1.7964 | 1.8074 | +0.0111 | +0.0001 |
| 2400 | 1.7685 | 1.7575 | 1.7673 | +0.0110 | +0.0012 |
| 3000 | 1.7432 | (killed) | 1.7363 | — | +0.0069 |

**Findings**:
1. β1=0 at lut_lr=1e-3 is **consistently worse** than β1=0.9 at lut_lr=1e-3 by +0.011–0.036 bpb.
2. β1=0 at lut_lr=1e-3 ≈ β1=0.9 at lut_lr=3e-4 (i.e., the exp352 baseline): momentum was doing ~equivalent work to 3.3× LR.
3. Step 1200 showed a transient eval spike (1.9836) that recovered by step 1400 — likely eval noise with eval_steps=10; was *not* a real divergence.

**How to apply**: keep `β1=0.9` on the LUT param group. Momentum is **load-bearing** for sparse-row LUT training, contrary to the staleness-dilution worry. The implication for Lookahead: it would need to do MORE than replace momentum (since momentum is already pulling its weight) — it would need to add an additional smoothing layer on top.

**Open**: whether β1=0.5, 0.7, 0.95, 0.99 could outperform 0.9. Worth a separate sweep if budget allows. The current data only tells us β1=0 is worse, not that 0.9 is optimal.
