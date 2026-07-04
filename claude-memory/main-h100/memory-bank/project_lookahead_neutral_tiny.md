---
name: lookahead-neutral-tiny
description: "Lookahead (k=5, alpha=0.5) on the LUT param group is consistently slightly worse than no Lookahead at the same LR. Not a useful addition on top of AdamW(β1=0.9)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Lookahead on LUT params: small but consistent regression (2026-05-15, exp357, killed at step 2200)

exp357 = exp353 fork. Single change: enable Lookahead on the LUT param group only (k=5, α=0.5, Adam state NOT reset across slow snaps — standard form). Otherwise identical to exp353 (β1=0.9, lut_lr=1e-3, noise=0, bs=16).

| step | exp357 (Lookahead) | exp353 (no Lookahead, same LR) | exp352 (no Lookahead, lower LR) | Δ vs exp353 | Δ vs exp352 |
|---|---|---|---|---|---|
| 200  | 2.3014 | 2.2945 | 2.3181 | +0.0069 | −0.0167 |
| 400  | 2.0665 | 2.0588 | 2.0833 | +0.0077 | −0.0168 |
| 800  | 1.9207 | 1.9174 | 1.9293 | +0.0033 | −0.0086 |
| 1400 | 1.8325 | 1.8285 | 1.8409 | +0.0040 | −0.0084 |
| 1800 | 1.7976 | 1.7964 | 1.8074 | +0.0012 | −0.0098 |
| 2200 | 1.7701 | (n/a)  | 1.7775 | — | −0.0074 |

**Pattern**: Lookahead consistently trails the same-LR no-Lookahead run by +0.001–0.008 bpb. The gap is small but never positive. So Lookahead is a **mild regression** in this regime, not a help.

**Why this likely happens**: AdamW(β1=0.9) is already doing gradient-EMA smoothing; Lookahead adds *weight-EMA* smoothing on top. The two compound to slow effective progress more than they reduce variance. The reset (`fast := slow` every k steps) discards 5 steps' worth of Adam-driven movement and replaces it with a half-strength average, which is a net loss when the fast steps were already in a sensible direction (which momentum was ensuring).

**How to apply**:
- Don't use Lookahead with β1=0.9 on LUT params.
- The original Lookahead use-case is "with non-momentum optimizers" (SGD). Could be revisited as **Lookahead + AdamW(β1=0)** in case the combination outperforms either alone — but exp356 showed β1=0 alone is already worse, so this is a long shot.
- Sparse-LUT variance reduction probably needs a different mechanism: per-row gradient buffer / sparse-aware Adam / grad_accum simulation (= bigger effective batch).

**Hyperparam scope**: only k=5, α=0.5 tested. Larger k (10–20) or smaller α (0.2–0.3) could change the verdict, but the gap is small and the user concluded the picture is clear at this point.
