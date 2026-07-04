---
name: beta1-load-bearing-at-bs128
description: "At bs=128, dropping β1 on LUT param group (lut_beta1=0.0) costs ~+0.12 bpb by step 800 vs the β1=0.9 baseline. Confirms Adam momentum is doing real work integrating per-row gradient history even with 128 sequences/batch, because LUT row updates remain sparse at any practical batch size."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# β1 is load-bearing for LUT params even at bs=128 (2026-05-16, exp370)

**Setup**: fork of exp363 (bs=128, full AdamW, final=1.4105 @ 8000 steps). Single change: `lut_beta1=0.0` on the LUT param group (decay + nodecay groups keep β1=0.9). All other hyperparams identical.

| step | exp370 (β1=0) | exp363 (β1=0.9) | Δ |
|------|--------|--------|---|
| 200  | 2.2698 | 2.2014 | +0.068 |
| 400  | 1.9717 | 1.9009 | +0.071 |
| 600  | 1.8643 | 1.7995 | +0.065 |
| 800  | 1.8524 | 1.7306 | **+0.122** |

Killed at step 800 once the verdict was clear — gap doubled at peak LR. Stable +0.07 during warmup, then divergence as warmup ends.

## Why this matters

β1=0.9 gives Adam an exponential moving average of gradients with effective window ≈ `1/(1-β1) = 10` micro-batches. At bs=128, this integrates **~10 × 128 = 1280 sequences of per-row gradient history** into each step. Drop β1 → each step only sees the current minibatch's 128 sequences, and per-row LUT signal goes sparse-noisy again because each row is only touched in 10-50% of any single batch.

**β1 is implicitly a sparse-aware integrator**. It's free, cheap, and effective. This explains both why exp356 (β1=0 at bs=16) failed and why exp370 (β1=0 at bs=128) also fails: row-sparsity persists at any practical batch size, and β1 was the only thing hiding it.

## How to apply

- Don't try to remove β1 from LUT optimizer settings, even at very large batch.
- The bs=128 → bs=16 "0.21 bpb gradient-quality gap" measured by exp367 should be reframed: bs=128 sees more *per-batch* data, but β1=0.9 is the *integrator* that turns that into stable per-row updates.
- The interesting direction for sparse-aware Adam is **not** to replace β1, but to *augment* it: smarter per-row integration than β1's blind exponential — e.g. EMA that advances only on touched steps, or per-row visit-corrected v_t.
- Up-knob direction tested in exp371: higher β1 (=0.99) at bs=16 to match bs=128's effective per-row history (~1600 vs ~1280 sequences). If this works, β1 is a free way to recover some of the bs=16↔bs=128 gap.
