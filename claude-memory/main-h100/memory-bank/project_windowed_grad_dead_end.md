---
name: windowed-grad-dead-end
description: "Sliding-window gradient smoothing on LUT params (W=8 buffer of last 8 micro-batch grads, mean substitutes for p.grad before Adam step) — tested both on top of and replacing Adam β1=0.9 momentum; both regressed vs no-smoothing baseline. Closes the \"gradient-space variance reduction\" direction at fixed phys-batch."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Windowed-grad smoothing on LUT params — dead end (2026-05-16, exp368 + exp369)

**Idea**: maintain a ring buffer of the last W per-parameter LUT gradients; on each step, replace `p.grad` with the mean of the buffer before `optimizer.step()`. Hypothesis: denser per-row Adam statistics + lower variance → closer to bs=128 trajectory at near-1× wall-clock vs true grad_accum=8 (which costs 8× wall-clock; exp367 already confirmed accum=8 reproduces bs=128 within +0.01).

Implementation: `nanochat_exps/exp368_windowed_grad_w8/windowed_grad.py` — `WindowedGradSmoother`, ring buffer over `lut_params` only (29 M params, 924.8 MB at W=8 fp32). Other params untouched.

## exp368: W=8 + β1=0.9 (smoothing on top of Adam momentum)
**Result**: net regression vs no-smoothing (exp365) at every step.

| step | exp368 | exp365 | Δ |
|------|--------|--------|---|
| 200  | 2.3490 | 2.2975 | +0.052 |
| 400  | 2.1342 | 2.0627 | +0.072 |
| 600  | 2.0307 | 1.9665 | +0.064 |
| 800  | 1.9820 | 1.9154 | +0.067 |
| 1000 | 1.9634 | 1.8786 | +0.085 |
| 1200 | 1.9474 | 1.8498 | +0.098 |

Killed at step 1200. **Root cause: double-smoothing.** β1=0.9 already provides an exponential gradient EMA with effective window ~10; adding a uniform W=8 sliding window on top creates an unnecessarily wide filter with stale-grad bias (8-step-old gradients computed against different LUT row values). Same lesson as exp357 (Lookahead on top of β1=0.9).

## exp369: W=8 + β1=0 (smoothing REPLACES momentum)
**Result**: matches exp365 early (steps 200–800), then diverges + becomes unstable.

| step | exp369 (β1=0, W=8) | exp365 | Δ |
|------|--------|--------|---|
| 200  | 2.3058 | 2.2975 | +0.008 |
| 400  | 2.0641 | 2.0627 | +0.001 |
| 600  | 1.9713 | 1.9665 | +0.005 |
| 800  | 1.9464 | 1.9154 | +0.031 |
| 1000 | 1.8951 | 1.8786 | +0.017 |
| 1200 | **1.9688** | 1.8498 | **+0.119** ← spike |
| 1400 | 1.9510 | 1.8288 | +0.122 |
| 1600 | 1.8920 | 1.8081 | +0.084 |
| 1800 | 1.8711 | 1.7967 | +0.074 |
| 2000 | 1.8756 | 1.7804 | +0.095 |

Killed at step 2000. Early-phase parity (steps 200–800 within ±0.03) suggests **W=8 uniform window ≈ β1=0.9 exponential EMA at low LR**. But during the peak-LR phase (around step 1200, post-warmup) the uniform window can't damp gradient noise as well as adaptive EMA → instability spike, then ~+0.08 underperformance the rest of the way.

## Bottom line

**Gradient-space variance reduction on LUT params at fixed phys-batch is a closed direction**. Two reasons:
1. Adam's β1=0.9 already does the variance reduction at no extra cost; you can't double-smooth and expect gain.
2. Replacing momentum with a uniform window destabilises at peak LR — uniform isn't adaptive.

**How to apply:**
- Don't try variants of windowed-grad (different W, different param subsets, EMA mix vs uniform). They'll all share these two failure modes.
- The only working lever at bs=16 remains **actually seeing more data** — either true grad_accum (8× wall-clock, exp367) or breakthroughs that need fewer samples per row (sparse-aware Adam with per-row visit-corrected stats — still untried, see below).
- Code (`windowed_grad.py`) is preserved in exp368/exp369 folders for archaeology; don't reuse.

## What's still untried for bs=16 convergence speed
- **Sparse-aware Adam**: per-row visit counters on LUT weights, only update m_t/v_t for visited rows in a given step, scale step by 1/√visits. Truly fixes the "row touched once vs row touched 100×" Adam-state imbalance that windowed grad was a (failed) proxy for.
- **Vanilla-teacher distillation** from exp328 logits — denser per-token signal, orthogonal to batch size.
