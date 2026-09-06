# The LookupFFN line: does a confidence gate help our LUT FFN?

Issue [#112](https://github.com/anatoli-starostin/spiky/issues/112). Branch
`research/ffn_replacement_fix`. Everything here is at the **anchor sizing** —
`sweep_s05_dout48_H4_tph256_c256_din32`: `H=4`, `tph=256`, `nap=8`, `d_in=32`, `d_out=48`,
4,000 steps, effective batch 24 sequences — scored on the corrected protocol
(`evaluate_bpb_fixed`, bs48 × 100, 2,451,456 held-out tokens).

**The reference points.** Baseline S5 (identical model, gate off, seed 1) = **1.434572**.
The sweep's vanilla-dense zero-line S0 = **1.474749**. The measured seed sd at this 4k proxy
budget is **0.0096**, and it is a *lower bound* — only 26.8% of parameters are re-drawn per
seed. Nothing below that margin is a result.

---

## What the gate is

Each gathered LUT row is multiplied by a smooth per-(token, table) score derived from the
routing margins `d = x[anchor_a] − x[anchor_b]`. The hard sign address is unchanged, the gate
is applied identically in train and eval, and it adds **no parameters**. Three forms, plus a
scale knob:

| form | score | at nap=8, measured |
|---|---|---|
| `bounded` | `∏_j σ(2\|d_j\|)` | mean 0.0542 |
| `margin` | `(Σ_j \|d_j\|) · ∏_j σ(2\|d_j\|)` — the exact LookupFFN kernel | mean 0.229 |
| `bounded_norm` | `(∏_j σ(2\|d_j\|))^(1/nap)` — geometric mean | mean 0.684 |
| `confidence_gain` | a constant multiplier on any of the above | default 1.0 |

---

## The arms

| arm | form | score mean | p75/p25 | result vs S5 |
|---|---|---|---|---|
| baseline S5 | gate off | — | — | **1.434572** |
| A | `bounded` | 0.0542 | 2.06 | **stopped at 3,200**, gap +0.225 and widening |
| A′ | `bounded_norm` | 0.6838 | 1.09 | **1.441122** (+0.006550) |
| B | `bounded_norm`, Light impl | 0.6838 | 1.09 | **1.477708** (+0.043136) |
| C | `bounded` × gain 12.61 | 0.6839 | 2.06 | **1.434988** (+0.000416) |
| D | `margin` × gain 2.99 | 0.6836 | 3.04 | _running_ |

A′ and C have **the same forward scale to within 0.011%** and differ only in selectivity, so
C − A′ isolates whether discriminating between confident and unconfident routings is worth
anything. A and A′ each moved two variables at once and cannot answer that on their own.

---

## Arm A: the product form collapses at nap=8

`bounded` is a product over `nap` factors. At the measured median margin of 0.381 each factor
is ≈0.68, and `0.68^8 ≈ 0.05`. Measured on 6,291,456 real margin vectors:

```
|d| margins    p25 0.179   median 0.381   p75 0.656   mean 0.454
score bounded  p25 0.0326  median 0.0465  p75 0.0670  mean 0.0542
```

Every gathered row is multiplied by ~0.054 — an **18.4× forward attenuation** — and the
gradient reaching the tables is cut to **0.052×** of gate-off while the routing gradients are
untouched (1.044×). The run's gap to S5 widened at every eval (+0.135 → +0.225) and it was
stopped at step 3,200.

**Verdict: not a bug, a bad default.** Train and eval are bit-identical under the gate, the
native CUDA hard-eval kernel is correctly disabled, no path double-applies the score. The
implementation does what it documents; the *choice* of a product form at nap=8 is what fails.

### It is a nap problem, not a "gates are bad" problem

Measured across `nap` on the empirical margin distribution:

```
 nap      bounded    bounded_norm       margin
   1     0.692675        0.692675     0.355763
   2     0.480238        0.687666     0.493968
   4     0.230857        0.685143     0.475681
   8     0.053137        0.683605     0.218524
  16     0.002820        0.682954     0.023177
```

`bounded` falls **246×** between nap=1 and nap=16. `bounded_norm` drifts 1.4% — nap-invariant,
as designed. `margin` is **non-monotone**, peaking near nap=2 and collapsing after, so it is
not a general fix either: at nap=8 it is merely a milder failure (0.219, a 4.6× attenuation),
not a correct one.

### Would it have healed itself in a longer run?

No. Margins *do* widen as training proceeds — measured by re-running the capture through the
trained baseline checkpoint (`dump_margins.py --trained`):

| | \|d\| median | \|d\| mean | `bounded` | `bounded_norm` | `margin` |
|---|---|---|---|---|---|
| at init | 0.381 | 0.455 | 0.0542 | 0.6838 | 0.229 |
| after 4,000 steps | 0.643 | 0.766 | 0.1302 | 0.7592 | 0.944 |

Margins grow **1.69×**, and `bounded`'s score rises 2.40× — so its attenuation eases from
18.4× to **7.7×**. It never approaches 1. For `bounded` to reach ~0.5 the median margin would
have to be ≈1.201, a further **1.87×** beyond where a *full, healthy* run ends up. And this is
the **gate-off** trajectory, i.e. the optimistic case: under the gate the FFN contributes ~18×
less, so the compressed code has correspondingly less pressure to spread at all. Arm A was not
merely slow — it was structurally stuck.

A side-observation worth keeping: **`margin` self-normalises through training** (0.229 → 0.944,
essentially unit scale by the end) because its `Σ\|d\|` factor grows as margins widen. That is
presumably why LookupFFN gets away with it, and it is a real argument for the `margin` form
over `bounded` at whatever nap the original paper used — though at our nap=8 it still starts
4.6× attenuated, and its nap-dependence is non-monotone.

---

## Arm A′: the normalisation works — and reveals the real question

**1.441122 vs 1.434572, +0.006550 — inside the 0.0096 seed sd, so indistinguishable at one
seed.** It beats the vanilla-dense zero-line by −0.0336, as the baseline does. The gap never
exceeded +0.007 at any eval:

| step | A′ | S5 | gap A′ | gap A |
|---|---|---|---|---|
| 500 | 1.8554 | 1.8523 | +0.0031 | +0.1354 |
| 1000 | 1.6975 | 1.6934 | +0.0041 | +0.1748 |
| 1500 | 1.6108 | 1.6076 | +0.0032 | +0.1973 |
| 2000 | 1.5538 | 1.5487 | +0.0051 | +0.2049 |
| 2500 | 1.5079 | 1.5018 | +0.0061 | +0.2168 |
| 3000 | 1.4748 | 1.4679 | +0.0069 | +0.2249 |
| final | **1.4411** | **1.4346** | **+0.0066** | (stopped) |

**But parity is not the same as usefulness.** The normalisation also flattened the gate almost
to a constant:

| | p75/p25 | within-token CV |
|---|---|---|
| `bounded` | 2.06 | 0.536 |
| `margin` | 3.03 | 0.870 |
| `bounded_norm` | **1.09** | **0.061** |

The within-token CV is the honest number: of the score's total variance, ~84% lies *across
tables within one token* (the part that actually reweights the 256-table ensemble sum) and
~16% across tokens (which merely rescales that token's whole FFN output). At CV 0.061 the
gate is close to a constant multiplier — and **a constant multiplier is exactly absorbable
into the linear, zero-initialised `decompress` that follows it**, so it cannot change anything
the model could not already express. Independently confirmed: with `bounded_norm` the gate
barely rotates Fast's gradient at all, `cos(gated, gate-off) = +0.990`.

So A′ says *a near-constant gate is harmless*. It does **not** say gating helps. That is what
arm C is for.

---

## Triaged out without spending GPU

**Sharpening the geometric mean.** `exp(mean_j logsigmoid(2·β·\|d_j\|))` with β>1 looked like
the obvious way to buy back selectivity while keeping nap-invariance. Measured, it does not
work — the spread *peaks* and then falls, because every factor saturates toward 1:

```
   beta       mean        p25        p75    p75/p25       CV
    1.0     0.6838     0.6518     0.7133       1.09    0.067
    2.0     0.7837     0.7442     0.8233       1.11    0.073
    4.0     0.8728     0.8348     0.9144       1.10    0.065
   12.0     0.9538     0.9278     0.9892       1.07    0.043
```

Target was `bounded`'s 2.06; the best β reaches 1.11. **Dead — no run spent on it.** The
tension is structural: in a product form, spread and scale both come from the compounding, so
you cannot normalise the scale away and keep the spread. Which is exactly why the scale had to
become a separate knob (`confidence_gain`) rather than being folded into the form.

**Compensating the residual 0.67× table-gradient attenuation** (Part 1b). `bounded_norm` leaves
`grad_tables` at 0.671× of gate-off. Setting `confidence_gain = 1.46` would restore unit mean.
I expect this to be **inert** and did not queue it: the trainer is AdamW, whose update is
`m̂/(√v̂+ε)`, so a uniform rescale of a parameter's gradient is almost entirely absorbed. The
quantity that actually matters is the *forward* magnitude — how much decompress must grow
before the FFN contributes — and 1.46× is nothing next to arm A's 18.4×. Cheap to test if
wanted; I would not spend an hour on it before arm C.

---

## Arm B: Light (detached routing)

Light's defining property is that the routing address is `sign(d).detach()` — no STE, no
temperature surrogate — so `x` receives gradient **only** through the confidence score.
Measured before launching, at the anchor sizing with `bounded_norm`:

```
                        |out|     grad_x   grad_tables   grad_dec   grad_com
  fast, gate off       0.03525    0.12975     294.2        20.261     2.6901
  fast + bounded_norm  0.035242   0.14138     195.89       13.49      2.9347
  light + bounded_norm 0.035242   0.023045    195.89       13.49      0.47296
```

* Light's **table and decompress gradients are identical to Fast's** (195.89 / 13.49) — those
  flow through the gathered rows and never touch the surrogate, so the 75.5M table parameters
  are not handicapped at all. Forward outputs match exactly too.
* The handicap is confined to the input side: **16.3%** of Fast's `grad_x`.
* But a norm ratio is the wrong statistic under AdamW. Direction:
  **`cos(light, fast) = +0.576`** on `compress.weight` (+0.542 on the bias).

Substantially aligned, but not a rescaled copy — the surrogate carries real information Light
cannot see. Prediction on that evidence: **arm B trains rather than collapses, and lags.** It
is not arm A's failure mode; there is no forward attenuation here.

Param count 104,952,576 = baseline − 12 (Light has no learnable temperatures: 2 per layer ×
6 layers). Projections and table budget match Fast exactly.

### Result: 1.477708 — the prediction held

**+0.043136 vs baseline S5, which is 4.5× the seed sd — a real regression, not noise.** It is
also marginally *worse* than the vanilla-dense zero-line (1.474749, +0.00296), so at this
budget Light gives up the whole advantage the LUT FFN had over a dense FFN.

| step | B | S5 | gap B | gap A′ | gap A |
|---|---|---|---|---|---|
| 500 | 1.8670 | 1.8523 | +0.0147 | +0.0031 | +0.1354 |
| final | **1.4777** | **1.4346** | **+0.0431** | +0.0066 | (stopped) |

So the directional surrogate is **worth about 0.043 bpb** here. That is the honest measure of
what Fast's temperature-surrogate backward buys over LookupFFN's pure-autograd one on our
geometry — and it is exactly the middle outcome the pre-launch measurement pointed to:
`cos(light, fast) = +0.576` said the score path is *aligned but not equivalent*, so B should
train (unlike arm A, which had no forward signal at all) and lag (unlike A′, which had the
full backward). It did both.

**Caveat, stated rather than buried:** one seed. +0.043 is comfortably outside the noise, so
the *direction* is safe; the *magnitude* is one sample and should not be quoted to three
decimals as if it were a converged estimate.

---

## Arm C: does selectivity buy anything?

`bounded` with its selectivity fully intact and its scale corrected by a measured constant:
gain **12.61**, chosen because bounded means 0.05423589 and bounded_norm 0.68383604 on the
same margins — a ratio of 12.6086, giving 0.683915, within **0.011%** of A′'s mean. So C vs A′
is a one-variable comparison.

| | C ~ A′ | C < A′ | C > A′ |
|---|---|---|---|
| reading | selectivity is inert; the score mechanism buys nothing on our geometry | selectivity helps, and arm A failed purely on scale | down-weighting uncertain rows costs more than it saves |

### Result: 1.434988 — +0.000416, the closest any arm gets to the baseline

Arm C tracked S5 essentially exactly, at every eval, and ended **0.0004** away:

| step | C | S5 | gap C | gap A′ | gap A |
|---|---|---|---|---|---|
| 500 | 1.8523 | 1.8523 | −0.00003 | +0.0031 | +0.1354 |
| 1000 | 1.6931 | 1.6934 | −0.00029 | +0.0041 | +0.1748 |
| 1500 | 1.6067 | 1.6076 | −0.00089 | +0.0032 | +0.1973 |
| 2000 | 1.5484 | 1.5487 | −0.00032 | +0.0051 | +0.2049 |
| 2500 | 1.5035 | 1.5018 | +0.00172 | +0.0061 | +0.2168 |
| 3000 | 1.4691 | 1.4679 | +0.00121 | +0.0069 | +0.2249 |
| final | **1.4350** | **1.4346** | **+0.00042** | +0.0066 | (stopped) |

**This settles what arm A's failure was: purely scale.** The identical score shape that
diverged to +0.225 lands at +0.0004 once multiplied by a constant. Nothing about the gate's
*form* was wrong — only its magnitude, and the magnitude is a one-line knob.

**What it does not settle** is whether selectivity is *useful*. C is numerically the closest
arm to baseline and A′ is +0.0066 behind it, but that difference is itself **inside the
0.0096 seed sd**, so C > A′ is not a claim I can make from one seed each. The defensible
statement is narrower and still worth having:

> At matched forward scale, the confidence gate is **neutral** — it neither helps nor hurts
> beyond noise, whether it discriminates strongly (C, CV 0.584) or barely at all (A′, CV
> 0.061). Every difference among the scaled arms is smaller than the seed noise; the only
> effect large enough to see is the one caused by getting the scale wrong.

Since the gate costs a per-token score computation and buys nothing measurable, the practical
recommendation is to **leave `forward_confidence` off** at this sizing — and if it is used, to
treat `confidence_gain` as mandatory rather than optional.

---

## Code

| commit | what |
|---|---|
| `30fc396e` | arm A stop, `STOPPED.md`, `diag_confidence_gate.py`, `diag_confidence_backward.py` |
| `f1276ec4` | `bounded_norm` + its analytic backward + tests |
| `5eb9bcec` | issue #112 filed; wrong `#111` refs corrected; `dump_margins.py`, `diag_confidence_forms.py` |
| `c063d0c7` | `diag_light_vs_fast.py` |
| `1fd5b6aa` | `confidence_gain`, threaded everywhere, + tests; arm B config |
| `c0fc3c33` | arm C config |

361 lutorch tests pass. `bounded` was never redefined, so every earlier run stays
bit-reproducible; `confidence_gain=1.0` is bit-identical to no gain, forward and backward.
