# STOPPED at step 3,200 / 4,000 — the bounded confidence gate attenuates by ~18×

Stopped deliberately on Anatoli's call, not a failure of the harness. Seed 2 (`exp_n_0179`) was
never launched. The run directory is kept because the partial curve *is* the evidence.

## The curve, and why it was not going to recover

| step | arm A (bounded gate) | baseline S5 | gap |
|---|---|---|---|
| 500 | 1.987700 | 1.852331 | +0.135369 |
| 1000 | 1.868186 | 1.693385 | +0.174801 |
| 1500 | 1.804922 | 1.607591 | +0.197331 |
| 2000 | 1.753607 | 1.548721 | +0.204886 |
| 2500 | 1.718596 | 1.501781 | +0.216815 |
| 3000 | 1.692834 | 1.467895 | +0.224939 |

**The gap widens at every single eval.** This is not a slow start that catches up — the three
baseline seeds only *separated* after step 2000, whereas here the deficit is already 0.135 at
step 500 and grows monotonically.

## The measured cause

Measured on real activations at anchor sizing (`nap = 8`), not estimated:

```
|d| margins   min 7.0e-06   p25 0.179   median 0.381   p75 0.656   max 2.65   mean 0.454
score bounded min 0.0053    p25 0.0326  median 0.0465  p75 0.0670  max 0.507  mean 0.0542
score margin  min 0.0026    p25 0.0943  median 0.165   p75 0.286   max 4.65   mean 0.229
```

`score = prod_j sigmoid(2|d_j|)` over **eight** factors. With a median margin of 0.38 each
factor is ~0.68, and `0.68^8 ≈ 0.05`. So **every gathered table row is multiplied by ~0.054 — an
18.4× attenuation.** (For reference, `0.5^8 = 0.0039` is the floor if margins were exactly zero;
we are above that but only by an order of magnitude.)

The consequence is in the backward, measured with the gate on and off at anchor sizing:

| | grad tables | grad decompress | grad x | grad compress |
|---|---|---|---|---|
| bounded | **0.052×** | 0.051× | 1.044× | 1.045× |
| margin | 0.262× | 0.250× | 1.332× | 1.342× |

**The bounded gate divides the learning signal reaching the LUT tables by ~19×** — matching the
forward attenuation exactly — while leaving the routing gradients essentially untouched. That is
a ~19× cut in the effective learning rate of the 75.5M parameters that do the actual work.

## Verdict: not a bug, a bad default

Everything the gate is documented to do, it does. Train and eval are bit-identical under the
gate (max |delta| = 0), and the native CUDA hard-eval kernel is correctly disabled
(`and not self.forward_confidence`), so there is no train/eval skew and no path double-applies
or skips the score. The implementation is sound.

What is wrong is the *choice* of `confidence_form="bounded"` at `nap = 8`: the score is a
product over `nap` factors, so its attenuation compounds with the anchor count, and at our
margin scale it lands at 0.054. For the bounded score to sit near 0.5 the margins would have to
be about **1.20** rather than 0.38 — roughly 3.2× wider than the compressed code currently
produces.

`margin` — `(sum_j |d_j|) * prob` — is 4.2× better (mean 0.229 vs 0.054) and cuts the table
gradient by 3.8× rather than 19×, but it still attenuates. It is the exact LookupFFN kernel
form, and the fact that it is needed here is informative: the `sum |d|` factor is not decorative,
it is what keeps the product from collapsing.
