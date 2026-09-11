# learned_margin: a fully parametric, learnable confidence form

Branch `research/ffn_replacement_fix`. Run: **exp_g_0247**, an exp_g_0193 fork.

## Motivation

Every confidence form tested so far carries hand-picked constants chosen by sweeping a cached margin
dump:

* the 2 in sigmoid(2m)
* sharp_margin's exponent (1.75 / 3.0)
* the confidence_gain (3.9 / 25.4 / 37.4)
* tanh_margin's a = 2

Those are fragile choices to defend in a paper. This arm replaces all of them with learned per-layer
parameters. It is initialised exactly at the margin baseline, so the run starts as a literal
reproduction of exp_g_0193 and can only depart from it if departing helps.

## Form

    s = exp(g) · (Σ_j m_j) · (Π_j σ(β m_j))^γ,     m = |d|,  β = exp(log β),  γ = exp(log γ)

**One (g, log β, log γ) triple per LightMultiHeadLUT module, i.e. per layer** — not per head, not per
table. The model therefore has 6 × 3 = 18 new scalars. They are registered as `confidence_g`,
`confidence_log_beta` and `confidence_log_gamma` only when the form is `learned_margin`, so every other
form's parameters and state_dict keys are unchanged.

Nesting:

* margin ≡ (g = 0, β = 2, γ = 1), the init
* sharp_margin γ=1.75, gain 3.9 ≡ (g = log 3.9, β = 2, γ = 1.75); γ=3, gain 25.4 ≡ (log 25.4, 2, 3)
* bounded is not nested (it has no Σm factor)

The form is discontinuous at cell boundaries for every finite β, γ: σ(0) = 0.5 > 0.

## Numerics

**Log space.** The implementation computes

    s = (Σ_j m_j) · exp(g + γ · Σ_j logsigmoid(β m_j))

* **The power stays in log space.** The probability power is never formed as P^γ, even though P is
  typically ~0.05 and γ is learnable.
* **The Σm factor stays outside the exp.** This is the same log-space computation as
  `exp(g + log Σm + γ log P)` but never forms log(Σm):
  * **All-zero margins** (Σm = 0) give s = 0 exactly, with finite gradients, and no clamp or epsilon.
    A clamp would change the value and zero the gradient at that corner.
  * **Bit-identity at init.** At (0, 2, 1) the op sequence is literally margin's
    (`Σm * exp(Σ logsigmoid(2m))`, with `0 + x` and `1 · x` exact). The init is therefore
    bit-identical to margin, not merely within tolerance.
* **Underflow.** Σm ≥ 0 cannot overflow at this scale. For β > 0, log P ≥ NAP·log 0.5 = −5.55 at
  NAP=8, so fp32 underflow of s (g + γ log P < ≈ −103) would need γ ≳ 18 or g ≲ −97.

**Positivity.** β and γ stay positive through the log parameterisation. They are initialised from Python
floats via `math.log`. exp(float32(log 2)) is exactly 2.0 and exp(0) exactly 1.0, verified on CPU and
CUDA. Creating them consumes no RNG, so every other parameter initialises bit-identically to
exp_g_0193's.

**Derivative route: autograd only.** The analytic sibling `_confidence_score_and_dscore` raises
`NotImplementedError` for this form, because it cannot return the g/β/γ gradients.
FastMultiHeadLut and BH4MultiHeadLUT refuse the form. LightMultiHeadLUT computes the score through
autograd; `light_multi_head_lut.py` imports only `_confidence_score`. A test replaces the analytic
functions with ones that raise and runs n=1 and n=2 training steps.

Closed-form gradients, pinned in the tests against autograd, gradcheck and finite differences:

    ∂s/∂m_j      = exp(g) P^γ + s γ β σ(−β m_j)
    ∂s/∂g        = s
    ∂s/∂log β    = s γ Σ_j (β m_j) σ(−β m_j)
    ∂s/∂log γ    = s γ log P

**Weight decay: excluded.** The three scalars are 0-dim, so train.py's `setup_optimizer` puts them
in the weight_decay = 0 group through its `ndim < 2` rule. They are also LightMultiHeadLUT's own
parameters, which the standard `lut_tables_no_decay: true` exempts. Verified by executing the fork's
own `setup_optimizer` on the built model.

**Identifiability caveat, stated up front.** The gain g is only weakly identified: a global rescale
of the score can be absorbed by the tables and by decompress. Its trajectory is reported, but no
conclusion is drawn from it. Whether (β, γ) are similarly degenerate is measured after the run, not
assumed.

## Plumbing

Config: `lut_confidence_form: learned_margin` plus three REQUIRED keys, `lut_learned_margin_g_init`,
`lut_learned_margin_beta_init` and `lut_learned_margin_gamma_init`. model_build refuses a partial set,
or the keys with any other form. It passes them as `LightMultiHeadLUT(learned_margin_init=(g, β, γ))`
and prints each layer's init as read back from the parameters. `lut_confidence_gain` stays 1.0;
LightMHL refuses any other gain for this form, since exp(g) is the gain.

CUDA: form id 6 is not in the native scored-eval kernel, so no-grad eval takes the torch path.

## Logging

The trainer is exp_g_0193's train.py plus one strictly additive, read-only block, the pattern
exp_g_0195 used for tau:

* metrics.csv gains columns `lm_g_L*`, `lm_beta_L*` and `lm_gamma_L*` after the existing ones, written
  at every eval
* summary.json gains a `learned_confidence` entry with per-layer init and final values

The block uses no RNG, builds no graph, and never touches the optimiser.

## Pre-training verification

All checks passed; code a5f6b847.

* **Tests:** 320 on CUDA, 91 of them new. The four gradients (∂s/∂m, ∂s/∂g, ∂s/∂log β, ∂s/∂log γ)
  match their closed forms under autograd, gradcheck and finite differences, at random, zero,
  near-zero, near-equal and all-zero margins.
* **Regression:** 110/110 bit-identical before vs after for bounded, bounded_norm, margin,
  min_margin, tanh_margin and sharp_margin; FastMHL CPU check 60/60.
* **Init == exp_g_0193:** on 0193's trained checkpoint, every score, block output and the loss are
  torch.equal (max discrepancy 0). Fresh init with the same seed is bit-identical on every shared
  tensor.
* **Eval path:** at identical weights, margin's native kernel and this form's torch fallback differ
  by a relative 2.6e-5 in loss. Every non-native form carries this difference.
* **Weight decay:** all 18 scalars are in the weight_decay = 0 group.
* **Gradients reach the parameters:** at step 1 of a fresh model every LUT gradient is exactly 0,
  because decompress is zero-initialised. With upstream signal, all 18 get nonzero, finite gradients:
  on 0193's trained weights, and after 5 real AdamW steps.
* **No prod/cumprod:** none in the training step's op trace.
* **Continuity at init:** the boundary jump is 4.11%, margin's number.

## Results — exp_g_0247 (one seed; no LUT-seed replicates at this geometry)

**corrected_val_bpb 1.169344**, 0.939 h. The microbenchmark predicted 0.91 h (192 vs 186 ms per step;
no torch.prod on this path). Deltas are in units of the vanilla 2-seed range 0.00335, the budget-law
residual sd 0.0035, and the 4K LUT 3-seed sd 0.009642 (a lower bound):

| reference | bpb | delta | noise units |
|---|---|---|---|
| exp_g_0193 margin | 1.172852 | −0.0035 | −1.05 / −1.00 / −0.36 |
| exp_g_0245 sharp γ1.75 | 1.167381 | +0.0020 | +0.59 / +0.56 / +0.20 |
| exp_g_0244 tanh_margin | 1.186730 | −0.0174 | −5.2 / −5.0 / −1.8 |
| exp_g_0243 min_margin | 1.197236 | −0.0279 | −8.3 / −8.0 / −2.9 |
| exp_g_0195 n=2 | 1.160637 | +0.0087 | +2.6 / +2.5 / +0.9 |

The curve tracks exp_g_0193 at the start: +0.00096 at step 500, as a bit-identical init implies. It
is worse at only 3 of 32 eval steps (500, 1,500, 2,000) and better from step 2,500 on: −0.0018 at
4,000 and −0.0035 at 16,000. Against margin and against sharp_margin γ1.75 the difference is
**not resolved** (≈1 unit on the 16K figures, 0.36 on the LUT figure).

### Learned parameters (`analyze_learned_margin.py`, `metrics.csv`)

| layer | g | β | γ | drift 14K→16K (β / γ) | 0.5^(8γ) | 0.5^γ |
|---|---|---|---|---|---|---|
| L0 | −0.256 | 1.801 | 1.356 | +0.025 / +0.001 | 5.4e-4 | 0.391 |
| L1 | −0.252 | 2.102 | 1.468 | +0.026 / +0.015 | 2.9e-4 | 0.362 |
| L2 | −0.179 | 2.214 | 1.580 | +0.024 / +0.018 | 1.6e-4 | 0.334 |
| L3 | −0.148 | 2.132 | 1.482 | +0.017 / +0.014 | 2.7e-4 | 0.358 |
| L4 | −0.094 | 2.073 | 1.383 | +0.013 / +0.011 | 4.7e-4 | 0.383 |
| L5 | +0.005 | 2.115 | 1.044 | +0.012 / +0.005 | 3.1e-3 | 0.485 |

* **γ moved up from 1 in every layer.** It sits at 1.36–1.58 in L0–L4 and stays near 1 (1.04) in L5.
  It rose mainly between steps 2,000 and 10,000. It did **not** converge near the hand-picked 1.75:
  the maximum is 1.58 in L2, and the depth profile is humped. At 16K it is still creeping up in L1–L4
  (+0.011 to +0.018 per 2K steps) and is flat in L0.
* **β moved little from 2:** −10% in L0 (dipping to 1.69 at 8K, then recovering), +4–11% elsewhere.
  It is still rising in every layer at 16K (+0.012 to +0.026 per 2K).
* **g is negative in L0–L4 and ≈0 in L5, and flat after ~10K.** It is weakly identified, because
  downstream weights absorb a global rescale, so its trajectory is reported and **not interpreted**.

**Degeneracy of (β, γ).** Measured on the run's own trained margins, after removing the best gain
offset, which g absorbs:

* **L0 is near-degenerate.** The centred sensitivities of log s to log β and log γ correlate at
  +0.84 (condition number 21.6). A ridge of near-equivalent scores runs from (γ 1.25, β 2.12) to
  (γ 1.5, β 1.53), with residual ≤ 0.04 in log s against a log-score sd of 1.09. L0's individual
  (β, γ) values are therefore uninformative.
* **L1–L5 are better conditioned** (corr 0.02–0.19, condition number 6–7). The valley has a clear
  minimum at the learned γ.
* **β's departure from 2 barely matters for the score shape in any layer.** Holding β = 2 and
  refitting γ reproduces the learned score to 0.012–0.036 log-RMS. The refit γ is 1.30, 1.47, 1.59,
  1.49, 1.38 and 1.05 — the learned γ within 0.01, except L0. For comparison, the learned scores
  depart from margin by 0.17, 0.27, 0.34, 0.28 and 0.22 in L0–L4, and 0.03 in L5.
  * γ carries the learned shape change.
  * β's movement is small, still drifting, and nearly shape-irrelevant.
  * Jointly with γ, the sigmoid slope does not do anything detectable on the score, just as it did
    nothing useful when swept alone.
  * This is a statement about score shape on these margins, not a measured bpb effect.

### Selectivity on its own trained margins (`selectivity_trained.py`)

| run | wCV overall | wCV L0 / L1–L5 | p75/p25 | frac<1e-3 (L0) | mean |
|---|---|---|---|---|---|
| **0247 learned** | **1.200** | 2.95 / 0.87–1.25 | 8.09 | 0.02% (0.1%) | 0.479 |
| 0193 margin | 0.983 | 1.84 / 0.86–0.87 | 5.11 | 0 | 0.547 |
| 0245 sharp γ1.75 | 1.820 | 8.65 / 1.44–1.56 | 9.29 | 0.07% (0.4%) | 0.518 |
| 0246 sharp γ3 | 4.200 | 9.70 / 2.67–3.64 | 22.53 | 4.61% (20.2%) | 0.433 |
| 0243 min_margin | 2.000 | 2.92 / 1.62–1.70 | 12.33 | 0.85% (3.1%) | 0.655 |
| 0244 tanh_margin | 1.872 | 6.46 / 1.35–2.04 | 59.96 | 10.55% (40.3%) | 0.722 |

The learned form chose moderately more within-token selectivity than margin: wCV +22% and p75/p25
8.1 vs 5.1. That is well short of sharp_margin γ1.75, and nowhere near the continuous forms or γ=3.

### Continuity

It stays **discontinuous**. `continuity_probe_trained.py`, own weights, 8 val rows, 4,000 samples
per layer: n=1 boundary jump median **2.19%** of ‖y_h‖, 1.9–2.8% per layer (p10 0.57%, p90 7.0%).
For comparison: margin 3.45%, 0245 1.66%, 0246 0.43%, min_margin and tanh_margin 0. The boundary
survival values at the learned γ are in the table above.

### Reading

Stated with one seed and no LUT replicates, and not overclaimed.

* Started exactly at margin and free to move, the three shape parameters moved the score *towards
  modest extra selectivity*: γ ≈ 1.4–1.6 in L0–L4, not 1.75, with L5 staying margin-like. They did
  not move towards the sharper or continuous forms that lost.
* The learned run ends between margin and sharp_margin γ1.75 in bpb. It is not resolved from either.
* The hand-picked 1.75 was in the right direction but above what gradient descent settled on.
* β is not meaningfully learned. The one learnable constant that measurably changed the score is γ.
* Parameters were still drifting at 16K (β everywhere, γ in L1–L4), so these are not converged values.
