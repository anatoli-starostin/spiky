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

---

# Frozen-gain fork — exp_g_0248

**Question.** Was 0247's learnable per-layer log-gain g doing real work, decoupling scale from γ's
shape, or bookkeeping? The fork holds g at exactly 0 (a buffer) while β and γ stay learnable. If γ
lands on the same profile, g is dropped.

## Pre-flight (code trace, before any GPU time)

The n=1 read-out has **no threshold on the confidence score**:

* **Address bits** come from margins, never from the score: `d.detach() > 0` in torch, and
  `delta > cmp_eps` with cmp_eps = 0.0 in both native kernels.
* **The score** is consumed only as `F.embedding_bag` `per_sample_weights` (`_bagged_sum`).
* **The cell** passes through unchanged: `_apply_cell` returns the bag as-is (constant cells; no
  margin read-out, codebook, blend or gate).
* **CompressionMHL** (light, multi-head): compress → lut_light → decompress Linear.
* **Block:** `x + ffn(ln2(x))`, with no gate and `lin` None (config gamma 0).
* **The scored CUDA kernel** only multiplies by the gain, and refuses form id 6.

A global rescale of the score is therefore exactly absorbable by decompress. That makes it
representationally free, but not free under the optimiser: decompress is weight-decayed at 0.1,
the tables and the learned scalars are not, and Adam and gradient clipping both see the scale.

## Downstream-compensation check on 0247

`downstream_compensation_0247.py`. Note: this check had not been run before; it was run for this
task.

At 4K, 8K, 12K and 16K, 0247 vs 0193:

* Decompress weights grew where g went negative, with corr(−g, log decompress ratio) = +0.86 over
  24 layer×checkpoint points.
* FFN output RMS stayed within ±4% of 0193's in every layer, while exp(g) fell to 0.77.
* The weights did not simply invert exp(g). At 16K, decompress × tables ratios were
  1.17/1.09/1.12/1.11/1.09, against exp(−g) = 1.29/1.29/1.20/1.16/1.10.

**Correction from 0248** (`scale_compare_0248.py`). 0248's decompress norms equal 0247's (ratio
0.987–0.999). The decompress growth of 0247 over 0193 therefore came from the (β, γ) change, not
from g. The correlation with −g above is not evidence that decompress compensated g.

## Implementation and verification

Code 56929153.

* `LightMultiHeadLUT(learned_margin_freeze_g=True)` registers `confidence_g` as a buffer: same
  state_dict key, same score op sequence, no gradient, in no optimiser group. The config key is
  `lut_learned_margin_freeze_g`.
* **Tests:** 329 passed on CUDA.
* **Regression:** before (unchanged source) vs after, 127/127 torch.equal, covering bounded,
  bounded_norm, margin, min_margin, tanh_margin, sharp_margin and learned_margin (at moved
  parameters, with its parameter gradients).
* **Init:** on 0193's trained checkpoint, 0248 == 0193 == 0247 torch.equal on every score, every
  block output and the loss (max discrepancy 0).
* **Fresh init:** bit-identical to 0193 on all shared tensors; 67,351,692 params (+12).
* **Optimiser:** the 12 β/γ scalars are in weight_decay 0; g is in no group.
* **Training steps:** after 5 real AdamW steps g is still exactly 0.0, β and γ gradients are
  nonzero in every layer, and there is no prod/cumprod.
* **Fork:** config differs from 0247's only in `lut_learned_margin_freeze_g: true`; train.py is
  byte-identical to 0247's.

## Results (one seed; no LUT-seed replicates at this geometry)

**corrected_val_bpb 1.169328**, 0.946 h. Commit c22458a9.

| vs | bpb | Δ | vanilla 2-seed range 0.00335 | 4K LUT sd 0.009642 (lower bound) |
|---|---|---|---|---|
| exp_g_0247 (g learnable) | 1.169344 | −0.000016 | −0.005 | −0.002 |
| exp_g_0193 margin | 1.172852 | −0.003524 | −1.05 | −0.37 |
| exp_g_0245 sharp γ1.75 | 1.167381 | +0.001948 | +0.58 | +0.20 |
| exp_g_0195 n=2 | 1.160637 | +0.008691 | +2.59 | +0.90 |

The 0248 and 0247 curves coincide: |Δ| ≤ 0.0022 at every eval, ≤ 0.0007 after step 2,000, and 0248
is worse at 19 of 32 steps.

**γ (the deciding measurement)** is unchanged:

| layer | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| γ 0248 | 1.372 | 1.482 | 1.595 | 1.491 | 1.378 | 1.042 |
| γ 0247 | 1.356 | 1.468 | 1.580 | 1.482 | 1.383 | 1.044 |
| Δ | +0.016 | +0.014 | +0.015 | +0.008 | −0.005 | −0.001 |
| drift 14K→16K, 0248 | −0.000 | +0.015 | +0.018 | +0.013 | +0.011 | +0.005 |

* The trajectories correlate at +1.000 with 0247's in every layer.
* γ crosses 1.1 at the same eval step as in 0247.
* The late drift is identical to 0247's (+0.001/+0.015/+0.018/+0.014/+0.011/+0.005), so γ is still
  not converged in L1–L4.
* The humped depth profile is reproduced.

**β** follows 0247's trajectory shape (corr 0.91–1.00), 0.02–0.06 lower in L0–L4: 1.755 2.046 2.173
2.101 2.056 2.120. It is still rising at 16K (+0.012 to +0.024 per 2K). It remains shape-irrelevant:
holding β = 2 and refitting γ reproduces each layer's learned score to 0.008–0.029 log-RMS. L0 is
again near-degenerate in (β, γ), corr +0.87 and condition number 25.

**Where the removed gain went** (`scale_decomposition_0248.py`). Exact per-layer decomposition of
log E[s] (0248) − log E[s] (0247, gain included), on 4 real val rows; the terms sum with residual 0:

| layer | total | = −g47 | + β | + γ | + margins | median \|m\| 47 → 48 |
|---|---|---|---|---|---|---|
| L0 | +0.006 | +0.256 | −0.056 | −0.045 | −0.148 | 0.332 → 0.318 |
| L1 | +0.022 | +0.252 | −0.061 | −0.025 | −0.144 | 0.621 → 0.595 |
| L2 | +0.004 | +0.179 | −0.044 | −0.023 | −0.109 | 0.662 → 0.641 |
| L3 | +0.041 | +0.148 | −0.032 | −0.013 | −0.061 | 0.678 → 0.665 |
| L4 | +0.009 | +0.094 | −0.017 | +0.007 | −0.076 | 0.719 → 0.700 |
| L5 | −0.001 | −0.005 | +0.004 | +0.002 | −0.001 | 0.690 → 0.690 |

0248 reached 0247's score scale without g, with the removed gain re-absorbed as follows:

* **Mostly upstream,** by 2–4% smaller margins: 41–81% of −g in L0–L4.
* **Then β:** 18–24% in L0–L3 — its lower values carry scale, not shape.
* **γ least:** 18%, 10%, 13% and 9% in L0–L3, and −8% in L4.

The **predicted failure mode — γ's gradient carrying the scale, pulling γ toward 1 — did not
occur.** γ took the smallest share, and in the direction of *higher* γ, not toward 1.

Downstream, decompress ratio 0248/0247 is 0.987–0.999 and FFN output RMS ratio 0.95–1.04.

**Selectivity** on its own margins is the same as 0247's: wCV 1.206 (L0 3.00, L1–L5 0.87–1.28),
p75/p25 8.09, frac<1e-3 0.01% (L0 0.08%), mean 0.485. 0247 had 1.200 / 8.09 / 0.02% / 0.479;
margin had 0.983 / 5.11 / 0 / 0.547.

**Continuity: still discontinuous.** Trained-weights boundary jump median 2.22% of ‖y_h‖ (per layer
1.87–2.88%; 0247 2.19%, margin 3.45%). 0.5^(8γ) by layer: 5.0e-4, 2.7e-4, 1.4e-4, 2.6e-4, 4.8e-4,
3.1e-3.

## Verdict

**Drop g; report two parameters (β, γ) per layer.** Without g:

* the γ profile is the same (max |Δγ| 0.016, trajectories identical);
* bpb is identical (Δ −0.000016);
* selectivity and continuity are identical;
* the scale g carried was re-absorbed mostly upstream, not by γ.

Two caveats:

* **β is weakly informative.** It moves little, is shape-irrelevant (a β=2 refit matches within
  0.03 log-RMS), and absorbed ~20% of the removed scale. Its individual value should not be
  interpreted. A one-parameter γ form with β fixed at 2 is suggested by the score-shape analysis,
  but has not been run.
* **The evidence is limited.** One seed, no replicates. The equality of 0248 and 0247 in bpb shows
  that g bought nothing measurable here; it cannot rule out a small effect below the noise floor.

# exp_g_0249 — 0248 + Hamming-1 cell TV, weight 10

Fork of exp_g_0248 (learned_margin, g frozen) with `lut_cell_smoothness: 10.0`:
`(10 · model.lut_tv_penalty()).backward()` once per optimiser step, after the micro-batch backwards
and before clipping. The tables still get no weight decay. Code and config: 991ca119. Analysis:
`cmp_sharp.py`, `compare_tv_0248_0249.py`, `continuity_probe_trained.py`, `selectivity_trained.py`,
`analyze_learned_margin.py`, `norms_0248_0249.py`.

## bpb

**1.163698** (0.992 h; final = best).

| vs | bpb | Δ | × vanilla 2-seed range | × 4K LUT 3-seed sd |
|---|---|---|---|---|
| 0248 frozen-g | 1.169328 | −0.005630 | −1.68 | −0.58 |
| 0247 learned | 1.169344 | −0.005646 | −1.69 | −0.59 |
| 0193 margin | 1.172852 | −0.009154 | −2.73 | −0.95 |
| 0195 n=2 blend | 1.160637 | +0.003061 | +0.91 | +0.32 |

The gain arrives late. Matched-step Δ vs 0248: +0.0009 at 8K (slightly worse), −0.0010 at 10K,
−0.0035 at 12K, −0.0051 at 14K, −0.0056 at 15K and 16K. The gap stopped widening in the last 1K.
Both runs are still improving at 16K at a similar rate (last 2K: −0.00474 for 0249, −0.00423 for
0248). One seed: the Δ is 1.7× the vanilla seed range but 0.6× the conservative LUT sd.

## TV penalty and cell differences

`lut_tv` (mean over layers) in 0249:

* 7.6e-5 at step 500 (the tables start at zero),
* a peak of 6.3e-3 around 6–8K,
* 4.84e-3 at 16K, 24% below the peak.

0248, which had no TV term, measured on its checkpoints: 2.37e-2 at 4K, 4.72e-2 at 8K, 5.64e-2 at
12K, 5.84e-2 at 16K. 0249 ends 12× lower. Per layer, the trained Hamming-1 mean ‖v_c − v_c'‖² is
0.033× 0248's in L0 and 0.070–0.101× in L1–L5.

**Most of that came from shrinking the tables, not from making neighbours agree.** The cell norm
fell almost as much: table L2 norm is 0.22× (L0) to 0.35× (L5) of 0248's. The scale-free ratio
mean ‖v_c − v_c'‖² / mean ‖v_c‖², which is 2.0 for independent random cells, moved only from
1.74–1.95 to 1.34–1.57. The shrink was not undone downstream: decompress weight norm is ×0.99–1.08
and compress ×1.03–1.13, while the confidence score's mean rose from 0.485 to 0.892.

## Boundary jump — fell about 20%, still discontinuous

Median jump / ‖y_h‖ on own weights (`continuity_probe_trained.py`, 4,096 val tokens):

| layer | L0 | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|---|
| 0248 | 2.88% | 1.95% | 1.87% | 2.09% | 2.16% | 2.44% | 2.22% |
| **0249** | **1.31%** | **1.59%** | **1.71%** | **1.79%** | **1.97%** | **2.20%** | **1.78%** |

L0 more than halves; L1–L5 fall 9–18%. Relative to the whole FFN output the median is 0.485% vs
0.577%. Tokens sit about as close to boundaries as before (|u_j| < 1e-2: 0.89% vs 0.98%). The
overall 1.78% is between margin (3.45%) and sharp_margin γ1.75 (1.66%).

## β / γ

**γ is lower in every layer**, by 0.11–0.17: 1.259, 1.352, 1.478, 1.322, 1.242, 0.909. L5 is now
below 1. γ has converged: 14K→16K drift is −0.015 to +0.009 (0248: +0.000 to +0.018).

**β is higher**, by 0.08–0.16: 1.847, 2.177, 2.336, 2.180, 2.182, 2.226. It is still rising
(+0.020 to +0.038 per 2K, as in 0248). β stays shape-irrelevant: holding β = 2 and refitting γ
reproduces each layer's score to 0.017–0.054 log-RMS. L0 is again near-degenerate in (β, γ),
corr +0.81 and condition number 20.

## Selectivity — back toward margin

| run | wCV overall | wCV L0 / L1–L5 | p75/p25 | frac<1e-3 (L0) | mean |
|---|---|---|---|---|---|
| **0249 TV10** | **1.044** | 2.52 / 0.75–1.13 | 7.05 | 0.00% (0.02%) | 0.892 |
| 0248 frozen-g | 1.206 | 3.00 / 0.87–1.28 | 8.09 | 0.01% (0.08%) | 0.485 |
| 0193 margin | 0.983 | 1.84 / 0.86–0.87 | 5.11 | 0 | 0.547 |

## Reading

bpb and continuity both moved the way TV was meant to move them: −0.0056 bpb and a 20% smaller
boundary jump. Selectivity and γ relaxed toward margin at the same time. Two cautions:

* **The TV number overstates the smoothing.** It was reached mostly by shrinking the tables, so the
  cells are only modestly more alike relative to their own scale.
* **The bpb gain is not established.** It is one seed and inside the conservative noise band.
