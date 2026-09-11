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
