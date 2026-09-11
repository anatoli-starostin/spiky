# sharp_margin: a selectivity-matched DISCONTINUOUS control

Branch `research/ffn_replacement_fix`. The two continuous-at-the-boundary confidence forms both lost to
`margin` at the paper geometry (LightMHL n=1, H4 tph128 nap8, d 48/48, 16K, corrected eval):

| run | form | corrected_val_bpb | vs margin | within-token CV | p75/p25 | frac<1e-3 |
|---|---|---|---|---|---|---|
| exp_g_0193 | margin = (Σm)·Πσ(2m) | 1.172852 | — | 0.983 | 5.11 | 0 |
| exp_g_0243 | min_margin = (min m)·Πσ(2m), gain 37.4 | 1.197236 | +0.0244 | 2.000 | 12.33 | 0.85% |
| exp_g_0244 | tanh_margin = (Σm)·Πtanh(2m) | 1.186730 | +0.0139 | 1.872 | 59.96 | 10.6% |

(Selectivity on each run's own trained margins, 4 val rows, `selectivity_trained.py`.)

Both continuous forms also roughly doubled within-token selectivity, so neither arm can attribute its
deficit to continuity. The control keeps the score DISCONTINUOUS at cell boundaries (σ(0) = 0.5 > 0)
while raising selectivity to the continuous forms' level:

    sharp_margin   s = gain · (Σ_j m_j) · (Π_j σ(2 m_j))^γ ,  γ > 1   (γ = 1 is margin)

Crossing one boundary (m_j → 0) multiplies the score by σ(0)^γ / σ(2 m_j)^γ ≥ 0.5^γ, never 0; the
all-margins-zero corner value of the product factor is 0.5^(n·γ).

## Preregistered prediction (written before the calibration was finalised and before any training)

Outcome bins for exp_g_0245, with "near" meaning within 0.0035 (the 16K budget-law residual sd on
record; the vanilla two-seed range is 0.00335) of the reference, and the gap margin → tanh_margin
being 0.0139 ≈ 4 of those units:

* **A — lands near margin** (1.1694 ≤ bpb ≤ 1.1764): the discontinuous control recovers margin's
  performance despite continuous-form-level selectivity → **continuity costs performance**.
* **B — lands near tanh_margin** (1.1832 ≤ bpb ≤ 1.1902): matched selectivity alone reproduces the
  deficit → **sharp gating is simply worse**; continuity is not implicated.
* **C — intermediate** (1.1764 < bpb < 1.1832): **both contribute**; the position is reported as the
  fraction (bpb − 1.172852)/(1.186730 − 1.172852).
* Outside both references (below 1.1694 or above 1.1902) is reported as such, not forced into a bin.

My own prior, stated so it can be wrong: **C, leaning towards B** (roughly 1.178–1.186). A sharper
product suppresses both the forward read and the gradient into low-confidence tables in the same way
the continuous forms do, and tanh_margin (continuous, far more near-zero scores) lost *less* than
min_margin (continuous, few near-zeros), which is hard to square with continuity being the main cost.

What one run cannot settle: one seed per arm and no LUT-seed replicates. The 16K noise figures on
record are a vanilla two-seed range (0.00335, not an sd, not LUT) and a budget-law residual sd
(≤0.0035); the only LUT multi-seed figure is the 4K three-seed sd 0.009642, a lower bound because
26.8% of the parameters are re-drawn per seed. Differences below ~0.007 are not resolved by any arm.

## Calibration (PART 1) — result: STOPPED, no single value matches the target profile

`sharp_margin_calibration.py`, exp_g_0193's own trained margins (4 val rows, 6 layers, 6.29M margin
vectors, TPH 128), every candidate gain-matched to margin's mean score 0.5473 before the selectivity
statistics are taken. Target = tanh_margin on its own margins (exp_g_0244): within-token CV 1.872
[L0–L5 6.46 2.04 1.85 1.70 1.58 1.35], p75/p25 59.96, frac<1e-3 10.55% [40.3 8.3 6.0 4.2 3.1 1.5 %].
"boundary/before" = median score ratio after setting one random margin to 0 (margin 0.62; any
continuous form 0).

| candidate | gain | wCV | per-layer wCV | p75/p25 | frac<1e-3 | L0 frac | boundary/before | 0.5^(n·γ) |
|---|---|---|---|---|---|---|---|---|
| margin (γ=1) | 1.00 | 0.983 | 1.84 0.86–0.87 | 5.11 | 0 | 0 | 0.62 | 3.9e-3 |
| γ=1.5 | 2.54 | 1.541 | 5.98 1.23–1.34 | 8.70 | 0.08% | 0.5% | 0.51 | 2.4e-4 |
| **γ=1.75** | 3.91 | **1.901** | 10.33 1.47–1.65 | 11.33 | 0.60% | 3.6% | 0.47 | 6.1e-5 |
| γ=2 | 5.91 | 2.332 | 16.73 1.75–2.04 | 14.77 | 2.1% | 12.5% | 0.43 | 1.5e-5 |
| γ=2.25 | 8.75 | 2.850 | 24.96 2.07–2.56 | 19.29 | 4.6% | 27% | 0.39 | 3.8e-6 |
| γ=2.5 | 12.70 | 3.474 | 34.11 2.42–3.23 | 25.17 | 7.6% | 44% | 0.36 | 9.5e-7 |
| γ=2.75 | 18.11 | 4.225 | 42.96 2.82–4.12 | 32.85 | 10.7% | 59% | 0.33 | 2.4e-7 |
| **γ=3** | 25.38 | 5.128 | 50.64 3.28–5.31 | **42.83** | **13.6%** | 71% | 0.30 | 6.0e-8 |
| γ=3.5 | 47.43 | 7.490 | 61.65 4.39–8.93 | 72.87 | 19.8% | 86% | 0.25 | 3.7e-9 |
| γ=4 | 83.33 | 10.78 | 68.32 5.82–14.92 | 123.98 | 26.9% | 93% | 0.21 | 2.3e-10 |
| γ=6 / 8 | 474 / 1463 | 36.6 / 82.0 | — | 1041 / 8757 | 59% / 81% | ~99% | 0.10 / 0.05 | 3.6e-15 / 5.4e-20 |
| β=3…16 (σ(βm) form) | 0.55…0.16 | 0.90…0.49 | L0 1.63…0.70 | 5.14…2.88 | 0 | 0 | 0.55…0.45 | 3.9e-3 |
| *tanh a=2, same margins* | 1.49 | 2.228 | 14.79 1.64–2.31 | 101.77 | 16.9% | 66.7% | 0 | — |
| *min_margin, same margins* | 37.39 | 1.969 | 5.52 1.66–1.73 | 12.22 | 1.03% | 3.9% | 0 | — |

Gain (margin:form mean-score ratio) at other training stages, sweep_s05 margins (TPH 256):
γ=1.75: 6.60 at init, 3.31 after 4K (3.91 on exp_g_0193); γ=2: 11.9 / 4.76 (5.91);
γ=3: 106.6 / 17.5 (25.4); γ=3.5: 286 / 31.1 (47.4). For comparison tanh a=2: 2.57 / 1.29 (1.49);
min_margin: 38.9 / 36.3 (37.4).

Findings:

* **β form rejected.** Raising β *lowers* selectivity monotonically (CV 0.98 → 0.49 at β=16): σ(βm)
  saturates to 1 for typical margins, the product flattens and only Σm is left. It cannot reach the
  target in the β>2 range at all.
* **γ form: selectivity rises with γ and discontinuity survives everywhere** (a single boundary
  crossing keeps 47% of the score at γ=1.75 and 30% at γ=3, vs 62% for margin), but **no single γ
  reproduces tanh_margin's profile.** Within-token CV matches at γ≈1.75 (1.90 overall, L1–L5
  1.47–1.65), where p75/p25 is 11.3 (target 60) and frac<1e-3 0.6% (target 10.6%). p75/p25 and the
  near-zero fraction match at γ≈3 (42.8, 13.6%), where CV is 5.1 (2.7× target, L1–L5 3.3–5.3). On the
  same margins the γ family has 2.3–2.7× tanh's CV at equal near-zero fraction: its extra selectivity
  sits in a heavy upper tail, tanh's in a suppressed bulk.
* **The near-zero match is not a boundary artefact that a discontinuous form cannot reach**
  (checked, since that was the tempting explanation): among scores < 1e-3, the minimum margin is below
  0.05 for 89% of tanh's and 77% of γ=3's (base rate 39%), and both populations have similarly small
  Σm (median 1.94 and 1.77 vs 4.69 overall). γ=3 does reproduce the near-zero population; what it
  cannot do at the same time is keep CV at ~1.9.
* **Scale.** γ=1.75's gain drifts 6.6 → 3.3 → 3.9 over training (about tanh's 2.6 → 1.3 → 1.5).
  γ=3's drifts 107 → 17.5 → 25.4, so a fixed gain of 25.4 puts the score ~4× below margin's
  scale at init. γ ≥ 3 matches the near-zero statistics only by giving up the scale match.
* **γ≈1.75 reproduces min_margin's full profile** on the same margins (1.90 / 11.3 / 0.6% vs
  1.97 / 12.2 / 1.0%; on min_margin's own margins 2.00 / 12.33 / 0.85%).

Per the brief ("if selectivity can't be matched without destroying discontinuity or scale: stop and
report"), PART 1 stopped here: no implementation and no training until the owner picks between
γ≈1.75 (CV / min_margin-profile match, stable scale), γ≈3 (tanh p75/p25 + near-zero match, 2.7× CV,
unstable scale), or both arms.

## Decision: both arms, run sequentially (0245 first)

* **exp_g_0245 — γ = 1.75, gain 3.9: the matched discontinuous control for exp_g_0243
  (min_margin).** On exp_g_0193's trained margins γ=1.75 reproduces min_margin's *entire* selectivity
  profile (wCV 1.90 vs 1.97, p75/p25 11.3 vs 12.2, frac<1e-3 0.6% vs 1.0%), so this pair differs in
  essentially nothing but continuity. **This is the primary test.**
* **exp_g_0246 — γ = 3.0, gain 25.4: the control for exp_g_0244 (tanh_margin) on the p75/p25 and
  near-zero axes** (42.8 vs 60, 13.6% vs 10.6%), accepting within-token CV 5.1 vs 1.87.

**Neither arm matches tanh_margin on all selectivity axes at once.** That is why both are run: with
one arm, whichever statistic went unmatched would stay available as an escape hatch when
interpreting the result. Together they bracket the target.

Per-run γ: a config key `lut_sharp_margin_gamma`. model_build requires it for sharp_margin and
passes it to `LightMultiHeadLUT(sharp_margin_gamma=…)`; each layer's γ and gain are printed to
train.log. `SHARP_MARGIN_GAMMA` = 1.75 is only the module default. train.py stays byte-identical to
exp_g_0193's.

Preregistered reading, per arm, with the ±0.0035 bins above:
* **Near exp_g_0193 (margin, 1.1729):** selectivity is not the cause; continuity costs performance.
* **Near its continuous partner** (0245 vs exp_g_0243 1.1972; 0246 vs exp_g_0244 1.1867): continuity
  is not the cause; sharp gating is simply worse.
* **Intermediate:** both contribute.

The 0245/0243 pair is weighted most, and any disagreement between the arms is reported as such.
For 0245 the partner bin is 1.1937–1.2007 and the margin bin 1.1694–1.1764. The gap to its partner
is 0.0244 (≈ 7 noise units), so the intermediate range is wide.

## Results

Both runs: code 9c26b451, train.py byte-identical to exp_g_0193's, with LD_LIBRARY_PATH set.

### Discontinuity, measured before the bpb was read

`continuity_probe.py` at H4/tph128/nap8/d48, random-init tables, 300 draws: n=1 jump as a fraction
of ‖y_h‖.

`continuity_probe_trained.py` on each run's own checkpoint: 8 val rows, 4,000 samples per layer,
boundary score from that run's own form, gain and γ.

| run | jump at init: median (p10–p90) | jump trained: median (p10–p90) | 0.5^(8γ) | calibration bnd |
|---|---|---|---|---|
| exp_g_0193 margin | 4.11% | 3.45% (1.1–8.6%) | 3.9e-3 | 0.62 |
| exp_g_0245 γ=1.75 | 1.68% (0.36–5.9%) | 1.66% (0.35–6.4%) | 6.1e-5 | 0.47 |
| exp_g_0246 γ=3 | 0.299% (0.028–2.05%) | 0.43% (0.04–3.3%) | 6.0e-8 | 0.30 |
| exp_g_0243 min_margin | 1.5e-9 (counterfactual) | 0 | — | 0 |
| exp_g_0244 tanh_margin | 1.03e-9 | 0 | — | 0 |

Both arms are genuinely discontinuous, at initialisation and after training: 6.6–7.4 decades above
the ~6e-10 same-side control. γ=3's typical jump is 8–14× smaller than margin's, though its p90
(3.3%) is of margin's median order. It is not a collapsed control, but a weaker discontinuity.

### bpb

In-run corrected eval, bs48×100 skip-12. Deltas are in units of the three noise figures on record:
the vanilla two-seed range 0.00335, the budget-law residual sd 0.0035, and the 4K LUT three-seed sd
0.009642 (a lower bound).

| run | bpb | h | vs 0193 (1.172852) | vs 0243 (1.197236) | vs 0244 (1.186730) | vs 0195 (1.160637) |
|---|---|---|---|---|---|---|
| 0245 γ=1.75 | **1.167381** | 0.913 | −0.0055 (−1.6 / −1.6 / −0.57) | −0.0299 (−8.9 / −8.5 / −3.1) | −0.0193 (−5.8 / −5.5 / −2.0) | +0.0067 (+2.0 / +1.9 / +0.70) |
| 0246 γ=3 | **1.182109** | 0.911 | +0.0093 (+2.8 / +2.6 / +0.96) | −0.0151 (−4.5 / −4.3 / −1.6) | −0.0046 (−1.4 / −1.3 / −0.48) | +0.0215 (+6.4 / +6.1 / +2.2) |

**Preregistered bins:**
* **0245:** OUTSIDE both references, below the margin bin. Position (bpb − margin)/(partner −
  margin) = −0.22. Better than 0243 at 32/32 eval steps. Worse than 0193 at the first 11/32 steps;
  its matched-step delta crosses zero near step 5,500–6,000 and ends at −0.0055.
* **0246:** INTERMEDIATE, position 0.667, and 0.0011 short of the partner bin. Worse than 0193 and
  better than 0244 at 32/32 steps.

### Achieved selectivity on each run's own trained margins (`selectivity_trained.py`)

| run | wCV overall | wCV L0 / L1–L5 | p75/p25 | frac<1e-3 (L0) | mean |
|---|---|---|---|---|---|
| 0245 γ=1.75 | 1.820 | 8.65 / 1.44–1.56 | 9.29 | 0.07% (0.4%) | 0.518 |
| *0243 min_margin (target of 0245)* | 2.000 | 2.92 / 1.62–1.70 | 12.33 | 0.85% (3.1%) | 0.655 |
| 0246 γ=3 | 4.200 | 9.70 / 2.67–3.64 | 22.53 | 4.61% (20.2%) | 0.433 |
| *0244 tanh_margin (target of 0246)* | 1.872 | 6.46 / 1.35–2.04 | 59.96 | 10.55% (40.3%) | 0.722 |
| *0193 margin* | 0.983 | 1.84 / 0.86–0.87 | 5.11 | 0 | 0.547 |

**0245** held min_margin's level on overall CV and roughly on p75/p25. It has 12× fewer near-zero
scores and 3× the L0 CV. **0246** drifted from its calibration (5.13 / 42.8 / 13.6% on 0193's
margins): it ended *further* from tanh_margin on exactly the axes it was meant to match (p75/p25
0.38×, near-zero fraction 0.44×) while overshooting CV 2.2×. It also started ~4× below margin's
score scale; its gain ratio was 107 at init.

### Reading

Stated with one seed per arm and no LUT-seed replicates at this geometry.

* **Primary pair (0245 vs 0243).** A discontinuous form at roughly min_margin's selectivity lands
  *at or slightly below margin* (−0.0055, not resolved against the LUT figure) and 0.030 below
  min_margin (3.1× even the conservative LUT figure). **min_margin's deficit is not explained by
  its within-token selectivity.** Of the preregistered readings, the result sides with "continuity
  costs performance". What 0243 and 0245 still differ in besides continuity: the min operator (a
  single anchor carries the score and its gradient; non-smooth argmin switches), the gain (37.4 vs
  3.9), and the residual profile gaps (near-zero fraction, L0 CV). The deficit belongs to
  continuity *or* to min_margin's construction; this pair does not separate the two.
* **Secondary arm (0246 vs 0244).** A discontinuous but much sharper gate (CV 4.2, 4.6% near-zero)
  loses to margin by 0.0093 and recovers only a third of tanh_margin's deficit. **Sharp gating at
  this level costs performance even without continuity.** The remaining 0.0046 is below every noise
  figure's resolution, so it cannot be attributed to tanh_margin's continuity. The match is also
  loose (own-margin profile off by 2–2.6× on every axis, 8× smaller jump, early scale deficit), so
  this arm is weaker evidence.
* **The arms do not tell one story.** 0245 shows that CV ≈ 1.8 (equal to tanh_margin's own 1.87)
  costs nothing by itself. 0246 shows that a stronger gate costs even when discontinuous. This fits
  a selectivity cost that is ~0 at min_margin's level and substantial at γ=3's, with min_margin's
  deficit coming from something other than selectivity. tanh_margin's deficit is unresolved: it
  could come from its bulk suppression (p75/p25 60, 10.6% near-zero, matched by neither arm), from
  its continuity, or from both.
* **Noise.** The ±0.0035 bins are built on vanilla / budget-law figures. Against the 4K LUT
  lower bound (0.0096), only 0245 vs 0243 (3.1×), 0246 vs 0195 (2.2×) and 0245 vs 0244 (2.0×) reach
  2 units. 0245 vs margin, and 0246 vs either reference, do not. Seed replicates of 0245 and 0243 are
  the cheapest way to firm up the primary conclusion.

Wall clock: 0.913 h and 0.911 h, matching the 0.88 h microbenchmark prediction. The sharp_margin
form has no torch.prod. See `tanh_prod_slowness/README.md` for why exp_g_0244 took 1.721 h.
