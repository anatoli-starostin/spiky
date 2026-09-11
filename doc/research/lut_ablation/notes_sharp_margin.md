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
