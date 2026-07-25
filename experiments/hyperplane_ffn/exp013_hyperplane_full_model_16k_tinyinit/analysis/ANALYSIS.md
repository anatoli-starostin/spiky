# exp013 (near-zero init, std=1e-4) coefficient-growth post-mortem

exp013 stalled at best val_bpb **1.2756** (~0.08 worse than exp010 anchor 1.1940 and
exp012 random-0.05 1.1953; also below exp752 fixed-anchor 1.2162 and exp001 dense-MLP
1.2014). This analyzes the learned hyperplane weights (final checkpoint, CPU-only) to
explain why. Init: Gaussian rows std 1e-4 → init row-norm ‖w‖≈1e-4·√384≈0.00196, b=0.
soft_score_temp T_soft=0.5. Pooled over 6 layers per site.

## 1. GROWTH / wake-up — they woke up, but UNDER-grew
| site | ‖w‖ (mean) | growth ×init | ≥10× init | ≥100× init | ≥0.05 (exp012 scale) | near floor |
|------|-----------|--------------|-----------|------------|----------------------|-----------|
| qk_lut       | 0.426 | ×217 | 0.96 | 0.63 | 0.026 | 0.01 |
| v_lut        | 0.399 | ×204 | 0.96 | 0.61 | 0.020 | 0.01 |
| out_proj     | 0.354 | ×181 | 0.95 | 0.53 | 0.016 | 0.02 |
| residual_lut | 0.390 | ×199 | 0.95 | 0.58 | 0.020 | 0.01 |
| emb_resid    | 0.592 | ×302 | 0.97 | 0.74 | 0.098 | 0.01 |

- **Almost everything woke up**: only ~1–2% of coefficients are still near the init
  noise floor; ~95–97% grew past 10× init. So it is NOT "stuck-tiny."
- **But it under-grew**: only ~50–74% passed 100× init, and just ~2–10% reached the
  ~0.05 magnitude that the healthy exp012 run sits at. Final ‖w‖≈0.35–0.59 — roughly
  a THIRD of exp012's ~1.08 and a quarter of exp010's ~1.48.

## 2. SPARSITY — DENSE, not sparse (like exp012, unlike exp010)
Participation ratio ≈ 111–132 (dense; a dense Gaussian row in R^384 has PR≈128), only
~5% of ‖w‖² in the top-2 coords. Within a row the top-10% of coords hold ~44% of the
mass → moderately concentrated but broadly spread; growth was ~UNIFORM-dense, NOT a few
coefficients dominating (no rich-get-richer collapse to sparsity). So near-zero init did
NOT discover the sparse coordinate-pair geometry exp010 uses — it landed dense.

## 3. DEGENERATE / weak margins — the core of the stall
Estimated decision margin scale ⟨w,x⟩ ≈ ‖w‖ (x is O(1)/coord after MeanAbsNorm):
margin/T_soft ≈ **0.71–1.18** (mostly ≤ 1). The smooth surrogate is p = a/(T_soft+|a|);
with |a| ≈ T_soft the bits are only ~half-decisive (|p|≈0.5). Contrast exp012 (‖w‖≈1.08 →
margin/T_soft ≈ 2.2 → firm bits). exp013's hyperplanes are still producing SOFT, weakly
decisive sign tests — consistent with the stalled loss.

## 4. Rank / redundancy — bits became correlated
Within-table mean|cos| ≈ 0.04–0.21 (qk 0.11, v 0.15, out 0.21) — ABOVE the 0.041 random
baseline, i.e. the K hyperplanes of a table are MORE aligned than random, and effective
rank/K dropped to 0.79–0.99 (out_proj 0.79). The bits are becoming redundant/low-rank, so
a table's K sign-tests carry less independent information. (exp010 was below random /
full rank; exp012 was at random / full rank.)

## 5. BIASES
|b| ≈ 0.011–0.025, std small — moved off 0 but stayed tiny, same order as exp010/exp012.
Not the issue.

## Three-way comparison (site-averaged)
| run | val_bpb | PR | top-2 mass | mean\|cos\| | ‖w‖ | erank/K | geometry |
|-----|---------|----|-----------|------------|------|---------|----------|
| exp010 anchor 2e-? | **1.1940** | 2.3 | 0.92 | 0.021 (below rand) | 1.48 | 0.99 | SPARSE, extra-orthogonal |
| exp012 random 0.05 | **1.1953** | 129 | 0.048 | 0.039 (= rand) | 1.08 | 0.99 | DENSE, random-orthogonal |
| exp013 tiny 1e-4   | **1.2756** | 122 | 0.053 | 0.116 (above rand) | 0.43 | 0.93 | DENSE but UNDER-grown, correlated, rank-collapsing |

## Headline
Near-zero init did NOT leave the hyperplanes stuck at zero (they grew ~200×) and did NOT
discover sparsity (it's dense like exp012). It failed a different way: the hyperplanes
**under-grew into a weak-margin (⟨w,x⟩≈T_soft), correlated, rank-collapsing** regime — soft,
redundant bits that can't drive decisive LUT row-selection, which pins the loss at 1.2756.
So the degenerate-margin hypothesis is CONFIRMED in spirit (margins are weak) but the
mechanism is "woke-up-but-under-grew + bit redundancy," not literal stuck-tiny. There is a
real init-scale floor: 0.05 boots into a firm-margin working regime (~1.194); 1e-4 stalls
in a soft-margin one (~1.276).
