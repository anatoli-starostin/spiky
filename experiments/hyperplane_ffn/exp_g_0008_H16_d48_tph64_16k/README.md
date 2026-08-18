# exp_g_0008 — H16 / d48 / tph64 — double the FFN slot via WIDTH

Built and smoke-tested only. **Not yet launched.**

## What this is

The width-doubled counterpart to `exp_n_0039` (which doubles via `tph`). Both start
from `exp_g_0006` (H8 / d48 / tph64, itself the width clone of `exp_n_0033`) and
double the LUT table tensor, by two different routes:

| | H | d_in=d_out | tph | route to 2× tables |
|---|--:|--:|--:|---|
| `exp_g_0006` (base) | 8 | 48 | 64 | — |
| `exp_n_0039` | 8 | 48 | **128** | more tables per head |
| **`exp_g_0008`** (this) | **16** | 48 | 64 | more heads |

`train.py` is **byte-identical** to `exp_n_0033`'s (`cmp` clean) — this is a
config-only experiment. Against `exp_n_0033` the config differs in exactly three
fields: `exp_name`, `lut_inner_in_dim` 24→48, `lut_inner_out_dim` 24→48. Against
`exp_g_0006` it differs in exactly two: `exp_name`, `lut_n_heads` 8→16.

## Smoke test — param count and breakdown

`SMOKE=1 python train.py` → **`Params: 38,552,256`**

| component | exp_g_0008 (H16/tph64) | exp_n_0039 (H8/tph128) | Δ |
|---|--:|--:|--:|
| LUT tables | 18,874,368 | 18,874,368 | **0** |
| tok_emb (tied to head) | 12,582,912 | 12,582,912 | 0 |
| attention (qkv+proj) | 3,538,944 | 3,538,944 | 0 |
| compress.weight | 1,769,472 | 884,736 | +884,736 |
| compress.bias | 4,608 | 2,304 | +2,304 |
| decompress.weight | 1,769,472 | 884,736 | +884,736 |
| decompress.bias | 2,304 | 2,304 | 0 |
| block LayerNorms | 9,216 | 9,216 | 0 |
| ln_f | 768 | 768 | 0 |
| LUT temps (soft_score) | 96 | 48 | +48 |
| LUT temps (select) | 96 | 48 | +48 |
| **TOTAL** | **38,552,256** | **36,780,384** | **+1,771,872** |

Structural checks pass: 6 CompressionMHL modules, 96 `FastMultiHeadLut` modules
(= depth × H, INDEPENDENT mode).

## The two routes are NOT param-matched — read this before comparing

Doubling the table tensor via `tph` touches **only the tables**. Doubling it via
`H` touches **the tables and the projections**, because `compress` is
`Linear(384 → H·d)` and `decompress` is `Linear(H·d → 384)`:

* tables: 9,437,184 → 18,874,368 — exactly 2× on **both** routes.
* projections: 1,774,080 → 3,545,856 on the `H` route (2× less the 2,304-param
  `decompress.bias`, which is pinned to the 384-d output and does not scale);
  **unchanged** at 1,774,080 on the `tph` route.

So `exp_g_0008` carries **+1,771,872 params (+4.82%)** over `exp_n_0039`. An
`exp_g_0008` vs `exp_n_0039` comparison is *route-of-doubling at roughly equal
table budget*, not a like-for-like parameter match. If a strict param match is
wanted, `exp_n_0039` is the arm that would need padding, not this one.

## Expected wall time (a second confound)

Wall time on this line tracks the **number of per-head `FastMultiHeadLut`
invocations per layer**, not FLOPs (established on exp_n_0033 / exp_g_0006 /
exp_g_0007). `exp_g_0008` makes 16 per layer; `exp_g_0006` and `exp_n_0039` make 8.
Expect `exp_g_0008` ≈ 1.28 h and `exp_n_0039` ≈ 0.93 h at 16k steps on the 5090 —
i.e. the width route costs ~38% more wall time for the same table budget.

## Status: RUN STOPPED EARLY BY DECISION at step 5,600 / 16,000

Not a failure and not a crash — the run was halted deliberately because it had already
answered its question, and the GPU was worth more elsewhere. `metrics.csv` holds the 28
completed evals (steps 200–5,600); there is no `summary.json` and no `loss.png`, because
`train.py` writes those only on normal completion. **Do not compare this partial curve's
endpoint to any finished run's endpoint** — 1.305 at step 5,600 is a mid-descent value,
not a result.

### Why it was stopped

`exp_g_0008` (width route: H16/d48/tph64) tracked **exactly on top of** `exp_n_0004`
(tables route: H8/d48/tph128) at equal LUT table budget — 18,874,368 tables either way.
Matched-step deltas, every 400 steps:

```
  step      0008      0033     d0033      0004     d0004
   400  2.156616  2.180203 -0.023587  2.164642 -0.008026
   800  1.867482  1.884572 -0.017090  1.874255 -0.006773
  1200  1.728855  1.744175 -0.015320  1.731982 -0.003127
  1600  1.631663  1.644111 -0.012448  1.634255 -0.002592
  2000  1.548743  1.567088 -0.018345  1.553280 -0.004537
  2400  1.474846  1.494078 -0.019232  1.475826 -0.000980
  2800  1.428004  1.441477 -0.013473  1.427340 +0.000664
  3200  1.394296  1.406083 -0.011787  1.394575 -0.000279
  3600  1.372046  1.385222 -0.013176  1.372351 -0.000305
  4000  1.354348  1.365938 -0.011590  1.354194 +0.000154
  4400  1.338652  1.348817 -0.010165  1.338849 -0.000197
  4800  1.326550  1.337571 -0.011021  1.326035 +0.000515
  5200  1.314381  1.326689 -0.012308  1.315032 -0.000651
  5600  1.305459  1.317376 -0.011917  1.306566 -0.001107
```

Read the two delta columns together. Against `exp_n_0033` the gap is **stable and large**,
sitting near **−0.012** with no sign of closing. Against `exp_n_0004` the initial −0.008
converged by ~step 2,400 and thereafter **oscillates around zero and repeatedly changes
sign** (−0.00098, +0.00066, −0.00028, −0.00031, +0.00015, +0.00052, −0.00065, −0.00111).
A delta that flips sign at the ±0.0006 level is noise, not a trend.

### Conclusion

**At equal table budget, the two ways of spending a 2× FFN slot land on the same loss
curve.** Whether the doubling goes into more tables per head or more heads, the result is
the same — *the benefit is the table budget, not its arrangement.*

That makes width the worse buy. It costs **+1,771,872 params** (the doubled
compress/decompress projections, which the `tph` route leaves untouched) and **16 vs 8
per-head LUT invocations per layer**, for no gain in loss. **Don't pay for width.**

### Caveats — this is suggestive, not decisive

- **Not a clean A/B.** `exp_g_0008` carries +4.82% params over `exp_n_0004` *and* has
  `lut_learnable_temps: true` while `exp_n_0004` does not. Two arms differing in three ways
  landing on one curve is evidence, not proof. `exp_n_0039` (= `exp_n_0004` + learnable
  temps) is the run that isolates the temps.
- **Single seed each.** The ±0.0006 wobble is about the size of the `exp_g_0006`-vs-`0033`
  edge that was itself judged too small to call real.
- **Stopped at 35% of budget.** Curves can separate late: `exp_g_0007` converged to +0.0063
  by step 8,000 and then re-opened to +0.008. This one was never given the chance to do
  that, so "equivalent" is established only over steps 2,400–5,600.

### Wall-time forecast: wrong, and by how much

This README predicted ≈1.28 h from the invocation-count rule. Actual pace was ~5,600 steps
in ~33 min, extrapolating to **≈1.6 h** — the rule **under-predicts** here by ~25%. The
16-invocation count evidently is not the whole story once the projections double as well.
Treat the invocation-count heuristic as a within-host ordering hint, not a quantitative
predictor. (Wall times are in any case **not comparable between hosts** — `exp_n_0004`'s
1.323 h was measured on the nebius H100, not this 5090.)

Tracking issue: **#108**.
