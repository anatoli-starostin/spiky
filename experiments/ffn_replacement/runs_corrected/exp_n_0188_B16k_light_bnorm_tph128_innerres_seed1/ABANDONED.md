# ABANDONED at step 9,000 / 16,000 — superseded by exp_g_0189

Stopped deliberately, not a crash. **The `metrics.csv` in this folder is a partial curve and
must not be read as a result.**

| | |
|---|---|
| last eval | step 9,000, val bpb **1.256461** |
| steps completed | 9,000 of 16,000 (56%) |
| final score | **none** — no `corrected_score.json`, no `checkpoint.pt` |

## Where it stood when stopped

Against its direct control exp_n_0185 (1.206222 final), at shared eval steps:

| step | 0188 (inner residual) | 0185 (control) | delta |
|---|---|---|---|
| 1000 | 1.775467 | 1.773901 | +0.001566 |
| 2000 | 1.545935 | 1.540202 | +0.005733 |
| 3000 | 1.419701 | 1.422092 | −0.002391 |
| 4000 | 1.362154 | 1.354515 | +0.007639 |

Sign sequence `+ + − +` — oscillating, no trend, and every gap of the same order as the 0.00335
16k seed spread. **This run produced no evidence either way about `inner_residual`**; it was
stopped because a more promising change came up, not because it was losing.

## Why it was stopped

Investigating whether weight decay could explain the layer-0 LayerNorm collapse turned up a
different and larger problem: `setup_optimizer` exempted only `FastMultiHeadLut` parameters
from weight decay (`isinstance(m, FastMultiHeadLut)`), so **`LightMultiHeadLUT`'s 3-D `tables`
fell into the decay group while Fast's identical tables were exempt** — Light trained 75.5M
(or 37.7M at tph=128) table parameters under `weight_decay=0.1` where Fast trained them under
`0.0`. That is an unintended confound in every Light-vs-Fast comparison, and correcting it
takes priority over the inner-residual question, which is why the GPU was reassigned.

`inner_residual` is not refuted and remains worth testing later — on top of the corrected
decay behaviour, so the two changes are not confounded with each other.

## Kept, not deleted

The folder stays so the partial curve remains inspectable and the abandonment is part of the
record. Per `claude/experiment-methodology.md`, a run is never silently removed.

Successor: **`exp_g_0189_B16k_light_bnorm_tph128_nodecay_seed1`** — exp_n_0185 with
`lut_tables_no_decay: true` as the single functional change.
