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

## Status

Built, config-diffed, smoke-tested. Training run **not** launched.
