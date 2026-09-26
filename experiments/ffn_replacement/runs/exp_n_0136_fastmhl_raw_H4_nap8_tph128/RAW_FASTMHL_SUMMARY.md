# Raw FastMHL FFN (no compress/decompress) — H4/nap8/tph128, full 16k (exp_n_0136)

Point 1 of the raw-FastMHL-as-FFN ablation. FFN slot = `FastMultiHeadLut(input_dim=384, n_heads=4,
n_outputs=384, nap=8/256 cells, tph=128)` — routes on the full 384-d input, tables emit the full 384-d
output (heads summed), **no compress and no decompress Linear**. Raw head-to-head vs grid anchor 0121
(same H4/nap8/tph128 but CompressionMHL compress48 + decompress48). Grid-standard: UNTIED, 16k, bs12/ga4.

## Result — final val_bpb 1.20567 (converged)
| comparison | bpb | Δ (raw − other) |
|---|---|---|
| **exp_n_0136 raw FastMHL** (330.7M total, 302.0M FFN, 42.7× vanilla) | **1.20567** | — |
| untied vanilla 4× MLP (exp_n_0135, 7.08M FFN) | 1.20144 | **+0.00423** (raw is WORSE) |
| CompressionMHL anchor 0121 (same H/nap/tph, 38.6M FFN, 5.46×) | 1.19146 | **+0.01421** (raw is much WORSE) |

## Headline — the projections do real work; raw FastMHL is inefficient
The raw FastMHL FFN, at **42.7× the vanilla FFN params (302M)**, is **worse than the plain vanilla 4× MLP
(+0.0042)** and far worse than the *same-routing* CompressionMHL (0121) which uses **8× fewer FFN params**
(38.6M) yet beats it by −0.0142. So the CompressionMHL compress→LUT→decompress bottleneck is **not merely a
parameter-saving trick** — the learned input projection (which shapes the routing space) and the learned
decompress (which mixes the per-table outputs) add representational capacity the raw gather cannot buy back
with 8× the params.

## Cost (predicted == measured; profiler models only CompressionMHL, FLOP/vBW analytic)
- total params 330,704,652; FFN table 301,989,888 = 302.0M (= 42.7× vanilla).
- FFN-FLOP ~1.28M (pure gather+sum, no matmul — *lower* than 0121's 2.015M).
- vBW ~2.36M (2·6·4·128·384 selected 384-wide rows; ~0121's 2.081M; no dense weights).
- fits at bs12, peak 15.6GB.

## Implication for the bpb-match search (target = untied vanilla 1.20144)
The task expected this over-capacity point to land *below* 1.20144 and to bracket downward. It landed
**above** it (+0.0042). So raw FastMHL at H4/nap8/tph128 already *fails to match vanilla* despite 42.7×
params — to reach 1.20144 the raw config would need **more** capacity, not less, and the efficiency story is
clear regardless: raw FastMHL is a poor FFN. The interesting paper number is this contrast (raw 42.7× loses
to compressed 5.46× and to vanilla 1.0×), not a matched-capacity point.
