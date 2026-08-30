# FFN-LUT paper grid — consolidated results (task 6e8b7c23 + add-ons)

CompressionMHL FFN slot on MinimalGPT (depth6, n_embd384, 6 attn heads, seq512, vocab32768, untied
unembedder, standard dense attention). All runs: full 16k, LR schedule identical (lr 3e-4, warmup 1600,
cosine→3e-5), effective batch 24,576, standard LUT optimizer. 8 new runs (0126–0133) + 6 reuse points.
Predicted == measured params/FLOP/vBW throughout. Vanilla 4× MLP FFN = 7.08M params; dense-V baseline
= exp_n_0084 = **1.19866**.

| id | new | H | d | nap(cells) | tph | total params | FFN params (×vanilla) | FFN-FLOP | vBW | val_bpb | Δ vs dense-V |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0126 | NEW | 4 | 48 | 7(128) | 64 | 39.04M | 10.3M (1.46×) | 1.886M | 1.933M | 1.20694 | +0.00828 |
| 0127 | NEW | 4 | 48 | 7(128) | 128 | 48.48M | 19.8M (2.79×) | 2.003M | 2.081M | 1.19471 | −0.00395 |
| 0128 | NEW | 4 | 48 | 8(256) | 64 | 48.48M | 19.8M (2.79×) | 1.892M | 1.933M | 1.20228 | +0.00362 |
| 0125 | reuse | 8 | 24 | 8(256) | 64 | 48.48M | 19.8M (2.79×) | 1.942M | 1.933M | 1.20332 | +0.00466 |
| 0131 | NEW | 2 | 96 | 8(256) | 128 | 67.35M | 38.6M (5.46×) | 1.966M | 2.081M | **1.18883** | −0.00983 |
| 0121 | reuse | 4 | 48 | 8(256) | 128 | 67.35M | 38.6M (5.46×) | 2.015M | 2.081M | 1.19146 | −0.00720 |
| 0132 | NEW | 8 | 24 | 8(256) | 128 | 67.35M | 38.6M (5.46×) | 2.114M | 2.081M | 1.19263 | −0.00603 |
| 0120 | reuse | 4 | 48 | 9(512) | 64 | 67.35M | 38.6M (5.46×) | 1.898M | 1.933M | 1.19859 | −0.00007 |
| 0084 | reuse | 4 | 48 | 7(128) | 256 | 67.35M | 38.6M (5.46×) | 2.236M | 2.375M | 1.19866 | 0 (dense-V) |
| 0129 | NEW | 4 | 48 | 8(256) | 256 | 105.10M | 76.4M (10.79×) | 2.261M | 2.375M | **1.18148** | −0.01718 |
| 0119 | reuse | 4 | 48 | 9(512) | 128 | 105.10M | 76.4M (10.79×) | 2.028M | 2.081M | 1.18386 | −0.01480 |
| 0130 | NEW | 4 | 48 | 10(1024) | 64 | 105.10M | 76.4M (10.79×) | 1.905M | 1.933M | 1.19405 | −0.00461 |
| 0118 | reuse | 4 | 48 | 9(512) | 256 | 180.60M | 151.9M (21.45×) | 2.286M | 2.375M | **1.17460** | −0.02406 |
| 0133 | NEW | 4 | 48 | 10(1024) | 128 | 180.60M | 151.9M (21.45×) | 2.040M | 2.081M | 1.17961 | −0.01905 |

## params↔bpb Pareto frontier (all points)
| params | winner | config | val_bpb | Δ vs dense-V |
|---|---|---|---|---|
| 39.04M | 0126 | nap7/tph64 | 1.20694 | +0.008 |
| 48.48M | 0127 | nap7/tph128 | 1.19471 | −0.004 |
| 67.35M | 0131 | H2/d96 nap8/tph128 | 1.18883 | −0.010 |
| 105.10M | 0129 | nap8/tph256 | 1.18148 | −0.017 |
| 180.60M | 0118 | nap9/tph256 | 1.17460 | −0.024 |

Routed-FFN **beats dense-V at equal params** (67.35M: 0131 −0.0098) and **with 28% fewer** (0127, 48.48M,
−0.004). Only the smallest (39M/1.46×) sits above dense-V. dense-V's own config (0084, nap7/tph256) is one
of the *worst* 67.35M allocations — better allocations at equal params beat it by up to −0.010.

## Finding 1 — tables > cells, but starving tph is the real killer
At each iso-param level the **low-tph / high-cell** corner is consistently the worst; high-tph the best:
- 2.8× (48.48M): tph128 (0127, 1.19471) **>** tph64 (0128, 1.20228) — Δ−0.0076
- 10.8× ceiling (105.1M), monotone: nap8/tph256 1.18148 < nap9/tph128 1.18386 < nap10/tph64 1.19405
- 21.5× (180.6M): nap9/tph256 (0118, 1.17460) **>** nap10/tph128 (0133, 1.17961) — Δ−0.0050
- 5.45× (67.35M) is the exception — an **interior optimum**: nap8/tph128 (0121, 1.19146) beats *both*
  extremes nap7/tph256 (0084, 1.19866) and nap9/tph64 (0120, 1.19859), which are ~equal.

Net: keep **tph ≥ 128**; every tph64 config underperforms its iso-param siblings. Spend the table budget
on tables first; at low-to-mid budget a balanced nap8/tph128 is safest, at high budget push tph.

## Finding 2 — fewer, wider routing heads win (head line @67.35M, H·d=192 fixed)
Monotone: **H2/d96 1.18883 < H4/d48 1.19146 < H8/d24 1.19263**. Fewer/wider heads beat more/narrower at
fixed params & table budget. (And 0132 H8/tph128 1.19263 ≫ 0125 H8/tph64 1.20332 — tph≥128 again.)

## Finding 3 — cheaper-compute redistribution at fixed params
At iso-param/iso-C, high-tph/low-cell (0118 nap9/tph256) and low-tph/high-cell (0133 nap10/tph128) both
exist; the high-tph one wins on bpb *and* the low-tph one is cheaper on FLOP/vBW (2.040M/2.081M vs
2.286M/2.375M). So there's a small quality↔compute trade within a fixed param budget, but the quality
winner (more tables) costs modestly more compute.

## Best configs for the paper
- **Max quality:** 0118 nap9/tph256 (180.6M, 1.17460, −0.024 vs dense-V).
- **Best at dense-V's param budget (67.35M):** 0131 H2/d96 (1.18883, −0.010).
- **Cheapest that beats dense-V:** 0127 nap7/tph128 (48.48M, 2.79×, 1.19471, −0.004) — 28% fewer params.

See `FFN_GRID_plots.png` — (a) params↔bpb Pareto, (b) iso-param diagonals (bpb vs vBW; more tables win),
(c) the H2/H4/H8 head line.
