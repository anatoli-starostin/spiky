# FFN-LUT paper grid — consolidated results (vanilla zero-line)

CompressionMHL FFN slot on MinimalGPT (depth6, n_embd384, 6 attn heads, seq512, vocab32768, untied
unembedder, standard dense attention). All runs: full 16k, LR schedule identical (lr 3e-4, warmup 1600,
cosine→3e-5), effective batch 24,576, standard LUT optimizer. 8 new runs (0126–0133) + reuse points.
Predicted == measured params/FLOP/vBW throughout.

**Zero-line = VANILLA 4× MLP FFN = `exp073_tied_vanilla_baseline_16k` = 1.19665** (7.08M FFN params /
14.16M FFN-FLOP / 14.16M vBW; same 16k schedule / effective batch). Per the researcher, tied-vs-untied
vanilla bpb is within noise, so exp073's tied number is used directly as the untied-grid anchor — no
correction. Every other point is a routed CompressionMHL FFN; exp_n_0084 (nap7/tph256) is one such reuse
point, not a separate baseline, and it lands **+0.002 above vanilla** (slightly worse), which is worth
noting because nap7/tph256 is a poor allocation of its 67.35M budget.

| id | kind | H | d | nap(cells) | tph | total params | FFN params | ×Van | FFN-FLOP | vBW | val_bpb | Δ vs vanilla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0118 | reuse | 4 | 48 | 9(512) | 256 | 180.60M | 151.9M | 21.45× | 2.286M | 2.375M | 1.17460 | **−0.02205** |
| 0133 | NEW | 4 | 48 | 10(1024) | 128 | 180.60M | 151.9M | 21.45× | 2.040M | 2.081M | 1.17961 | −0.01704 |
| 0129 | NEW | 4 | 48 | 8(256) | 256 | 105.10M | 76.4M | 10.79× | 2.261M | 2.375M | 1.18148 | −0.01517 |
| 0119 | reuse | 4 | 48 | 9(512) | 128 | 105.10M | 76.4M | 10.79× | 2.028M | 2.081M | 1.18386 | −0.01279 |
| 0131 | NEW | 2 | 96 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 1.966M | 2.081M | 1.18883 | −0.00782 |
| 0121 | reuse | 4 | 48 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 2.015M | 2.081M | 1.19146 | −0.00519 |
| 0132 | NEW | 8 | 24 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 2.114M | 2.081M | 1.19263 | −0.00402 |
| 0130 | NEW | 4 | 48 | 10(1024) | 64 | 105.10M | 76.4M | 10.79× | 1.905M | 1.933M | 1.19405 | −0.00260 |
| 0127 | NEW | 4 | 48 | 7(128) | 128 | 48.48M | 19.8M | 2.79× | 2.003M | 2.081M | 1.19471 | **−0.00194** |
| **exp073** | vanilla | — | — | dense 4×MLP | — | 23.21M (tied) | 7.08M | 1.00× | 14.16M | 14.16M | **1.19665** | **0 (zero-line)** |
| 0120 | reuse | 4 | 48 | 9(512) | 64 | 67.35M | 38.6M | 5.46× | 1.898M | 1.933M | 1.19859 | +0.00194 |
| 0084 | reuse | 4 | 48 | 7(128) | 256 | 67.35M | 38.6M | 5.46× | 2.236M | 2.375M | 1.19866 | +0.00201 |
| 0128 | NEW | 4 | 48 | 8(256) | 64 | 48.48M | 19.8M | 2.79× | 1.892M | 1.933M | 1.20228 | +0.00563 |
| 0125 | reuse | 8 | 24 | 8(256) | 64 | 48.48M | 19.8M | 2.79× | 1.942M | 1.933M | 1.20332 | +0.00667 |
| 0126 | NEW | 4 | 48 | 7(128) | 64 | 39.04M | 10.3M | 1.46× | 1.886M | 1.933M | 1.20694 | +0.01029 |

## Headline vs vanilla
- **Cheapest routed FFN beating vanilla:** 0127 (48.48M, 2.79× FFN, **−0.00194**) — beats the vanilla 4× MLP
  while the LUT reads only selected rows.
- **Best in the 10.8× budget:** 0129 nap8/tph256 (**−0.01517**).
- **Best overall:** 0118 nap9/tph256 (21.5×, **−0.02205**).
- **0084 (nap7/tph256) sits +0.00201 ABOVE vanilla** — its nap7/tph256 arrangement is one of the worst
  67.35M allocations; better routed configs at equal params (0131 −0.0078, 0121 −0.0052) clear both it and
  vanilla.
- Everything from 2.79× up beats vanilla except the two cell-heavy tph64 configs (0128, 0125) and the
  smallest 1.46× point (0126).

## Findings (deltas reframed vs vanilla)
1. **Don't starve tph.** The low-tph / high-cell corner is the worst at every iso-param level; keep
   tph ≥ 128. Ceiling diagonal (10.8×) is monotone: nap8/tph256 −0.01517 < nap9/tph128 −0.01279 <
   nap10/tph64 −0.00260. The 5.45× level is an interior optimum — nap8/tph128 (0121, −0.0052) beats both
   extremes nap7/tph256 (0084, +0.0020) and nap9/tph64 (0120, +0.0019).
2. **Fewer, wider routing heads win** (head line @67.35M, H·d=192 fixed): H2/d96 −0.00782 < H4/d48 −0.00519
   < H8/d24 −0.00402.
3. **Fixed-param compute trade:** at iso-param/iso-C, more-tables wins bpb while fewer-tables is cheaper
   FLOP/vBW (0133 2.040M/2.081M vs 0118 2.286M/2.375M at 180.6M).

## Best configs for the paper
- **Max quality:** 0118 nap9/tph256 (180.6M, 1.17460, −0.0221 vs vanilla).
- **Best at ~vanilla-adjacent budget (67.35M):** 0131 H2/d96 (1.18883, −0.0078).
- **Cheapest beating vanilla:** 0127 nap7/tph128 (48.48M, 2.79×, 1.19471, −0.0019).

See `FFN_GRID_plots.png` — (a) params↔bpb Pareto with the vanilla 1.19665 reference line (all routed
points, including 0084, plotted uniformly by exp id), (b) iso-param diagonals (bpb vs vBW; more tables
win) with the vanilla line, (c) H2/H4/H8 head line with the vanilla line.
