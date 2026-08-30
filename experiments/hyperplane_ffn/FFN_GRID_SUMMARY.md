# FFN-LUT paper grid — consolidated results (untied-vanilla zero-line)

CompressionMHL FFN slot on MinimalGPT (depth6, n_embd384, 6 attn heads, seq512, vocab32768, untied
unembedder, standard dense attention). All runs: full 16k, LR schedule identical (lr 3e-4, warmup 1600,
cosine→3e-5), effective batch 24,576, standard LUT optimizer. Predicted == measured params/FLOP/vBW.

**Zero-line = UNTIED VANILLA 4× MLP FFN = `exp_n_0135_untied_vanilla_baseline_16k` = 1.20144** (7.08M FFN
params / 14.16M FFN-FLOP / 14.16M vBW; 35.79M total; same 16k schedule / effective batch). This is the
apples-to-apples untied anchor for the untied grid. The **tied** vanilla (`exp073`, 1.19665) is kept as a
labeled *reference* row only — note it is **−0.00479 better than untied**: for the vanilla dense FFN,
untying the unembedder *hurts* (opposite of the LUT models, where untying helped), so the tied number was
the harder bar and the untied number is the methodologically-consistent one.

| id | kind | H | d | nap(cells) | tph | total params | FFN params | ×Van | FFN-FLOP | vBW | val_bpb | Δ vs untied vanilla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0118 | reuse | 4 | 48 | 9(512) | 256 | 180.60M | 151.9M | 21.45× | 2.286M | 2.375M | 1.17460 | **−0.02684** |
| 0133 | NEW | 4 | 48 | 10(1024) | 128 | 180.60M | 151.9M | 21.45× | 2.040M | 2.081M | 1.17961 | −0.02183 |
| 0129 | NEW | 4 | 48 | 8(256) | 256 | 105.10M | 76.4M | 10.79× | 2.261M | 2.375M | 1.18148 | −0.01996 |
| 0119 | reuse | 4 | 48 | 9(512) | 128 | 105.10M | 76.4M | 10.79× | 2.028M | 2.081M | 1.18386 | −0.01758 |
| 0131 | NEW | 2 | 96 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 1.966M | 2.081M | 1.18883 | −0.01261 |
| 0121 | reuse | 4 | 48 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 2.015M | 2.081M | 1.19146 | −0.00999 |
| 0132 | NEW | 8 | 24 | 8(256) | 128 | 67.35M | 38.6M | 5.46× | 2.114M | 2.081M | 1.19263 | −0.00881 |
| 0130 | NEW | 4 | 48 | 10(1024) | 64 | 105.10M | 76.4M | 10.79× | 1.905M | 1.933M | 1.19405 | −0.00739 |
| 0127 | NEW | 4 | 48 | 7(128) | 128 | 48.48M | 19.8M | 2.79× | 2.003M | 2.081M | 1.19471 | **−0.00673** |
| **exp073** | reference (tied) | — | — | dense 4×MLP | — | 23.21M | 7.08M | 1.00× | 14.16M | 14.16M | **1.19665** | −0.00479 |
| 0120 | reuse | 4 | 48 | 9(512) | 64 | 67.35M | 38.6M | 5.46× | 1.898M | 1.933M | 1.19859 | −0.00285 |
| 0084 | reuse | 4 | 48 | 7(128) | 256 | 67.35M | 38.6M | 5.46× | 2.236M | 2.375M | 1.19866 | −0.00278 |
| **exp_n_0135** | vanilla (untied) | — | — | dense 4×MLP | — | 35.79M | 7.08M | 1.00× | 14.16M | 14.16M | **1.20144** | **0 (zero-line)** |
| 0128 | NEW | 4 | 48 | 8(256) | 64 | 48.48M | 19.8M | 2.79× | 1.892M | 1.933M | 1.20228 | +0.00083 |
| 0125 | reuse | 8 | 24 | 8(256) | 64 | 48.48M | 19.8M | 2.79× | 1.942M | 1.933M | 1.20332 | +0.00188 |
| 0126 | NEW | 4 | 48 | 7(128) | 64 | 39.04M | 10.3M | 1.46× | 1.886M | 1.933M | 1.20694 | +0.00550 |

## Headline vs untied vanilla
- **Cheapest routed FFN beating untied vanilla:** 0127 (48.48M, 2.79× FFN, **−0.00673**).
- **Best in the 10.8× budget:** 0129 nap8/tph256 (**−0.01996**).
- **Best overall:** 0118 nap9/tph256 (21.5×, **−0.02684**).
- Against the untied zero-line, *every routed config beats it* except the two cell-heavy tph64 points
  (0128, 0125) and the smallest 1.46× point (0126). 0084 (nap7/tph256) now clears vanilla too (−0.00278),
  since untying moved the bar down.
- **Tied vanilla is a reference at −0.00479** (tied beats untied for the vanilla FFN). Using the untied
  number as the zero-line is the methodologically-consistent choice for the untied grid; it flatters the
  routed results relative to the tied bar by ~0.0048, so both numbers are reported.

## Findings (deltas vs untied vanilla)
1. **Don't starve tph.** The low-tph / high-cell corner is worst at every iso-param level; keep tph ≥ 128.
   Ceiling diagonal (10.8×) monotone: nap8/tph256 −0.01996 < nap9/tph128 −0.01758 < nap10/tph64 −0.00739.
   5.45× is an interior optimum — nap8/tph128 (0121, −0.00999) beats both extremes nap7/tph256 (0084,
   −0.00278) and nap9/tph64 (0120, −0.00285).
2. **Fewer, wider routing heads win** (head line @67.35M, H·d=192 fixed): H2/d96 −0.01261 < H4/d48 −0.00999
   < H8/d24 −0.00881.
3. **Fixed-param compute trade:** at iso-param/iso-C, more-tables wins bpb while fewer-tables is cheaper
   FLOP/vBW (0133 2.040M/2.081M vs 0118 2.286M/2.375M at 180.6M).

## Best configs for the paper
- **Max quality:** 0118 nap9/tph256 (180.6M, 1.17460, −0.0268 vs untied vanilla).
- **Best at ~vanilla-adjacent budget (67.35M):** 0131 H2/d96 (1.18883, −0.0126).
- **Cheapest beating vanilla:** 0127 nap7/tph128 (48.48M, 2.79×, 1.19471, −0.0067).

See `FFN_GRID_plots.png` — points labeled by architecture (cells×tables = 2^nap × tph; head count by
color H2·d96 / H4·d48 / H8·d24; H·d=192 for every point), the anchor (H4·d48, 2^8×128t) is a gold star,
and the untied vanilla 1.20144 line is the zero reference. (a) params↔bpb, (b) iso-param diagonals
(bpb vs vBW), (c) the H2/H4/H8 head line.
