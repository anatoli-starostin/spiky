# exp091 — C1 with tph 84→96 (over-budget probe), tied, 4k

Clone of Sweep-C **C1 = exp075** (tied, 6 heads, CompressionMHL inner_in=64/inner_out=64,
nap6, gamma0, AdamW-LUT, 4096 steps) with **ONLY tph changed 84 → 96**. Everything else
identical.

**Intentionally over the 2× budget:** tph=96 pushes the FFN slot above the param-matched
level (tph=84 = 30,292,224 total). Per layer: compress 147,840 + tables (6·96·64·64 =
2,359,296) + decompress 147,840 = **2,654,976** → total = 16,131,840 + 6·2,654,976 =
**32,061,696** (+1,774,080 = +5.9% over the 30.29M 2× target; +2.65M over exp075). Not
rescaled — the point is to see if more tables cross below the tied dense baseline.

References: C1/exp075 = 1.36613; tied dense exp055 (4k) = 1.35543.

AdamW two-group optimizer (LUT tables no-wd, rest wd 0.1), tied embeddings. One 4k run.
Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt (gitignored).
