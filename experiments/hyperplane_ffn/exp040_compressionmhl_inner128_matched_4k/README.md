# exp040 — CompressionMHL inner=128 (param-matched), 4k steps

The **inner_dim=128 point** of the CompressionMHL inner_dim sweep — continuing the monotonic
trend (32 < 64 < 96 so far). Plain `CompressionMultiHeadLUT` (compress → hard FastMHL →
decompress, **no inner residual**, n_heads=1), FFN slot = `x = x + compressionmhl(ln2(x))`.
Mirrors exp039 (inner=96) with **inner_dim = 128** (and tph re-chosen to param-match).

## Param-match (inner=128, NAP=6 → tph=132)
- fixed = compress (384·128+128 = 49,280) + decompress (128·384+384 = 49,536) = **98,816**
- FastMHL budget = 1,179,648 − 98,816 = 1,080,832 → `tph = 1,080,832 / (2^6·128) = 131.94` → **tph = 132**
  → FastMHL = 132·64·128 = **1,081,344**
- per-layer = 98,816 + 1,081,344 = **1,180,160**  (≈ FFN 1,179,648, +512/layer)

| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| **exp040 (inner=128, tph=132)** | **35,795,712** |
| Δ vs exp032 | **+3,072 (+0.0086%)** |

The fixed compress/decompress cost (98,816/layer ≈ 8.4% of the per-layer budget) keeps
growing with inner_dim, leaving less for the tables (132 wide tables over a 128-d vector).
This run tests whether inner keeps helping or the projection overhead starts to bite.

## Context — inner_dim sweep (all 4096 steps)
| exp | inner | tph | val_bpb |
|-----|-------|-----|---------|
| exp038 | 32 | 564 | 1.42181 |
| exp036 | 64 | 276 | 1.40699 |
| exp039 | 96 | 180 | 1.40404 |
| **exp040** | **128** | **132** | *(this run)* |

(dense-FFN baseline exp032 = 1.39371.)

## Status
Launched under the owner's GO. inner_residual=False; decompress weight zero-init; LUT tables
no-wd. Everything else byte-identical to exp039/exp032. Outputs: `metrics.csv`,
`summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
