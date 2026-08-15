# exp041 — CompressionMHL inner=192 (param-matched), 4k steps

The **inner_dim=192 point** (= half of d_model 384) of the CompressionMHL inner_dim sweep,
extending the monotonic trend (32<64<96<128 so far). Plain `CompressionMultiHeadLUT`
(compress → hard FastMHL → decompress, **no inner residual**, n_heads=1), FFN slot =
`x = x + compressionmhl(ln2(x))`. Mirrors exp040 (inner=128) with **inner_dim = 192**.

## Param-match (inner=192, NAP=6 → tph=84)
- fixed = compress (384·192+192 = 73,920) + decompress (192·384+384 = 74,112) = **148,032**
- FastMHL budget = 1,179,648 − 148,032 = 1,031,616 → `tph = 1,031,616 / (2^6·192) = 83.95` → **tph = 84**
  → FastMHL = 84·64·192 = **1,032,192**
- per-layer = 148,032 + 1,032,192 = **1,180,224**  (≈ FFN 1,179,648, +576/layer)

| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| **exp041 (inner=192, tph=84)** | **35,796,096** |
| Δ vs exp032 | **+3,456 (+0.0097%)** |

The fixed compress/decompress cost is now **148,032/layer ≈ 12.5%** of the per-layer budget
(up from 8.4% at inner=128), so proportionally less goes to the tables (84 tables over a
192-d vector). This point probes whether widening the compressed space still helps or the
projection overhead starts to dominate.

## Context — inner_dim sweep (all 4096 steps; dense-FFN exp032 = 1.39371)
| exp | inner | tph | val_bpb |
|-----|-------|-----|---------|
| exp038 | 32 | 564 | 1.42181 |
| exp036 | 64 | 276 | 1.40699 |
| exp039 | 96 | 180 | 1.40404 |
| exp040 | 128 | 132 | 1.39903 |
| **exp041** | **192** | **84** | *(this run)* |

## Status
Launched under the owner's GO. inner_residual=False; decompress weight zero-init; LUT tables
no-wd. Everything else byte-identical to exp040/exp032. Outputs: `metrics.csv`,
`summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
