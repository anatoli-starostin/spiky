# exp038 — CompressionMHL inner=32 (param-matched), 4k steps

The **inner_dim=32 point** of the CompressionMHL inner_dim sweep. Plain
`CompressionMultiHeadLUT` (compress → hard FastMHL → decompress, **no inner residual**), FFN
slot = `x = x + compressionmhl(ln2(x))`. Exact clone of
`exp036_compressionmhl_inner64_matched_4k` **except inner_dim = 32** (and tph re-chosen to
param-match). A/B vs exp036 (inner=64, **1.40699**).

## Param-match (inner=32, NAP=6 → tph=564)
- fixed = compress (384·32+32 = 12,320) + decompress (32·384+384 = 12,672) = **24,992**
- FastMHL budget = 1,179,648 − 24,992 = 1,154,656 → `tph = 1,154,656 / (2^6·32) = 563.8` → **tph = 564**
  → FastMHL = 564·64·32 = **1,155,072**
- per-layer = 24,992 + 1,155,072 = **1,180,064**  (≈ FFN 1,179,648, +416/layer)

| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| **exp038 (inner=32, tph=564)** | **35,795,136** |
| Δ vs exp032 | **+2,496 (+0.0070%)** |

Smaller inner ⇒ more, narrower tables (~564/layer over a 32-d addressed vector).
inner_residual=False; decompress weight zero-init; LUT tables no-wd. Everything else
byte-identical to exp036/exp032. (Confirmed by a build smoke.)

## Status
Launched under the owner's GO (sequentially, before exp039). Outputs: `metrics.csv`,
`summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
