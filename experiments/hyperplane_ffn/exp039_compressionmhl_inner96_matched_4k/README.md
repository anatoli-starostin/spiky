# exp039 — CompressionMHL inner=96 (param-matched), 4k steps

The **inner_dim=96 point** of the CompressionMHL inner_dim sweep. Plain
`CompressionMultiHeadLUT` (compress → hard FastMHL → decompress, **no inner residual**), FFN
slot = `x = x + compressionmhl(ln2(x))`. Exact clone of
`exp036_compressionmhl_inner64_matched_4k` **except inner_dim = 96** (and tph re-chosen to
param-match). A/B vs exp036 (inner=64, **1.40699**).

## Param-match (inner=96, NAP=6 → tph=180)
- fixed = compress (384·96+96 = 36,960) + decompress (96·384+384 = 37,248) = **74,208**
- FastMHL budget = 1,179,648 − 74,208 = 1,105,440 → `tph = 1,105,440 / (2^6·96) = 179.9` → **tph = 180**
  → FastMHL = 180·64·96 = **1,105,920**
- per-layer = 74,208 + 1,105,920 = **1,180,128**  (≈ FFN 1,179,648, +480/layer)

| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| **exp039 (inner=96, tph=180)** | **35,795,520** |
| Δ vs exp032 | **+2,880 (+0.0080%)** |

Larger inner ⇒ fewer, wider tables (~180/layer over a 96-d addressed vector); more of the
budget goes to the compress/decompress projections (74,208/layer fixed).
inner_residual=False; decompress weight zero-init; LUT tables no-wd. Everything else
byte-identical to exp036/exp032. (Confirmed by a build smoke.)

## Status
Launched under the owner's GO (sequentially, after exp038). Outputs: `metrics.csv`,
`summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
