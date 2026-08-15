# exp042 — CompressionMHL n_heads=3 (independent), inner=64, param-matched, 4k steps

A **per-head-routing A/B against exp041** (inner=192, single head). Same total params and
same 192-wide inner budget, but split **block-diagonally into 3 independent 64-d heads**
instead of one 192-d head. `CompressionMultiHeadLUT` with **n_heads=3,
joint_head_compression=False** (genuinely independent per-head), inner_in_dim=inner_out_dim=64,
no inner residual, NAP=6, hard. FFN slot = `x = x + compressionmhl(ln2(x))`.

## Independent per-head structure
- **compress** = one `Linear(384 → 3·64=192)`; its 3 row-blocks are the per-head compress maps.
- **3 × FastMHL** — each head runs its OWN single-head `FastMultiHeadLut(64 → 64)` on its own
  64-d slice `z_h` (per-head anchor seed = 1000+layer + h).
- **decompress** = one `Linear(3·64=192 → 384)` over the concatenated per-head outputs
  (≡ summed per-head decompress).

Contrast with exp041 (inner=192, single head): there, one FastMHL reads a full 192-d vector;
here, three FastMHLs each read a 64-d slice (block-diagonal addressing). Tests whether
per-head routing beats one wide head at equal params.

## Param-match (n_heads=3 independent, inner=64, NAP=6 → tph=84)
- fixed = compress (384·192+192 = 73,920) + decompress (192·384+384 = 74,112) = **148,032**
- FastMHL budget = 1,179,648 − 148,032 = 1,031,616 → `tph = 1,031,616 / (3·2^6·64) = 83.95` → **tph = 84**
  → FastMHL = 3·84·64·64 = **1,032,192**
- per-layer = 148,032 + 1,032,192 = **1,180,224**  (≈ FFN 1,179,648, +576/layer)

| | params |
|---|---|
| exp032 (vanilla) | 35,792,640 |
| **exp042 (3 heads, inner=64, tph=84)** | **35,796,096** |
| Δ vs exp032 | **+3,456 (+0.0097%)** |
| (same total as exp041 inner=192 single-head — clean A/B) | |

## Context — CompressionMHL sweep so far (all 4096 steps; dense-FFN exp032 = 1.39371)
| exp | config | val_bpb |
|-----|--------|---------|
| exp040 | inner=128, 1 head | **1.39903** (best) |
| exp039 | inner=96, 1 head | 1.40404 |
| exp036 | inner=64, 1 head | 1.40699 |
| exp041 | inner=192, 1 head | 1.40936 |
| **exp042** | **inner=64 × 3 heads (indep)** | *(this run)* |

## Status
Launched under the owner's GO. Everything else byte-identical to the sweep. Outputs:
`metrics.csv`, `summary.json`, `loss.png`, `checkpoint.pt` (checkpoint gitignored).
