# exp025 — FastMHL backbone control (single-stream + Linear)

Apples-to-apples control against the learnable-hyperplane runs. **Exact clone of
exp024**'s config/architecture with ONE change: the backbone LUT module type
`HyperplaneMultiHeadLUT` → **`FastMultiHeadLut`** (fixed balanced anchor-pair sign
tests, no learnable hyperplane projection). This isolates the value of *learning* the
address hyperplanes vs. the fixed anchor-pair addressing at the single-stream + Linear
architecture.

## Everything else = exp024 / exp023 (byte-identical)
Single residual stream; LayerNorm pre-norms incl. `ln_final`; plain
`nn.Linear(384 → 32768, bias=False)` unembedder; backbone 6L / E384 / 6h×d_qk=d_v=64 /
ctx512 / vocab32768 / RoPE1e4 / untied tok_emb; LUT qk nap4/tph256, v nap6/tph256,
out_proj nap7/tph512 (hard); FastMHL anchors = `get_balanced_anchor_pairs` under seed 42
(CANONICAL_FULL_COVERAGE); 16000 steps, 24,576 tokens/step (bs24×seq512×ga2), warmup 1600
cosine to 0.1× floor, Lion lut_lr 2e-4 tables / AdamW adam_lr 3e-4, grad clip 1.0,
eval every 200. The AdamW no-wd hyperplane param group is **empty** for FastMHL (expected).

## Params — Total 232,790,820 (−43,760,640 vs exp024's 276,551,460)
FastMHL drops all `hyperplane_weight`/`hyperplane_bias` params: `T·nap·(E+1)` per site,
summed over 6 layers:
| site | T = H·tph | nap | E+1 | per-layer | ×6 |
|---|---|---|---|---|---|
| qk | 6·256=1536 | 4 | 385 | 2,365,440 | 14,192,640 |
| v | 6·256=1536 | 6 | 385 | 3,548,160 | 21,288,960 |
| out_proj | 1·512=512 | 7 | 385 | 1,379,840 | 8,279,040 |
| **total** | | | | | **43,760,640** |

Remaining params identical to exp024: LUT tables 207,618,048 (Lion) · Linear unembedder
12,582,912 (AdamW wd 0.1) · tok_emb 12,582,912 · norms 6,948.
**Step-1 loss 10.5797** (≈ exp024's 10.5769; Linear-unembedder init).

## Baselines
exp010 dual-stream **1.1940**; exp023 single-stream + Linear learnable random-init
**1.2063**; exp024 learnable anchor-init **1.2034**. Same seed/data. exp025 measures the
fixed-anchor floor of this same architecture.

## Note
exp024's `train.py` `'fast'` branch had a latent bug: it passed
`backward_mode=cfg.get('backward_mode', 'ball')` to `FastMultiHeadLut`, which has no such
kwarg (would `TypeError` at construction). Removed in this exp025 `train.py`; the rest is
byte-identical to exp024's harness.
