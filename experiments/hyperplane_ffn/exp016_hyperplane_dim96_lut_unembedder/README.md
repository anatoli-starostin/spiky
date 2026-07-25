# exp016 — dim-96 single-stream LUT-GPT with a HyperplaneMHL unembedder

**BUILD FOR REVIEW — not launched.** Tests staying in "ranking space" all the way
to the logits: no ranking→Euclidean transition before unembedding.

(Revised from the initial dim-64 build to **E=96, 6 heads × 16**; directory renamed to
`exp016_hyperplane_dim96_lut_unembedder` to match.)

## Architecture (derived from exp014's train.py)
- dim **E=D=96**, single residual stream, **UNTIED** token embeddings, 6 layers, ctx 512, vocab 32768, RoPE 1e4.
- **Standard Euclidean softmax attention** (the one allowed exception), but q/k/v and the output projection are `HyperplaneMultiHeadLUT` (hyperplane, anchor_pairs init, forward "hard").
- **Head split: 6 heads × head_dim 16** (d_qk=d_v=16, 6·16=96). RoPE-even.
- `out_proj` LUT does **double duty**: unify heads (H·d_v=96 → E=96) + the FFN/nonlinearity role. **No separate MLP.**
- **REMOVED entirely**: `residual_lut`, `emb_resid_lut`, and all dual-stream machinery. The only residual contribution per block is `out_proj`.
- **UNEMBEDDER** = `HyperplaneMultiHeadLUT` mapping the 96-dim hidden **directly** to 32768 logits (replaces the linear unembedder).

## Per-site nap / tph (unchanged from the dim-64 build)
| site | nap (K=2^nap) | tph | n_heads | n_out | input_dim |
|------|---------------|-----|---------|-------|-----------|
| qk_lut       | 4 (16) | 64 | 6 | 32 (=2·d_qk) | 96 |
| v_lut        | 6 (64) | 64 | 6 | 16 (=d_v)    | 96 |
| out_proj     | 6 (64) | 64 | 1 | 96 (=E)      | 96 (=H·d_v) |
| **unembedder** | 6 (64) | 4 | 1 | **32768 (=V)** | 96 |

## Param breakdown (smoke-verified, all fp32) — TOTAL 19,893,886
| group | params |
|-------|--------|
| tok_emb (untied) | 3,145,728 |
| qk_lut  tables / hyperplanes | 1,179,648 / 893,952 |
| v_lut   tables / hyperplanes | 2,359,296 / 1,340,928 |
| out_proj tables / hyperplanes | 2,359,296 / 223,488 |
| **unembedder LUT** tables / hyperplanes | **8,388,608 / 2,328** |
| norms + learnable temps | 614 |
| LUT tables total (Lion) | 14,286,848 |
| hyperplane w/b total (AdamW no-wd) | 2,460,696 |
| linear unembedder | 0 (removed) |

Step-1 loss = 10.3972 = ln(32768): uniform logits at init (LUT tables start ~0), correct.

## Recipe (identical to exp014)
16000 steps · device_batch 24 × seq 512 × grad_accum 2 (24,576 tokens/step) · Lion
lut_lr 2e-4 (LUT tables) · AdamW adam_lr 3e-4 no-wd (hyperplane w/b, routed by NAME) ·
grad clip 1.0 · warmup 0.1 cosine · bf16 tables + fp32 master.
