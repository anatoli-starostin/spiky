# exp016 — dim-64 single-stream LUT-GPT with a HyperplaneMHL unembedder

**BUILD FOR REVIEW — not launched.** Tests staying in "ranking space" all the way
to the logits: no ranking→Euclidean transition before unembedding.

## Architecture (derived from exp014's train.py)
- dim **E=D=64**, single residual stream, **UNTIED** token embeddings, 6 layers, ctx 512, vocab 32768, RoPE 1e4.
- **Standard Euclidean softmax attention** (the one allowed exception), but q/k/v and the output projection are `HyperplaneMultiHeadLUT` (hyperplane, anchor_pairs init, forward "hard").
- **Head split: 4 heads × head_dim 16** (d_qk=d_v=16, 4·16=64). RoPE-even.
- `out_proj` LUT does **double duty**: unify heads (H·d_v=64 → E=64) + the FFN/nonlinearity role. **No separate MLP.**
- **REMOVED entirely**: `residual_lut`, `emb_resid_lut`, and all dual-stream machinery. The only residual contribution per block is `out_proj`.
- **UNEMBEDDER** = `HyperplaneMultiHeadLUT` mapping the 64-dim hidden **directly** to 32768 logits (replaces the linear unembedder): `unemb_nap` sign-tests on the E-dim stream → index bits → LUT rows → logits.

## Per-site nap / tph
| site | nap (K=2^nap) | tph | n_heads | n_out | input_dim |
|------|---------------|-----|---------|-------|-----------|
| qk_lut       | 4 (16) | 64 | 4 | 32 (=2·d_qk) | 64 |
| v_lut        | 6 (64) | 64 | 4 | 16 (=d_v)    | 64 |
| out_proj     | 6 (64) | 64 | 1 | 64 (=E)      | 64 (=H·d_v) |
| **unembedder** | 6 (64) | 4 | 1 | **32768 (=V)** | 64 |

## Param breakdown (smoke-verified, all fp32) — TOTAL 15,568,190
| group | params |
|-------|--------|
| tok_emb (untied) | 2,097,152 |
| qk_lut  tables / hyperplanes | 786,432 / 399,360 |
| v_lut   tables / hyperplanes | 1,572,864 / 599,040 |
| out_proj tables / hyperplanes | 1,572,864 / 149,760 |
| **unembedder LUT** tables / hyperplanes | **8,388,608 / 1,560** |
| norms + learnable temps | 550 |
| LUT tables total (Lion) | 12,320,768 |
| hyperplane w/b total (AdamW no-wd) | 1,149,720 |
| linear unembedder | 0 (removed) |

Step-1 loss = 10.3972 = ln(32768): uniform logits at init (LUT tables start ~0), correct.

## Recipe (identical to exp014)
16000 steps · device_batch 24 × seq 512 × grad_accum 2 (24,576 tokens/step) · Lion
lut_lr 2e-4 (LUT tables) · AdamW adam_lr 3e-4 no-wd (hyperplane w/b, routed by NAME) ·
grad clip 1.0 · warmup 0.1 cosine · bf16 tables + fp32 master.
