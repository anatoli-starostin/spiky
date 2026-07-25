# exp019 — phase 2: single-stream champion lineage + HyperplaneCodeLUT unembedder

Reduces the E=384 champion lineage to a **single residual stream** with the
**code-LUT unembedder**. From exp018 (LayerNorm version of exp010) with two changes:

1. **Removed the dual stream** — no `emb_resid_lut` (embedding→D seed), no per-layer
   `residual_lut` (E→D, ×6), no D-stream. The model is a single residual stream x of
   dim E=384. Each block: **LayerNorm(x) → qk_lut & v_lut → RoPE softmax attention →
   out_proj_lut → x += out_proj**. After all layers: **ln_final(x) → unembedder →
   logits** (the final LayerNorm is KEPT).
2. **Unembedder → HyperplaneCodeLUT** (`src/spiky/lutorch/hyperplane_code_lut.py`):
   nap=15 → V=2^15=32768, T=16 tables, input dim E=384. Per-code soft-sign scores gated
   by `w_cell`, voted over 16 tables — no stored per-cell rows, emits logits directly.

Everything else identical to exp010/exp018: 6 layers, 6 heads × d_qk=d_v=64, ctx 512,
vocab 32768, RoPE 1e4, untied tok_emb; LUT sites qk nap4/tph256, v nap6/tph256, out_proj
nap7/tph512 (HyperplaneMultiHeadLUT, anchor-pairs, hard, bf16 tables/fp32 hyperplanes);
LayerNorm everywhere; seed 42; 16000 steps, 24,576 tokens/step, LR over 16000
(warmup 1600, cosine to 0.1× floor), Lion lut_lr 2e-4 / AdamW adam_lr 3e-4, grad clip 1.0.

## Params (smoke-verified) — Total 264,585,236 (vs exp010's 324.7M)
- LUT tables 207,618,048 (Lion) — qk/v/out only (residual/emb_resid removed).
- Hyperplanes 43,853,040 (AdamW no-wd) — qk/v/out hyperplanes + **code-LUT hyperplane_weight/bias 92,400**.
- **w_cell 524,288 (AdamW wd 0.1)** — the unembedder gate (old linear-unembedder slot).
- tok_emb 12,582,912 · norms/temps 6,948.
- Unembedder total = 616,688 (92,400 + 524,288). All unembedder params routed (verified: groups sum to the total, nothing unrouted).
- Step-1 loss 10.4157 ≈ ln(32768)=10.397 (near-uniform code-LUT init).

## Baselines
exp010 dual-stream champion best_val_bpb **1.1940** @ 324.7M (this is a much smaller/cheaper
model); exp018 = the LayerNorm A/B of exp010. Same seed/data.
