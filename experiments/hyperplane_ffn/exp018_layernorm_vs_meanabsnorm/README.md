# exp018 — champion exp010 with MeanAbsNorm → LayerNorm (A/B)

Single-variable A/B of the champion **exp010** (E=384 dual-stream HyperplaneMHL,
best_val_bpb **1.1940**). **Exactly one change**: the three `MeanAbsNorm(E)` pre-norms
(`ln_pre`, `ln_resid`, `ln_emb_resid`) are replaced by `nn.LayerNorm(E)` (standard
affine). `ln_final` was already `nn.LayerNorm` and is unchanged.

Everything else is byte-identical to exp010:
- E=D=384, 6 layers, 6 heads × d_qk=d_v=64, context 512, vocab 32768, RoPE 1e4, untied embeddings.
- Dual-stream: E-stream (attention: qk/v/out LUTs) + D-stream (residual via per-layer `residual_lut` + `emb_resid_lut`) → ln_final → linear unembedder.
- LUT sites (HyperplaneMultiHeadLUT, forward "hard", anchor_pairs init): qk nap4/tph256, v nap6/tph256, out_proj nap7/tph512, residual nap6/tph256 (×6), emb_resid nap6/tph256.
- Recipe: 16000 steps, 24,576 tokens/step (bs24×seq512×ga2), LR over 16000 (warmup 0.1=1600, cosine to 0.1× floor), Lion lut_lr 2e-4 (LUT tables) / AdamW adam_lr 3e-4 (hyperplanes no-wd, unembed wd 0.1), grad clip 1.0, bf16 tables + fp32 master.

## Params (smoke-verified)
Total **324,736,562** = exp010's 324,726,578 **+ 9,984** — exactly the 13 added LayerNorm
affines (ln_pre + ln_resid per layer ×6 = 12, plus ln_emb_resid ×1; each 2·384=768).
LUT tables / hyperplanes / unembedder / tok_emb are identical to exp010. Step-1 loss
10.5636 ≈ exp010's own 10.5613 (the anchor-init champion's starting point).

## Comparison
Baseline = exp010 best_val_bpb **1.1940** (324.7M, 3.537h). This run tests whether standard
LayerNorm beats the MeanAbsNorm the champion used. Same seed/data.
