# exp017 — HyperplaneCodeLUT (T=32) unembedder, full 16k

Full 16000-step run of the code-scoring hyperplane unembedder
(`src/spiky/lutorch/hyperplane_code_lut.py`) at **T=32**, on the exp016 dim-96
single-stream backbone.

## Architecture
- Backbone identical to exp016: dim **E=D=96**, 6 layers, **6 heads × head_dim 16**,
  context 512, vocab 32768, RoPE 1e4, **untied** embeddings, single residual stream.
  LUT projections: qk nap4/tph64, v nap6/tph64, out_proj nap6/tph256 (out_proj doubles
  as unify-heads + FFN). No residual_lut / emb_resid / dual-stream. MeanAbsNorm
  pre-norms + LayerNorm on q/k/final.
- **Unembedder = HyperplaneCodeLUT**: nap=15 → V=2^15=32768 logits, **n_tables T=32**,
  w_cell_init 0.02, anchor-pairs hyperplane init, T_soft 0.5. Maps the 96-dim hidden
  directly to logits via per-code soft-sign scores gated by `w_cell`, voted over 32
  tables — no stored per-cell rows.

## Params (smoke-verified)
- **Total 20,346,436.** Unembedder **1,095,136** (~1.10M) = hyperplanes T·nap·(E+1)=46,560
  + w_cell T·V=1,048,576. (vs the best LUT unembedder's 67M at equal/worse quality.)
- Groups: LUT tables 12,976,128 (Lion) · all hyperplane w/b 3,175,392 (AdamW no-wd) ·
  w_cell 1,048,576 (AdamW wd 0.1) · tok_emb 3,145,728 · norms 612.
- Step-1 loss 10.4225 ≈ ln(32768) (near-uniform init).

## Recipe
16000 steps · device_batch 24 × seq 512 × grad_accum 2 = 48 seq / 24,576 tokens per step ·
LR schedule over the full 16000 (warmup 0.1 = 1600 steps, cosine to 0.1× floor) · Lion
lut_lr 2e-4 (LUT tables) · AdamW adam_lr 3e-4 (hyperplanes no-wd by name; w_cell wd 0.1;
tok_emb/norms wd 0) · grad clip 1.0 · bf16 tables + fp32 master · eval every 500.

## Motivation
The 2000-step T-sweep found **T=32 the quality/compute sweet spot**: it beat the best
LUT-unembedder (val_bpb 1.8942 @ 67M unemb params) at only ~1.1M unemb params, and
quality kept improving with T (T16 1.9633 → T32 1.8846 → T64 1.8194). This is the full
16k run to see where T=32 lands against the full-scale marks (exp010 1.1940 dense-LUT
scale is a different/larger model; exp016 LUT-unembedder runs are the direct baseline).
