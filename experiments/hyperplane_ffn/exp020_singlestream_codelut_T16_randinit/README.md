# exp020 — exp019 (single-stream code-LUT) with hyperplane_init=random (scale 0.05)

Exact copy of **exp019** (single-stream E=384 champion lineage + HyperplaneCodeLUT
nap15/T16 unembedder, LayerNorm everywhere incl. ln_final before the code-LUT) with
**one change**: every LUT's hyperplane init is **random Gaussian N(0, 0.05²), bias 0**
— applied to BOTH the qk/v/out_proj `HyperplaneMultiHeadLUT` sites AND the
`HyperplaneCodeLUT` unembedder's `hyperplane_weight`/`hyperplane_bias`.

`w_cell` is **unchanged** (near-uniform const 0.02 init).

## Class change
`HyperplaneCodeLUT` originally only did anchor-pairs init. Added a
`hyperplane_init_scale` argument (default `None` → falls back to `initial_weights_noise`,
so backward-compatible): when `hyperplane_init="random"`, the hyperplane rows are drawn
from `N(0, hyperplane_init_scale²)` and bias stays 0 — matching
`HyperplaneMultiHeadLUT`'s scale convention. A CPU test (`test_random_init_scale`)
verifies std≈0.05, bias 0, and that the default stays anchor-pairs (2-sparse).

## Everything else = exp019
6 layers, 6 heads × d_qk=d_v=64, ctx 512, vocab 32768, RoPE 1e4, untied tok_emb; single
residual stream; LUT qk nap4/tph256, v nap6/tph256, out_proj nap7/tph512 (hard); code-LUT
unembedder nap15→V32768, T16; seed 42; 16000 steps, 24,576 tokens/step, LR over 16000
(warmup 1600, cosine to 0.1× floor), Lion lut_lr 2e-4 / AdamW adam_lr 3e-4 (hyperplane w/b
no-wd by name, w_cell wd 0.1, tok_emb/norms wd 0), grad clip 1.0, eval every 500.

## Params / sanity
Total 264,585,236 (identical to exp019 — init doesn't change counts). Step-1 loss 10.4018
≈ ln(32768)=10.397 (near-uniform; slightly closer to uniform than exp019's anchor 10.4157).

## Motivation / baselines
exp012 showed random init ties anchor for the backbone (1.1953 vs 1.1940); random may suit
the code-LUT unembedder better. Baselines: exp010 dual-stream **1.1940**, exp019 anchor-init
single-stream. Same seed/data.
