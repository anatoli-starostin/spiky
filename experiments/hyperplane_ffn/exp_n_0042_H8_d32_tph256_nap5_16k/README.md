# exp_n_0042 — single-slot H8/d32, tph256, nap5, tied, 16k

Clone of **exp_n_0040_H8_d48_tph256_16k** with **TWO changes: inner d (both in/out) 48 → 32 AND nap 6 → 5**
(H stays 8, tph 256). Same as exp_n_0041 but with **nap 5 (2⁵=32 clusters per table instead of 2⁶=64)** —
halves the routing rows. Single-slot CompressionMHL, tied, learnable_temps=true, joint=false, hard, no
MeanAbsNorm/Lion, plain AdamW (0033 grouping). H·d = 8·32 = 256, so compress = Linear(384→256),
decompress = Linear(256→384) (same projections as 0041). Probes narrower-d + coarser routing.

**LUT table count = 8 heads · 256 tph · 2⁵ rows · 32 out = 2,097,152/layer × 6 = 12,582,912.**
- vs exp_n_0033 (9,437,184): **1.333×** (4/3).
- vs exp_n_0040 (37,748,736): **0.333×** (1/3).
- = **half of exp_n_0041's** 25,165,824 (nap 6→5 halves the rows).

**Params = 29,898,336 (SMOKE-confirmed).** Per-component:
- **LUT tables: 12,582,912** (nodecay).
- **Compress+decompress weights: 1,179,648** (2-D, decay) = Linear(384→256)+Linear(256→384) × 6 — same as 0041,
  ⅔ of exp_n_0040's 1,769,472.
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,301,504** (identical to
  0041).
- nodecay total = **12,596,832** = tables 12,582,912 + temps 96 + LayerNorm 1-D 9,984 + proj biases 3,840.
- = 1.288× tied dense.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,301,504 wd=0.1 | nodecay(LUT tables+temps+1-D)=12,596,832 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=12,582,912 in nodecay]`

Differs from exp_n_0040 config in exactly four keys: `lut_inner_in_dim`, `lut_inner_out_dim` (48→32),
`lut_n_anchor_pairs` (6→5), and `exp_name`. Not launched (built + SMOKE-passed only); queued serially after
exp_n_0040/0041.
