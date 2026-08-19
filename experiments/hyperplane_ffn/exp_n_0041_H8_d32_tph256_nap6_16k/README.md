# exp_n_0041 — single-slot H8/d32, tph256, nap6, tied, 16k

> **RESULT: final_val_bpb = 1.2169448 (best=final; 16k, 1.96 h). Narrowing d 48→32 HURTS: +0.0127 vs exp_n_0040
> (d48, 1.2042656).** Notably 0041 (d32, 42.5M params, 25.2M tables) ≈ exp_n_0039 (d48, 36.8M, 18.9M tables):
> 1.2169448 vs 1.2163933 — nearly identical despite 0041's extra tables/params. So **per-head width d is more
> param-efficient than raw table count** — shrinking d and compensating with tph does not pay. d48 stays the
> better shape.

Clone of **exp_n_0040_H8_d48_tph256_16k** with **ONE change: inner d (both in/out) 48 → 32** (H stays 8, tph
256, nap 6). Single-slot CompressionMHL, tied, learnable_temps=true, joint=false, hard, no MeanAbsNorm/Lion,
plain AdamW (0033 grouping). **H·d = 8·32 = 256 now** (was 384), so the projections shrink: compress becomes
Linear(384→256), decompress Linear(256→384). Probes narrower per-head width at the same head count + table depth.

**LUT table count = 8 heads · 256 tph · 2⁶ rows · 32 out = 4,194,304/layer × 6 = 25,165,824.**
- vs exp_n_0033 (9,437,184): **2.667×** (8/3).
- vs exp_n_0040 (37,748,736): **0.667×** (2/3 — tables scale with inner d: 48→32).

**Params = 42,481,248 (SMOKE-confirmed).** Per-component:
- **LUT tables: 25,165,824** (nodecay).
- **Compress+decompress weights: 1,179,648** (2-D, decay) = Linear(384→256)+Linear(256→384) × 6 — **⅔ of
  exp_n_0040's 1,769,472** (H·d 384→256 shrinks the projections by 589,824).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,301,504** (0040's
  17,891,328 − 589,824 projection shrink).
- nodecay total = **25,179,744** = tables 25,165,824 + temps 96 + LayerNorm 1-D 9,984 + proj biases 3,840.
- = 1.831× tied dense.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,301,504 wd=0.1 | nodecay(LUT tables+temps+1-D)=25,179,744 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=25,165,824 in nodecay]`

Differs from exp_n_0040 config in exactly three keys: `lut_inner_in_dim`, `lut_inner_out_dim` (48→32), and
`exp_name`. Not launched (built + SMOKE-passed only); queued serially after exp_n_0040.
