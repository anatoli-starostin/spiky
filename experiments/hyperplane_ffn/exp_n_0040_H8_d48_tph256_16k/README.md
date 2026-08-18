# exp_n_0040 — single-slot H8/d48, tph256, nap6, tied, 16k

Clone of **exp_n_0039_H8_d48_tph128_16k** with **ONE knob changed: tph (lut_tables_per_head) 128 → 256** (2× the
tables). Single-slot CompressionMHL FFN (`x = x + CompressionMHL(ln2(x))`), H8 (lut_n_heads=8), d48
(inner_in=inner_out=48), nap6, tie_unembedder=true, lut_learnable_temps=true, joint=false, hard forward, no
MeanAbsNorm, no Lion, plain AdamW (0033 grouping). Everything except tph (and exp_name) is identical to 0039.
Capacity probe on the wider-d H8/d48 shape: does 4× 0033's tables (2× 0039's) in a single H8/d48 slot reach lower?

**LUT table count = 8 heads · 256 tph · 2⁶ rows · 48 out = 6,291,456 per layer × 6 = 37,748,736**
(= 4× exp_n_0033's 9,437,184, = 2× exp_n_0039's 18,874,368).

**Params = 55,654,752 (SMOKE-confirmed).** Comparison:
- vs exp_n_0039 (36,780,384): **+18,874,368** — exactly one more 0039-tables-worth (the tph 128→256 doubling);
  everything else identical.
- vs exp_n_0033 (27,343,296): **+28,311,456** (all extra tables, −96 temps for H8 vs H16).
- vs exp_n_0004 (36,780,288, fixed-temp H8/d48/tph128): +18,874,464.
- = **2.398× tied dense** (23,209,728) — ties exp_n_0039's first-build (H16/d24/tph256) as the heaviest LUT FFN.

Per-component breakdown:
- **LUT tables: 37,748,736** (nodecay) — 4× 0033 / 2× 0039.
- **Compress+decompress weights: 1,769,472** (2-D, decay) — UNCHANGED from 0039/0033 (H·d=384 keeps projections
  384→384).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,891,328** (identical to
  0033/0039).
- nodecay total = **37,763,424** = tables 37,748,736 + learnable temps 96 (8 heads × 2 × 6) + LayerNorm 1-D
  (ln1+ln2/block + ln_f = 9,984) + proj biases 4,608.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=37,763,424 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=37,748,736 in nodecay]`

Differs from exp_n_0039 config in EXACTLY two keys: `lut_tables_per_head` (128→256) and `exp_name`. Not launched
yet (built + SMOKE-passed only).
