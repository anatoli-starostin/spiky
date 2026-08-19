# exp_n_0043 — single-slot H8/d48, tph128, nap8, tied, 16k

Clone of **exp_n_0039_H8_d48_tph128_16k** with **ONE knob changed: nap (lut_n_anchor_pairs) 6 → 8** (2⁸ = 256
rows per table instead of 2⁶ = 64 — 4× the routing rows). Single-slot CompressionMHL, H8, d48/48, tph128, tied,
learnable_temps=true, joint=false, hard forward, no MeanAbsNorm/Lion, plain AdamW (0033 grouping). Everything
except nap (and exp_name) is identical to 0039. Probes finer routing resolution (more clusters per table) at the
same head count / d / tph — i.e. spend capacity on routing granularity instead of table multiplicity.

**LUT table count = 8 heads · 128 tph · 2⁸ rows · 48 out = 12,582,912/layer × 6 = 75,497,472** (= **4×
exp_n_0039's** 18,874,368, since nap 6→8 = 2² = 4× rows).

**Params = 93,403,488 (SMOKE-confirmed).** Comparison:
- vs exp_n_0039 (36,780,384): **+56,623,104** — exactly 3× 0039's table tensor (4×−1× of 18,874,368);
  everything else identical.
- = **4.024× tied dense** (23,209,728) — the heaviest LUT FFN in the whole sweep by a wide margin.

Per-component breakdown:
- **LUT tables: 75,497,472** (nodecay) — 4× 0039.
- **Compress+decompress weights: 1,769,472** (2-D, decay) — UNCHANGED from 0039 (H·d = 8·48 = 384 keeps
  projections 384→384; nap doesn't touch projections).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,891,328** (identical to
  0039).
- nodecay total = **75,512,160** = tables 75,497,472 + learnable temps 96 + LayerNorm 1-D 9,984 + proj biases
  4,608.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=75,512,160 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=75,497,472 in nodecay]`

Differs from exp_n_0039 config in EXACTLY two keys: `lut_n_anchor_pairs` (6→8) and `exp_name`. Not launched yet
(built + SMOKE-passed only); third queued run after exp_n_0041 / exp_n_0042. Question: does 4× finer routing
resolution (nap 8) beat exp_n_0039's tph-doubling / exp_n_0040's capacity — i.e. is routing granularity a
stronger lever than raw table multiplicity?
