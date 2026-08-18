# exp_n_0039 — single-slot capacity probe: H16/d24, tph256, nap6, tied, 16k

Clone of **exp_n_0033**'s recipe/config/train.py with **ONE knob changed: tph (lut_tables_per_head) 64 → 256**
(4× the tables). This is a **single-slot capacity reachability probe** (NOT stacked) — the FFN slot is exactly
0033's single `x = x + CompressionMHL(ln2(x))`, same H16 (lut_n_heads=16), same d24 (inner_in=inner_out=24),
same nap6, learnable_temps=true, joint=false, hard forward, no MeanAbsNorm, no Lion, plain AdamW with the 0033
grouping (2-D→decay wd0.1; LUT tables+temps+1-D→nodecay wd0). **The single-slot forward is unchanged from 0033
— only tph differs.** Purpose: how far can a single wide LUT slot reach if you just throw 4× the table capacity
at it (does raw table count close the gap to dense, or does it saturate)?

**LUT table count = 16 heads · 256 tph · 2⁶ rows · 24 out = 6,291,456 per layer × 6 layers = 37,748,736**
(exactly 4× 0033's 9,437,184).

**Params = 55,654,848 (SMOKE-confirmed).** Comparison:
- vs exp_n_0033 (27,343,296): **+28,311,552** — the entire increase is the extra tables (= 3× 0033's table
  tensor: 4×−1× of 9,437,184).
- vs exp_n_0004 (36,780,288, H8/d48/tph128): **+18,874,560** — 0039 is much heavier (this is a deliberately
  over-provisioned capacity probe).
- = **2.398× tied dense** (23,209,728) — by far the largest LUT FFN in the sweep.

Per-component breakdown:
- **LUT tables: 37,748,736** (nodecay) — 4× 0033.
- **Compress+decompress weights: 1,769,472** (2-D, decay) = Linear(384→384)+Linear(384→384) × 6 — IDENTICAL to
  0033 (single slot, unchanged projections).
- **tok_emb (tied): 12,582,912** (2-D, decay). **attn qkv+proj: 3,538,944** (2-D, decay).
- decay(2-D) total = **17,891,328** — identical to 0033.
- nodecay total = **37,763,520** = tables 37,748,736 + learnable temps 192 + LayerNorm 1-D (ln1+ln2/block + ln_f
  = 9,984) + proj biases 4,608.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=37,763,520 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=37,748,736 in nodecay]`

Not launched yet (built + SMOKE-passed only). Question vs exp_n_0033 (1.228762) and exp_n_0038 (seq-2, 1.2259986,
depth-win at equal tables): does 4× raw table capacity in ONE slot reach lower loss, or does a single wide slot
saturate (making depth, not raw capacity, the real lever)?
