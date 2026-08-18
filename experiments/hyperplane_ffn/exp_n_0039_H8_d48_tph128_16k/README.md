# exp_n_0039 — single-slot H8/d48, tph128, nap6, tied, 16k

Clone of **exp_n_0033**'s recipe/config/train.py, evolved to a **single-slot H8/d48/tph128** probe: same single
FFN slot `x = x + CompressionMHL(ln2(x))`, but **lut_n_heads=8, inner_in=inner_out=48, tph=128, nap6** (was
H16/d24 in 0033). **H·d = 8·48 = 384 still**, so compress stays Linear(384→384) and decompress Linear(384→384),
and the table budget is set by tph. learnable_temps=true, joint=false, hard forward, no MeanAbsNorm, no Lion,
plain AdamW (0033 grouping), tied. Motivation (from gpustar's exp_g_0006): **H8/d48 matches H16/d24 in loss but
~28% faster** — this tests that wider-d/fewer-heads shape at 2× table budget in a single slot.

**LUT table count = 8 heads · 128 tph · 2⁶ rows · 48 out = 3,145,728 per layer × 6 = 18,874,368**
(exactly 2× 0033's 9,437,184 — same as an H16/d24/tph128 slot, since 8·48 = 16·24 = 384).

**Params = 36,780,384 (SMOKE-confirmed).** Comparison:
- vs exp_n_0033 (27,343,296): **+9,437,088** = +9,437,184 tables (1× 0033's table tensor from tph 64→128) − 96
  temp scalars (8 heads → 96 learnable temps vs 0033's 16 heads → 192).
- vs exp_n_0004 (36,780,288, tph128 UNTIED): **+96** — essentially identical; the +96 is exactly this run's 96
  learnable-temp scalars (exp_n_0004 predates the learnable-temps default, so its temps were fixed buffers, not
  params). Same H8/d48/tph128 table budget.
- = 1.585× tied dense (23,209,728).

Per-component breakdown:
- **LUT tables: 18,874,368** (nodecay) — 2× 0033.
- **Compress+decompress weights: 1,769,472** (2-D, decay) = Linear(384→384)+Linear(384→384) × 6 — IDENTICAL to
  0033 (H·d=384 keeps the projections 384→384; single slot).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,891,328** (identical to
  0033).
- nodecay total = **18,889,056** = tables 18,874,368 + learnable temps 96 (8 heads × 2 × 6) + LayerNorm 1-D
  (ln1+ln2/block + ln_f = 9,984) + proj biases 4,608.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=18,889,056 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=18,874,368 in nodecay]`

**H8/d48 single-slot forward wired:** n_heads=8, inner dim 48 (verified — tables 18,874,368 = 8·128·2⁶·48
confirms 8 heads × 48 out). Single slot, forward unchanged from 0033 apart from the width/tph. Not launched yet
(built + SMOKE-passed only). Question: does the wider-d/fewer-heads H8/d48 shape (gpustar's faster width) at 2×
tables reach lower loss than 0033 (1.228762) in a single slot?
