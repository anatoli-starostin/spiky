# exp_n_0045 — single-slot H8/d48, tph256, nap7, tied, 16k

> **⭐ MILESTONE RESULT: final_val_bpb = 1.1977670 (best 1.1975260; 16k, 3.28 h). Essentially MATCHES tied
> dense (1.19665) — gap +0.00112 final / +0.00088 best, the closest any LUT FFN has reached.** Reachability =
> YES: the CompressionMHL slot CAN hit dense quality, but at ~4× dense params (93.4M) and ~2.35× slower
> wall-clock. **Table-multiplicity A/B: beats equal-budget exp_n_0043 (tph128/nap8, 1.2029199) by −0.00515 — a
> CLEAR win** (mid-run ~0.0035 lead grew to −0.0052). So at the big 75.5M-table budget, table MULTIPLICITY
> (more tph, coarser routing) beats routing RESOLUTION (higher nap) — OPPOSITE of the small-scale 0035>0034
> finding; the lever flips with scale. New campaign best.

Clone of **exp_n_0043_H8_d48_tph128_nap8_16k** with **two knobs changed at FIXED table budget: tph 128 → 256
AND nap 8 → 7**. Because the table budget is `H · tph · 2^nap · d`, doubling tph (128→256) exactly cancels
halving the rows (nap 8→7, 2⁸→2⁷), so the **total tables and param count are IDENTICAL to 0043**. Single-slot
CompressionMHL, H8, d48/48, hard forward, tied, learnable_temps=true, joint=false, no MeanAbsNorm/Lion, plain
AdamW (0033 grouping). This isolates **table MULTIPLICITY (more tph) vs routing RESOLUTION (higher nap rows)**
at the same capacity — the same nap/tph trade tested small in exp_n_0034/0035, now at this large budget.

**LUT table count = 8 heads · 256 tph · 2⁷ rows · 48 out = 12,582,912/layer × 6 = 75,497,472** — **EXACTLY
equal to exp_n_0043's 75,497,472** (8·256·2⁷ = 8·128·2⁸ = 8·32768). ✓ Budget invariant confirmed.

**Params = 93,403,488 (SMOKE-confirmed) — identical to exp_n_0043.** Per-component:
- **LUT tables: 75,497,472** (nodecay) — same as 0043.
- **Compress+decompress weights: 1,769,472** (2-D, decay) — UNCHANGED (H·d = 8·48 = 384).
- **tok_emb (tied): 12,582,912** + **attn qkv/proj: 3,538,944** → decay(2-D) total **17,891,328**.
- nodecay total = **75,512,160** = tables 75,497,472 + temps 96 + LayerNorm 1-D 9,984 + proj biases 4,608.
- = 4.024× tied dense (same as 0043).

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=75,512,160 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=75,497,472 in nodecay]`

Differs from exp_n_0043 config in EXACTLY three keys: `lut_tables_per_head` (128→256), `lut_n_anchor_pairs`
(8→7), `exp_name`. Not launched yet (built + SMOKE-passed only); queued after exp_n_0044. Question: at the same
75.5M-table budget, does splitting into more tables (tph256) with coarser routing (nap7) beat exp_n_0043's fewer
tables + finer routing (nap8, 1.202920)? (Small-scale exp_n_0035/0034 said finer routing wins — does that hold
at scale?)
