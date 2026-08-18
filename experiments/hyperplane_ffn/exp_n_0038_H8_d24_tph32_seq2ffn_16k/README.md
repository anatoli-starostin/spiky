# exp_n_0038 — sequential TWO half-size CompressionMHL sub-blocks per FFN slot, H8/d24/tph32, nap6, tied, 16k

Clone of **exp_n_0033**'s recipe/config/train.py with **one architectural change**: the single FFN-slot op
`x = x + CompressionMHL(ln2(x))` is replaced by a **sequential two-sub-block** structure inside every
transformer block — two separate CompressionMHL modules stacked, each with its OWN pre-LayerNorm and its OWN
residual add:
```
h   = x + ffn_a(ln2_a(x))     # sub-block 1
out = h + ffn_b(ln2_b(h))     # sub-block 2
```
Attention + ln1 are exactly as 0033; ln_f and everything else unchanged. **Each sub-block is HALF-SIZE vs
0033: H (lut_n_heads)=8, d (inner_in=inner_out)=24, tph=32; nap=6 unchanged.** learnable_temps=true,
joint_head_compression=false, forward_mode=hard; no MeanAbsNorm, no Lion — plain **AdamW everywhere** with the
same grouping as 0033 (2-D weights→decay wd0.1; LUT tables+temps+1-D→nodecay wd0). Each sub-block's
compress = Linear(384→H·d=192), decompress = Linear(192→384).

**Block forward (verbatim):**
```python
def forward(self, x, cos, sin):
    x = x + self.attn(self.ln1(x), cos, sin)
    if self.ffn_type == 'dense':
        return x + self.mlp(self.ln2(x))
    B, T, C = x.shape
    ha = self.ln2_a(x)                                             # sub-block 1
    out_a = self.ffn_a(ha.reshape(B * T, C)).reshape(B, T, C).to(ha.dtype)
    h = x + out_a
    hb = self.ln2_b(h)                                             # sub-block 2
    out_b = self.ffn_b(hb.reshape(B * T, C)).reshape(B, T, C).to(hb.dtype)
    return h + out_b
```

**Params = 22,631,616 (SMOKE-confirmed) — LOWER than exp_n_0033's 27,343,296 by 4,711,680**, exactly as
expected: tables halve (two half-size sub-blocks), while the two 384→192 projection pairs equal one 384→384
pair; the small extras are a second pre-LN per block + proj biases. Per-component breakdown:
- **LUT tables: 4,718,592** = 2 sub-blocks × (8 heads · 32 tph · 2⁶ rows · 24) = 786,432/layer × 6 — **exactly
  HALF** of 0033's 9,437,184.
- **Compress+decompress weights: 1,769,472** = 2 × (384·192 + 192·384)/layer × 6 — **equal** to 0033's single
  384→384 pair (294,912/layer). (In the AdamW decay group.)
- decay(2-D) total = 17,891,328 (tok_emb 12,582,912 + attn qkv/proj 3,538,944 + the 1,769,472 projections) —
  identical to 0033's decay group.
- nodecay total = 4,740,288 = tables 4,718,592 + temps 192 + LN 1-D (ln1+ln2_a+ln2_b per block + ln_f =
  14,592) + proj biases 6,912.
- = 0.975× tied dense (23,209,728) — the two half-size LUT sub-blocks make this the leanest LUT FFN yet, just
  UNDER dense.

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=4,740,288 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=4,718,592 in nodecay]`

Not launched yet (built + SMOKE-passed only). Question: does depth (two sequential routed sub-blocks) buy more
than width (one big slot) at HALF the table budget — i.e. does stacking beat exp_n_0033 (1.228762) despite 0.5×
the tables?
