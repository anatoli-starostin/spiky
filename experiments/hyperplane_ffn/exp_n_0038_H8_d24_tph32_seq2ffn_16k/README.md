# exp_n_0038 — sequential TWO half-head-count CompressionMHL sub-blocks per FFN slot, H8/d24/tph64, nap6, tied, 16k

> **NOTE:** the dir name still says `tph32` (from the first build) but **tph is now 64** in both sub-blocks
> (updated per task). Dir/name kept for path stability; the actual config is H8/d24/**tph64**/nap6.

Clone of **exp_n_0033**'s recipe/config/train.py with **one architectural change**: the single FFN-slot op
`x = x + CompressionMHL(ln2(x))` is replaced by a **sequential two-sub-block** structure inside every
transformer block — two separate CompressionMHL modules stacked, each with its OWN pre-LayerNorm and its OWN
residual add:
```
h   = x + ffn_a(ln2_a(x))     # sub-block 1
out = h + ffn_b(ln2_b(h))     # sub-block 2
```
Attention + ln1 are exactly as 0033; ln_f and everything else unchanged. **Each sub-block: H (lut_n_heads)=8
(half of 0033's 16), d (inner_in=inner_out)=24, tph=64, nap=6.** So each sub-block has HALF the head count but
the SAME tph as 0033 — i.e. two stacked H8/tph64 slots vs one H16/tph64 slot. learnable_temps=true,
joint_head_compression=false, forward_mode=hard; no MeanAbsNorm, no Lion — plain **AdamW everywhere** with the
same grouping as 0033 (2-D→decay wd0.1; LUT tables+temps+1-D→nodecay wd0). Each sub-block's
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

**Params = 27,350,208 (SMOKE-confirmed) — essentially EQUAL to exp_n_0033's 27,343,296, only +6,912 larger.**
Now that tph=64, the total LUT tables EXACTLY equal 0033's, and the whole model is 0033 + the two structural
extras (a 2nd pre-LN per block + one extra sub-block's proj biases). Per-component breakdown:
- **LUT tables: 9,437,184** = 2 sub-blocks × (8 heads · 64 tph · 2⁶ rows · 24 out) = 1,572,864/layer × 6 —
  **exactly EQUAL** to 0033's 9,437,184 (2×H8/tph64 tables = 1×H16/tph64 tables).
- **Compress+decompress weights: 1,769,472** = 2 × (384·192 + 192·384)/layer × 6 — **equal** to 0033's single
  384→384 pair.
- decay(2-D) total = **17,891,328** — identical to 0033 (tok_emb 12,582,912 + attn qkv/proj 3,538,944 + the
  1,769,472 projections).
- nodecay total = **9,458,880** = tables 9,437,184 + temps 192 + LN 1-D (ln1+ln2_a+ln2_b/block + ln_f = 14,592)
  + proj biases 6,912.
- The **+6,912 over 0033** = extra `ln2_b` LayerNorm (768/block × 6 = 4,608) + extra sub-block proj biases
  (2,304 more than 0033's single slot). = 1.178× tied dense (same as 0033).

**Optimizer print (SMOKE):**
`AdamW (0033 grouping) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=9,458,880 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=9,437,184 in nodecay]`

Not launched yet (built + SMOKE-passed only). Clean depth-vs-width A/B at ~equal params & EQUAL total tables:
does splitting one H16/tph64 slot into two stacked H8/tph64 slots (each with its own norm+residual) beat
exp_n_0033's single slot (1.228762)?
