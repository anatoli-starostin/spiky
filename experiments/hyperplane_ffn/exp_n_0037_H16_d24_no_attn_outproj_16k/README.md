# exp_n_0037 — H16/d24, NO attention out-projection, orthoinit + AdamW, tph64/nap6, tied, 16k

Clone of **exp_n_0036**'s train.py with **one architectural change** (option B): the attention output
projection (`out_proj`) is **removed** from `MinimalAttention`. Instead of applying `proj`, the attention
sub-block returns the **concatenation of its per-head outputs directly** — width = `n_head * head_dim` =
6·64 = 384 = `n_embd`, so dimensions match with no projection.

Pipeline becomes: attention heads → concat (width 384) → residual add `x = x + attn(...)` → the existing
`ln2` LayerNorm → the CompressionMHL FFN slot (learned compression matrix intact). Everything downstream is
unchanged; only the attention out-projection is gone.

**Change (verbatim):**
```python
# MinimalAttention.__init__: self.proj = nn.Linear(...) REMOVED
# MinimalAttention.forward:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.transpose(1, 2).contiguous().view(B, T, C)   # concat per-head outputs, no out_proj
```
The `nn.init.zeros_(block.attn.proj.weight)` line in `MinimalGPT.__init__` is also removed (nothing to init).
Confirmed structurally: no `self.proj` / `.proj(` references remain in the attention code (out_proj is *gone*,
not merely zeroed).

**Param count: 26,458,560** (SMOKE-confirmed) = exp_n_0036's 27,343,296 **− 884,736** — exactly 6 layers ×
384×384 (out_proj weight, `bias=False`). The drop lands entirely in the AdamW decay group (2-D weights):
17,891,328 → **17,006,592**; nodecay (LUT tables+temps+1-D) unchanged at 9,451,968. = 1.140× tied dense
(23,209,728). Optimizer print:
`AdamW-everywhere (no Lion) | decay(2-D weights)=17,006,592 wd=0.1 | nodecay(LUT tables+temps+1-D)=9,451,968 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=9,437,184 in nodecay]`.

**exp_n_0036 recipe otherwise identical:** AdamW everywhere (no Lion), no MeanAbsNorm, learnable temperatures
ON, orthogonal per-head init of the compress projection, H16/d24/tph64/nap6, tied, warmup+cosine floor schedule,
grad-clip 1.0, 16k steps, all data/backbone settings.

Runs 16k. **Serial order: 0034 (running) → 0037 → 0036** (per the reorder — 0037 runs BEFORE 0036). exp_n_0034
and exp_n_0036 are NOT modified. Question: does dropping the attention out-projection (letting the FFN slot's
learned compression absorb the head-mixing) hurt, help, or wash — at −0.88M params?
