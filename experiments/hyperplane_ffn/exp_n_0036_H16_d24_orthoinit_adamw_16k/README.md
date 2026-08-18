# exp_n_0036 — H16/d24, orthogonal head-init + AdamW-everywhere, tph64/nap6, tied, 16k

Clone of **exp_n_0035**'s train.py with exactly **three changes**, making this the new **AdamW baseline recipe**
(and isolating orthogonal head-init as the single new ingredient). Everything else — H16/d24, tph64, nap6, tied,
vanilla exp073 backbone, std0.02 init, warmup+cosine floor schedule, grad-clip 1.0, 16k steps, all data/backbone
settings — is identical to exp_n_0035.

**Change 1 — AdamW EVERYWHERE (no Lion).** The Lion hybrid is removed entirely; a single standard AdamW drives
all params: lr 3e-4, betas (0.9,0.95), eps 1e-8, with the usual grouping (established by exp_n_0033's
single-AdamW baseline) — 2-D weights → decay (wd 0.1); LUT table tensors + the learnable temps + all 1-D params
→ nodecay (wd 0). Learnable temperatures stay ON (FastMultiHeadLut default). Optimizer print:
`AdamW-everywhere (no Lion) | decay(2-D weights)=17,891,328 wd=0.1 | nodecay(LUT tables+temps+1-D)=9,451,968 wd=0 | lr=0.0003 betas=(0.9, 0.95) eps=1e-8 [LUT tables=9,437,184 in nodecay]`.

**Change 2 — no MeanAbsNorm.** `lut_pre_meanabsnorm=False`: there is no MeanAbsNorm after compression. The
backbone's existing `LayerNorm` before CompressionMHL (the block pre-norm) stays.

**Change 3 — orthogonal per-head init of the compression projection (the one new ingredient).** The independent
per-head compress `nn.Linear(384 → 16·24)` has weight `[16·24, 384]`; viewed per head as `[16, 24, 384]`, each
head's `[24, 384]` row-block is initialised with `nn.init.orthogonal_` so the head's 24 output directions start
orthonormal in the 384-dim input space. Verified: worst `max|BBᵀ − I₂₄|` = 8.9e-07 over all 16 heads, unit
row-norms. Code:
```python
@torch.no_grad()
def _ortho_init_compress_heads(ffn):
    if not getattr(ffn, 'has_compress', False):
        return
    W = ffn.compress.weight                       # [n_heads*inner_in, input_dim]
    if ffn.joint_head_compression:
        torch.nn.init.orthogonal_(W); return
    H, din, cin = ffn.n_heads, ffn.inner_in_dim, W.shape[1]
    Wv = W.view(H, din, cin)                       # [n_heads, inner_in, input_dim]
    for h in range(H):
        q = torch.empty(din, cin, device=W.device, dtype=W.dtype)
        torch.nn.init.orthogonal_(q)               # orthonormal rows (din < input_dim)
        Wv[h].copy_(q)
```
Gated behind `compress_ortho_init` (config; default False), applied in `MinimalGPT.__init__` after the standard
std0.02 init and the output-projection zeroing.

**Params = 27,343,296 (SMOKE-confirmed)** — identical to exp_n_0035/0034 (orthogonal init changes weight values,
not counts; removing Lion/MeanAbsNorm is param-free). = 1.178× tied dense (23,209,728).

Runs 16k, **serial after exp_n_0034** (order 0033 done → 0035 (1.231325) → 0034 → 0036). exp_n_0034 stays
apples-to-apples with exp_n_0035 (Lion + MeanAbsNorm) — untouched. Question: does orthonormal head-init help a
clean AdamW baseline where the Lion+MeanAbsNorm best-practice pairing (exp_n_0035) did not (it regressed to
1.231325 vs plain-AdamW exp_n_0033's 1.228762)?
