# exp_n_0034 — H16/d24, nap5/tph128, MeanAbsNorm + Lion-tables-only, tied, 16k

**Now on the exact exp_n_0035 best-practice recipe** (re-engineered from its original plain-exp_n_0033 clone):
train.py is copied wholesale from exp_n_0035 (the version with the Lion-tables-only fix + MeanAbsNorm), and the
config differs from exp_n_0035 in **only two knobs: `lut_n_anchor_pairs` and `lut_tables_per_head`**. So
**exp_n_0034 vs exp_n_0035 is a clean single-variable A/B** on the nap/tph routing-vs-multiplicity trade at
fixed table budget.

**The one difference vs exp_n_0035:** nap **5** (2⁵=32 clusters/table) + tph **128**, vs exp_n_0035's nap **6**
(2⁶=64) + tph **64**. The product 2^nap·tph = 32·128 = 4096 = 64·64 is held fixed, so the table tensor size —
and the total param count — are identical. Trades routing RESOLUTION (fewer clusters) for table MULTIPLICITY
(2× tables/head).

**Recipe shared with exp_n_0035 (identical):**
- **MeanAbsNorm before the router** (`lut_pre_meanabsnorm=True`): param-free `z_h/(z_h.abs().mean(-1)+1e-6)`,
  eps 1e-6, on each head's compressed router input, right before FastMHL.
- **Learnable temperatures ON** (`lut_learnable_temps=True`), but the 0-dim log-temp scalars are on **AdamW**,
  not Lion (Lion-tables-only grouping `lut_ids = {id(m.weights) for m in … FastMultiHeadLut}`).
- **Hybrid optimizer:** Lion on LUT table tensors only (lr 2e-4, betas (0.9,0.95), wd 0) + AdamW on the rest
  (lr 3e-4, betas (0.9,0.95), eps 1e-8; 2-D decay wd 0.1, 1-D/0-D nodecay wd 0 incl. the temps).
- Same warmup+cosine floor schedule, grad-clip 1.0, std0.02 compress init, tied unembedder, H16/d24, 16k steps.

**Params = 27,343,296 (SMOKE-confirmed)** — identical to exp_n_0035 (table budget held fixed); optimizer print:
`Hybrid optimizer | Lion(LUT tables only)=9,437,184 lr=0.0002 betas=(0.9, 0.95) wd=0 | AdamW(rest incl. temps) lr=0.0003 decay_wd=0.1`.
= 1.178× tied dense (23,209,728).

Runs 16k, **serial after exp_n_0035** (order 0033 done → 0035 → 0034). Compare directly to exp_n_0035
(nap6/tph64, same recipe) to isolate finer routing vs more table-mixing at fixed budget, now both on the
historical best-practice recipe.
