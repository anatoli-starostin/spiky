---
name: project_hybrid_qkv_soft_localization
description: "The soft/magnitude-leakage win is qkv-localizable: exp507 (only qkv soft, rest argmax) = 1.4797 matches/beats all-soft exp493 (1.4806) at far lower inference cost. Soft v_branch worth ~0.004; V2D->Linear qkv (no LUT) trails the lookup by ~0.055."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Hybrid LUT-LM line (2026-05-23), all bs=16, 8000 steps, E=64/6L.** Reference points:
exp475 (all argmax TinyMHLut, all LION) = **1.4962**; exp493 (all soft MatmulMHLut, all
AdamW) = **1.4806**. Both 89.4M, param-matched.

**exp507 — the soft win is QKV-LOCALIZABLE.** Fork of exp475 with ONLY qkv swapped to the
exp493 soft `MatmulMultiHeadLut(softmax)`; v_lut/out_proj/residual_lut stay TinyMHLut(soft
argmax). **Optimizer split is load-bearing** (both module types have ndim-3 `.weights`, so
split by NAME): qkv dense grads -> AdamW wd=0.1 (like exp493); the 3 argmax modules ->
LION (like exp475; LION's edge is sparse-gradient-specific, useless for the dense soft qkv).
**Result 1.4797 — matches/slightly beats all-soft exp493 (1.4806)**, by making just ONE of
4 modules soft (qkv = 21.2M of 89.4M params), keeping the other 3 as matmul-free argmax LUTs.
The edge over exp493 is likely the hybrid optimizer (LION native on the sparse modules vs
exp493 forcing all onto AdamW). So: nearly all the magnitude-leakage benefit lives in the
**qkv (attention-input) projection**, and the hybrid is much cheaper to deploy than all-soft.

**exp508 — soft v_branch worth ~0.004 (capacity-matched).** In exp507 the soft qkv emitted
`2*d_qk + d_v`; the last d_v (the v_branch) was ADDED to v_lut -> v got a soft, magnitude-
leaking contribution. exp508 removes it (qk_lut = pure `2*d_qk`) and bumps v_lut tph 256->320
— a CAPACITY-MATCHED swap (both moves = exactly 393,216 params; total stays 89,393,456). So
exp508 vs exp507 is "soft shared-anchor v (leakage) vs equal hard dedicated v-tables", NOT a
removal. **exp508 = 1.4840, +0.0043 vs exp507** (gap ~0 mid-run, grew steadily late — the
recurring crossover). So the magnitude leakage in the v path is worth a small but real ~0.004
over equivalent hard capacity. (A pure-removal run with v_tph unchanged would ADD a capacity
confound — exp508's matched design is the right comparison.)

**exp509/exp510 — a Linear readout of dominance features does NOT replace the lookup.** Replace
qkv entirely with `VectorToDominance(E=64, smooth) -> [LayerNorm] -> Linear -> H*(2*d_qk+d_v)`
(no LUT; emits q,k,v; v_lut removed). Same soft-sign dominance features the LUT uses as its
index (all 2016 pairs), but a dense linear readout instead of a 2^NAP table lookup.
- exp509 (raw V2D->Linear): trailed exp507 by **+0.090** @ step 1000, gap widening — poorly
  conditioned (2016 unnormalized dominance feats into a Linear).
- exp510 (+LayerNorm after V2D, + small-std Linear init 0.005): recovered ~0.027 of that
  (LN/init fixed conditioning) but STILL trailed exp507 by ~**+0.055**, closing only very
  slowly. Both ~69M params (leaner, no LUT).
Conclusion: even well-conditioned, a linear map of dominance features doesn't match the LUT —
the table's per-row nonlinear lookup does real work beyond a linear readout of the same signs.

`QKVDominanceLinear` lives in exp509/exp510 train.py; uses `VectorToDominance` from
`spiky.lutorch.ranking_tools`. Relates to [[project_magnitude_leakage_softmax_package]]
(why qkv-soft beats argmax = magnitude leakage; not hardenable) and the exp475/exp493 line.
