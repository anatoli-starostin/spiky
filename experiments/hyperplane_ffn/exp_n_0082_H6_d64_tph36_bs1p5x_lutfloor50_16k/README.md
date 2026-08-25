# exp_n_0082 — decoupled LR floor: LUT tables anneal to 50% (not ~10%)

Follow-up to exp_n_0081 testing the "LUTs are more sensitive to LR annealing" hypothesis
(the exp_n_0081−exp073 gap opened in lockstep with the LR anneal, r=0.979 in the tail).
**Exact clone of exp_n_0081** (23M LUT, exp074 arch H6/inner64/tph36, batched path, 1.5×
batch = device_batch 72 / 36,864 tok/step / grad_accum 1, 16k steps, peak lr 3e-4 cosine,
10% warmup, wd 0.1, seed 1, AdamW, learnable temps). **Only change: a decoupled LR floor
for the LUT-table parameter group.**

**Decoupled schedule** (implemented in `setup_optimizer` + `get_lr_scale`):
- **LUT-table weights** (FastMHL `.weights`, 5,308,416 params) get their OWN AdamW group,
  cosine-annealed to floor **0.5 → min_lr 1.5e-4** (50% of peak).
- **Everything else** (2-D decay weights + temps + 1-D biases/norms) keeps exp_n_0081's
  floor **0.1 → min_lr 3.0e-5**.
- Warmup (to step 1600) is identical for all groups (both reach peak 3e-4).

Verified schedule (peak 3e-4): step 1600 base=lut=3.0e-4; step 8000 base 1.88e-4 / lut
2.38e-4; step 12000 base 7.8e-5 / lut 1.77e-4; **step 16000 base 3.0e-5 / lut 1.5e-4 (5×)**.
So the LUT tables retain a substantial LR exactly through the tail where exp_n_0081's gap
opened. Param count **23,214,348** (identical to exp_n_0081). Per-step log prints both
`lr_base` and `lr_lut`.

**Question:** does keeping the LUT tables "hotter" through the anneal let this run beat
exp_n_0081 (1.20542, LUT annealed to ~0.1×) and approach the 23M dense baseline exp073
(1.19665)? If it closes the gap, that confirms the LUT-anneal-sensitivity story and gives a
free recipe win.

Outputs: metrics.csv, summary.json, loss.png, checkpoint.pt.
