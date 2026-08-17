# exp_n_0009 — H8/d48/tph128 16k + inner RESIDUAL (skip around the LUT)

Clone of **exp_n_0004** (H8/d48/tph128, 16k, CompressionMHL FFN slot) with the LUT's **inner residual
skip** turned on: `y_h = lut(z_h) + z_h` per head (the LUT learns a DELTA on its compressed input, added
back before decompress). Isolates the residual's effect vs exp_n_0004 (the no-skip twin).

- Uses the existing `CompressionMultiHeadLUT(inner_residual=...)` flag (guarded: requires eff_in==eff_out,
  here 48==48 ✓), wired through the trainer as config `lut_inner_residual` (default False → gpustar's runs
  and all other exps are unaffected; clean/reversible). The skip adds **ZERO parameters**.
- **Params = 36,780,288** — identical to exp_n_0004 (SMOKE-confirmed). Everything else identical: H=8,
  d=48, nap=6, tph=128, tied, gamma0, AdamW two-group, 16000 steps, standard config.

Scheduling: runs SERIALLY on the H100 after the MDN E1 GPU jobs free the card (not concurrent).

Compare final val_bpb to: exp_n_0004 (1.21738, no-skip twin — isolates the residual), exp_n_0002 (1.20823,
best LUT slot so far), tied dense 16k exp073 (1.19665).
