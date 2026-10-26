# d24-lutffn-confn1-1xh100-s0-whiten500

**Question.** Does a Muon analogue for the LUT tables (instead of AdamW) train faster than the AdamW baseline over the first 500 steps of the v2 recipe?

**Motivation.** Dense Muon beats AdamW by ~0.05 bpb at equal steps (`../d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k/`). LUT tables always use AdamW. `../lut-grad-spectra-analysis/` shows the table gradients are very lopsided in the d=48 width direction (per-table top-1 energy 0.70-0.85, effective rank 2-3 of 48; the dominant direction is shared by the 64 tables of a head, unrelated across heads) and that a one-step lookahead on held-out tokens gains ~10-20% over Adam with "row-normalisation + whitening". A table read is linear in the cell vector, so there is nothing to orthogonalise across cells; the analogue works on the width side only.

## Optimizer (`nanochat/nanochat/optim.py::lut_whiten_step`, flag `--lut-table-optimizer=whiten`)
Per table parameter `[G=16 heads, tph=64, K=256 cells, d=48]`:
1. momentum `m = beta1 m + (1-beta1) g` (beta1=0.9, bias-corrected);
2. whitening per head: `U = m (m^T m + eps I)^(-1/2)`, rows pooled over the head's 64x256 cells, `eps = 1e-3 x mean eigenvalue` (48x48 eigh);
3. per-cell row normalisation to a fixed per-element RMS (`--lut-whiten-rms`, default 0.3; unhit cells have m=0 and stay put);
4. `p -= lr * U`; lr = the same table LR schedule as the baseline (3e-3 x 1/sqrt(dmodel/768), WSD), no weight decay.
Only the 4-D tables use it; confidence scalars, projections and biases stay AdamW. Cell-TV (lambda=10) and table dropout 0.2 are unchanged. State (`exp_avg`, `step`) is part of `optimizer.state_dict()`, so checkpoints resume. Single-rank only (asserted).

**Untuned choices (one shot):** RMS 0.3, beta1 0.9, eps 1e-3. The RMS sets the effective LR relative to Adam and was not calibrated; a flat or diverging curve may mean mis-scaling rather than a bad idea. The analysis evaluated directions at matched step norm, not the training dynamics.

## Setup
Identical to `../d24-lutffn-confn1-1xh100-s0-v2repro500/` (v2 recipe, `--no-lr-rescale`, `NUM_ITERATIONS=5568` schedule, `--max-steps=501`, seed 42, ClimbMix, fp8, batch 2^20) except the table optimizer. Launcher: `nanochat/runs/lut_ffn_d24_whiten500_1xh100.sh`. wandb: project `spiky-nanochat`, group `nanochat_lut_ffn`, tag `whiten500`; smoke: `SMOKE=1`, tag `smoke`.

**Checkpoints (for continuation).** `SAVE_EVERY=250`, `KEEP_CKPTS=4`: model + optimizer + dataloader state at steps 250 and 500 stay on the worker (not committed). To continue to the full 5568 steps the same launcher resumes from the latest complete checkpoint with `TRAIN_EXTRA="--max-steps=5569"`.

## Baseline for comparison
`../d24-lutffn-confn1-1xh100-s0-v2repro500/` (AdamW, same code, same seed) and the reference `../d24-lutffn-confn1-1xh100-s0-full-v2/`. Note the repro500 README: the two AdamW runs differ from each other by up to ~0.12 in loss around step 200 (reference spike) and 0.006-0.025 in val bpb, so differences of that size are not evidence.

## Acceptance criteria (to be judged in Slack by Anatoly)
train/loss (smoothed), val/bpb at 250 and 500, CORE-500, tok/s and step time (the 48x48 eigh x 24 layers adds overhead), vs AdamW repro500. Check no NaN/instability. Result pending.
