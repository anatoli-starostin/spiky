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
train/loss (smoothed), val/bpb at 250 and 500, CORE-500, tok/s and step time (the 48x48 eigh x 24 layers adds overhead), vs AdamW repro500. Check no NaN/instability.

## COMPARISON (steps 0-500; whiten500 = wandb `d24-lutffn-confn1-1xh100-s0-whiten500-c57246a7`, commit `3627c60a`, vs AdamW `...-v2repro500-2959ed04` and reference `...-full-v2-9811cced`)

Data: wandb histories (`train/loss`, `val/bpb`, `core_metric`, `train/dt`, `train/tok_per_sec`), `compare_whiten500_vs_adamw_first500.csv`, `compare_whiten500_vs_adamw_first500.png`. The run finished normally (exit 0, no NaN, no loss spikes), same seed/data order as the AdamW runs.

| metric | v2 ref (AdamW) | repro500 (AdamW) | whiten500 | whiten - repro500 |
|---|---|---|---|---|
| train loss, mean steps 1-10 | 9.838 | 9.841 | 9.891 | +0.049 |
| train loss, mean steps 91-100 | 4.635 | 4.667 | 4.622 | -0.045 |
| train loss, mean steps 241-250 | 3.633 | 3.510 | 3.504 | -0.005 |
| train loss, mean steps 491-500 | 3.188 | 3.132 | 3.129 | -0.003 |
| val bpb, step 0 | 3.1808 | 3.1714 | 3.1714 | 0 |
| val bpb, step 250 | 1.0814 | 1.0560 | 1.0538 | -0.0022 |
| val bpb, step 500 | 0.9662 | 0.9603 | 0.9588 | -0.0015 |
| CORE, step 500 (in-training, 500-per-task subsample) | 0.0988 | 0.0973 | 0.0919 | -0.0054 |
| s/step, mean steps 11-500 | 14.50 | 14.21 | 14.49 | +0.28 (+2%) |
| tok/s, mean steps 11-500 | 72,308 | 73,784 | 72,363 | -2% |
| peak memory (allocated / reserved, MiB) | 64,377 / 65,458 | 64,378 / 65,460 | 63,225 / 65,460 | -1.15 GiB allocated |

Smoothed (10-step mean) loss, whiten500 minus repro500, by step range: 11-30 +0.016; 31-60 -0.020; 61-100 -0.047; 101-200 -0.047; 201-300 -0.011; 301-400 -0.009; 401-500 -0.005. Against the v2 reference: 61-100 -0.031, 101-200 -0.055, 201-300 -0.184 (the reference's step ~200 spike, shared by neither new run), 301-500 about -0.02.

**First divergence.** The 10-step smoothed loss differs from repro500 by more than 0.01 already at step 10 (+0.049 over steps 1-10, peaking +0.064 at step 15, i.e. whiten starts slightly slower in warmup) and by more than 0.05 at step 11. The sign flips by step ~30; whiten is lower by 0.02-0.05 over steps 30-200, and the gap shrinks to -0.005 by steps 400-500.

**Noise scale.** Per-step loss scatter around its local trend is ~0.015 per run (steps 300-500); repro500 vs the v2 reference differ by 0.03-0.12 (windows) and 0.006-0.025 bpb for identical recipes (see `../d24-lutffn-confn1-1xh100-s0-v2repro500/README.md`: no seed replicate exists, so true run-to-run noise is unmeasured).

## Verdict (honest)
**No evidence that the Muon-analogue table optimizer helps over AdamW in the first 500 steps; at best a tie.** Final-window loss is -0.003 (steps 491-500) and val bpb -0.0015 at step 500 / -0.0022 at step 250 relative to repro500, all well inside the repro500-vs-v2 discrepancy (0.006-0.025 bpb, up to 0.12 loss), so none of this is distinguishable from run-to-run noise with one run per arm. The only visible differences are early (steps 30-200, whiten ~0.02-0.05 lower loss, after a slightly slower first ~20 steps) and they close to ~0 by step 500; a transient early advantage that vanishes is the opposite of the ~0.05 bpb persistent gain that dense Muon showed. In-training CORE-500 is lower (0.0919 vs 0.0973), but that is a subsample metric with unknown noise (repro vs ref differ by 0.0015 there), so I would not call it a regression either. Costs: +2% s/step (the 48x48 eigh over 24 layers is cheap), memory not higher. Caveats: single seed, RMS 0.3 / beta1 0.9 / eps 1e-3 untuned (a different RMS could change the effective LR; the early-phase behaviour hints it may be in a sensible range but this was not tested). Stability is fine (no NaN, smoothed loss monotone-ish, largest 10-step-mean increase after step 30 is 0.007). Worth continuing to 5568 steps only if one wants to see whether the early lead reopens at lower LR in the decay phase; checkpoints for that exist.

## Checkpoints (worker only, not committed)
`/home/lab-ds-h100-anatoly/work/nanochat_base/base_checkpoints/d24_lutffn_confn1_1xh100_whiten500/` (symlink to `results/d24-lutffn-confn1-1xh100-s0-whiten500/checkpoints/` in the clone): `model_000250.pt`, `optim_000250_rank0.pt`, `meta_000250.json`, `model_000500.pt`, `optim_000500_rank0.pt`, `meta_000500.json` (3.9 GB model + 5.6 GB optim each).
