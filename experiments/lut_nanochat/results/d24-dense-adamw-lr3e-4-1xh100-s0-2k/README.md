# d24-dense-adamw-lr3e-4-1xh100-s0-2k

**Question.** How much does Muon deliver in the dense (non-LUT) d24 model? Fork of `d24-dense-1xh100-s0` with **AdamW instead of Muon for the transformer matrices**, run for the first 2000 steps of the full 5,568-step schedule.

**Reference.** `d24-dense-1xh100-s0` (wandb `spiky-nanochat`, group `nanochat_baseline`, run `d24-dense-1xh100-s0-4bad20ea`, Muon). Its history for steps 0-2000 is the comparison.

## What changes (and what does not)
- Unchanged vs the dense baseline: d24, fp8, ClimbMix, global batch 2^20 (dbs 16 x accum 32), seed 42, WSD schedule over 5,568 steps (`NUM_ITERATIONS=5568`), **LR rescale ON** (the dense baseline did not use `--no-lr-rescale`; batch_lr_scale = 1.4142), embedding / unembedding / scalar groups still AdamW with their baseline LRs.
- Changed: the transformer-block matrices (attention + MLP weights) go to AdamW. New flags in `base_train.py` (default = old behaviour, `--matrix-optimizer=muon`): `--matrix-optimizer=adamw --adamw-matrix-lr --adamw-matrix-wd --adamw-matrix-betas`. Shape grouping is kept; the only code touched is `GPT.setup_optimizer` and the `train/lr_matrix` log key (now selects the group tagged `is_matrix`, previously `kind=='muon'`).
- AdamW hyperparameters (LR 3e-4 and wd 0.1 set by Anatoly; betas/eps are the usual defaults): base LR **3e-4** (chosen by Anatoly), betas (0.9, 0.95), eps 1e-8, decoupled weight decay 0.1 constant (the Muon wd schedule and momentum warm-up only touch Muon groups). The effective LR is base x batch_lr_scale (1.4142) x 1/sqrt(1536/768) (0.7071) = 3e-4: the two factors cancel at d24 / batch 2^20.
- Same LR warm-up (40 steps) and WSD shape as the baseline, applied to the AdamW LR.

## Why the LR is the real question
Muon's LR (0.02) means nothing for AdamW, and the AdamW matrix LR is the single hyperparameter that decides whether this comparison is fair. 3e-4 was set by Anatoly; it is not tuned. The launcher takes `ADAMW_LR`; the run name carries it. If you want a fair "AdamW at its best" number, a short sweep over LR (e.g. 1e-4, 3e-4, 1e-3) is needed: each point costs the same ~4.6 h. This PR configures one point; further points reuse the same launcher.

Weight decay 0.1 is as set by Anatoly. Group: `nanochat_baseline` (confirmed).

## Setup / cost
wandb `spiky-nanochat`, group `nanochat_baseline`, run name = folder name, tags `adamw2k,matrix_opt=adamw,adamw_lr=3e-4`. `--max-steps=2001` so val bpb, CORE and the checkpoint at step 2000 run. Checkpoints every 500 (pruned to the newest 2). Dense speed ~8.3 s/step: about **4.6 h** plus evals. Smoke: `SMOKE=1`, 12 steps.

## Acceptance criteria (judged in Slack)
Train loss (10-step smoothed) and val bpb at 250/500/.../2000, CORE at 500/1000/1500/2000, versus the Muon dense baseline; also the steps-to-equal-loss ratio. One seed per setting, so differences of a few hundredths are not distinguishable from noise; a clear gap (the Muon speed-up is usually larger than that) would be.

## Status
Setup only (this PR). After approval: smoke -> full run -> results.

## COMPARISON (steps 0-2000, Muon reference vs AdamW matrices)
Runs (wandb `spiky-nanochat`, group `nanochat_baseline`):
- AdamW: https://forge.coreweave.com/wandb/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-adamw-lr3e-4-1xh100-s0-2k-213e13da
- Muon reference: https://forge.coreweave.com/wandb/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-1xh100-s0-4bad20ea

Files: `compare.csv` (per-step train loss, `train/lr_matrix`, val bpb, CORE for both), `compare.png` (train loss 10-step mean, val bpb, CORE). Both histories read with the `step` key.

**Config check.** Identical in both: batch (device 16, total_batch_size -1 = default 2^20 tokens), seq 2048, fp8 tensorwise, seed 42, embedding/unembedding/scalar/final_lr_frac, matrix_lr 0.02 (Muon group), lr-rescale on (Muon `lr_matrix` = 0.02 x 1.4142 = 0.0283; AdamW 3e-4 after the cancelling factors). `train/lrm` is identical at every step 0-2000 (max abs diff 0), so the WSD schedule is the same: the Muon run has `num_iterations=-1` (auto) vs explicit 5568 in the AdamW run, but the schedules coincide over 0-2000. Differences: the matrix optimizer (the point of the experiment), `max_steps=2001` for AdamW, `save_every` (250 vs 500), and new AdamW/LUT-ffn config keys absent from the older reference config (LUT-ffn is off). Step-0 val bpb differs slightly (3.171 vs 3.181), i.e. init/nondeterminism noise of ~0.01.

| step | val bpb Muon | val bpb AdamW | delta (AdamW-Muon) | CORE Muon | CORE AdamW | delta |
|---|---|---|---|---|---|---|
| 250 | 0.9924 | 1.2812 | +0.2888 | | | |
| 500 | 0.9022 | 1.0512 | +0.1490 | 0.1228 | 0.0514 | -0.0714 |
| 750 | 0.8651 | 0.9783 | +0.1132 | | | |
| 1000 | 0.8455 | 0.9354 | +0.0900 | 0.1714 | 0.1249 | -0.0465 |
| 1250 | 0.8321 | 0.9092 | +0.0771 | | | |
| 1500 | 0.8229 | 0.8862 | +0.0633 | 0.1807 | 0.1363 | -0.0444 |
| 1750 | 0.8160 | 0.8719 | +0.0559 | | | |
| 2000 | 0.8090 | 0.8588 | +0.0499 | 0.1907 | 0.1553 | -0.0353 |

Train loss, mean of steps 1951-2000: Muon 2.662, AdamW 2.822 (+0.16).

**Verdict.** At 2000 steps Muon is clearly ahead: -0.050 bpb (0.809 vs 0.859) and +0.035 CORE (0.191 vs 0.155; the CORE gap is 0.07 at step 500 and narrows). The gap shrinks monotonically (0.29 -> 0.05 bpb), so this is mostly a speed-up, not a fixed offset, and it is still closing. In steps: Muon reaches AdamW's step-2000 bpb (0.8588) at about step 830 (interpolated between 750 and 1000), a ~2.4x speed-up in steps at that point. Reading the other way, AdamW reaches Muon's step-500 bpb (0.902) at ~1330 steps (2.7x) and Muon's step-750 bpb (0.865) at ~1880 steps (2.5x); AdamW has not reached Muon's bpb at step 1000 or later within 2000 steps (so "AdamW needs > 2000 steps" for Muon@1000 and beyond; extrapolation from this curve would be unreliable). Wall-clock per step was not compared here.

**Caveats.** One seed per setting (step-0 offset ~0.01 bpb suggests noise at about that level, small relative to the gaps). AdamW LR 3e-4 and wd 0.1 were set by hand and not tuned; a better AdamW LR could shrink the gap, so this is Muon vs *this* AdamW, not vs AdamW at its best. Step 2000 is mid-schedule of a 5568-step WSD run (stable LR phase, not decayed), so these are not final-quality numbers and the gap at the end of training may differ. CORE is a noisy 4-point metric at this scale; val bpb is the more reliable signal. The steps-to-equal values are linear interpolations over 250-step eval spacing.
