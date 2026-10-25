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
