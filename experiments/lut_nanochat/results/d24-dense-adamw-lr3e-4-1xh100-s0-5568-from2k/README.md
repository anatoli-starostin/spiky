# d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k

**Question.** Continuation of the dense AdamW d24 run (`d24-dense-adamw-lr3e-4-1xh100-s0-2k`, steps 0-2000, see its README for the Muon-vs-AdamW comparison at 2k) up to the **full 5,568-step schedule**, so that AdamW vs Muon (`d24-dense-1xh100-s0`, Muon) can be compared at the end of training (final val bpb and CORE), not only mid-schedule.

## What is the same / what changes
- Same model, data, fp8, global batch 2^20 (dbs 16 x accum 32), seed 42, WSD schedule over 5,568 steps (`NUM_ITERATIONS=5568`; the first run had the same horizon, so the LR schedule is continuous), LR rescale ON, AdamW matrices with LR 3e-4 and wd 0.1 (betas 0.9/0.95), group `nanochat_baseline`.
- Nothing about the optimization changes. This is a resume from the **step-2000 checkpoint** (model + optimizer state) of the first run, `--resume-from-step=2000`, with no `--max-steps`, to step 5568. Evals as in the baseline: val bpb every 250, CORE every 500 (in-training subsample), plus the standalone `base_eval` (full CORE) and `report_results` at the end, like the dense baseline.
- Caveat (inherent to nanochat resume, same as the baseline launcher's note): the **dataloader resume is approximate**, so steps 2000+ see a slightly different data order than an uninterrupted run would. The model/optimizer state is exact. Treat the continuation as equal to an uninterrupted run up to this noise.

## Separate experiment, full curves
- Own results folder, own checkpoint dir, own wandb run `d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k-<id>` (group `nanochat_baseline`, tag `continued_from_2k`).
- The step-2000 checkpoint is **hard-linked** from the first run's `checkpoints/` (no extra disk; the original is untouched). The launcher stops if it is missing.
- **History replay** (decided with Anatoly): at the start of the new wandb run, `base_train.py --replay-wandb-from=<entity>/<project>/<first run id>` re-logs every row of the first run with `step < 2000` (train, val, CORE, lr, ...) into the new run, each with its own `step` key (the x-axis of every metric since the x-axis fix). Live logging then starts at step 2000. So the new run shows the complete 0-5568 curves. Replay happens only into a fresh wandb run, so a crash-resume of the new run does not duplicate it. The wandb config of the run records `replayed_from_run`, `replayed_steps` (0-1999) and `replayed_rows`. The replayed part is a copy: the first run remains the source of truth for steps < 2000.

## Cost
Steps 2000 -> 5568 at ~8.4 s/step: about **8.3 h** plus evals/ checkpoints (every 500, pruned to the newest 2, ~11 GB each) and the final base_eval, i.e. roughly **9-9.5 h** on 1xH100. Smoke: `SMOKE=1` resumes at 2000, runs 3 steps with evals at 2000/2002 and the replay, wandb tag `smoke`.

## Launch
```
SMOKE=1 bash runs/dense_adamw_d24_continue_1xh100.sh   # smoke
bash runs/dense_adamw_d24_continue_1xh100.sh           # full continuation (resumes by itself after a crash)
```
Prerequisite: the first run's step-2000 checkpoint on the worker (`results/d24-dense-adamw-lr3e-4-1xh100-s0-2k/checkpoints/`). This branch is forked from `experiment/lut_nanochat-dense-adamw2000` (PR #175), which carries the AdamW code; if #175 is merged first, this PR retargets automatically.

## Acceptance criteria (judged in Slack)
Smoke: the replay logs the first run's rows (count reported), val at 2000 matches the first run's step-2000 val bpb (0.8588) within eval noise, loss continues smoothly (no jump at 2000), no errors. Full: complete curves 0-5568 in wandb; final val bpb / CORE vs the Muon dense baseline at 5568; no NaN/spike. One seed, untuned AdamW LR: say so in the verdict.

## Status
Setup only (this PR). After approval: smoke (own task) -> full run (own task) -> results (compare vs Muon over the full schedule).
