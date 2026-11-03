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

## RESULT (run finished 2026-10-26 17:08 UTC, exit 0)
- Steps 2000 -> 5568, ~8.17-8.21 s/step, ~128k tok/s (typical; the first step after resume 114 s incl. startup), total training time 768.7 min (including the 2000 replayed steps' bookkeeping, as logged), peak memory 55,276 MiB allocated / 57,468 MiB reserved, grad accum 32. One resume (`resumes.log`, from step 2000 as designed); no WARNING/ERROR/Traceback/NaN and no loss spike in the log (largest single-step loss increase after 2100: +0.031, Muon has +0.037).
- Final in-training val bpb 0.763161 (step 5568), standalone base_eval val bpb 0.764247. **Standalone full CORE 0.2125**, in-training CORE (subsample) 0.2225 at 5568. `report_results` verdict line: `FAIL: below the GPT-2 bar` (0.2125 < 0.256525).
- wandb: https://wandb.ai/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k-94a95f93

## COMPARISON (full schedule 0-5568, Muon vs AdamW)
Muon = `d24-dense-1xh100-s0` (wandb `d24-dense-1xh100-s0-4bad20ea`), AdamW = this run (steps < 2000 are the replayed copy of the first run). Same model, data, batch, WSD schedule, seed. Data: `compare.csv`, plot: `compare.png` (dashed line = step 2000, where the AdamW run was resumed). CORE in the table is the in-training subsample CORE (every 500); the standalone full CORE is below.

| step | val bpb Muon | val bpb AdamW | delta (AdamW-Muon) | CORE Muon | CORE AdamW | delta |
|---|---|---|---|---|---|---|
| 500 | 0.9022 | 1.0512 | +0.1490 | 0.1228 | 0.0514 | -0.0714 |
| 1000 | 0.8455 | 0.9354 | +0.0900 | 0.1714 | 0.1249 | -0.0465 |
| 2000 | 0.8090 | 0.8588 | +0.0499 | 0.1907 | 0.1553 | -0.0353 |
| 3000 | 0.7727 | 0.8180 | +0.0453 | 0.2297 | 0.1780 | -0.0518 |
| 4000 | 0.7469 | 0.7918 | +0.0449 | 0.2380 | 0.1924 | -0.0456 |
| 5000 | 0.7267 | 0.7716 | +0.0449 | 0.2469 | 0.2107 | -0.0362 |
| 5568 | 0.7193 | 0.7632 | +0.0438 | 0.2547 | 0.2225 | -0.0321 |

Standalone base_eval full CORE: **Muon 0.2547 vs AdamW 0.2125 (delta -0.0422)**; standalone val bpb 0.7207 vs 0.7642 (+0.0435). Final train loss (mean of last 50 steps): Muon 2.345, AdamW 2.508.

**Final gap: +0.0438 bpb (AdamW worse).** In steps: Muon reaches AdamW's final val bpb (0.7632) at step ~3355 (linear interpolation on the 250-step val grid), i.e. AdamW needs 5568 steps for what Muon does in ~3355: Muon is ~2200 steps (40%) ahead, ~1.66x step-efficiency at the end of the schedule (at 2000: Muon matched AdamW's bpb at step ~830, 2.4x).

**Verdict.**
- At the end of the schedule Muon still delivers a clear, consistent advantage: -0.044 val bpb and +0.042 full CORE (about 5x the CORE per-run noise of ~0.008, about 4x the step-0 init noise of ~0.01 in bpb). The gap is not noise.
- After step 2000 the gap shrinks only slightly: +0.050 (2000) -> +0.048 (2500) -> +0.045 (3000) and then flat at +0.044..0.045 until the end (5568: +0.0438). The change (~0.006 bpb) is of the order of the init noise, so read it as "stable, at most a small shrink"; the LR decay did not close it. CORE gaps fluctuate between -0.03 and -0.054 with no trend (noise-limited).
- In step-equivalents the lead moves from ~1170 steps (2000) to ~2200 steps (5568), while the relative advantage drops from 2.4x to 1.66x.

**Caveats.** One seed per optimizer; AdamW LR 3e-4 / wd 0.1 untuned (a tuned AdamW could close part of the gap), Muon at its default; dataloader resume at 2000 is approximate (data order after 2000 differs from an uninterrupted run; model/optimizer state exact); step-0 init noise ~0.01 bpb, CORE noise ~0.008; the in-training CORE is a subsample, the standalone one is the full CORE. The `report_results` line `FAIL: below the GPT-2 bar` is the fixed threshold (0.2565) for this model size, not an invalid run (Muon's 0.2547 is also just below it).

wandb: AdamW continuation https://wandb.ai/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-adamw-lr3e-4-1xh100-s0-5568-from2k-94a95f93 ; Muon reference https://wandb.ai/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-1xh100-s0-4bad20ea ; first AdamW run (steps 0-2000) https://wandb.ai/anatoli-starostin-relocation/spiky-nanochat/runs/d24-dense-adamw-lr3e-4-1xh100-s0-2k-213e13da
