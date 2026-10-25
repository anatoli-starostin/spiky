# d24-lutffn-confn1-1xh100-s0-nodropout1500

**Question.** What does LUT table dropout (0.2 in v2) cost or buy in the first 1500 steps? Same run as v2, table dropout switched off.

**Only change vs `d24-lutffn-confn1-1xh100-s0-v2repro500` / v2:** `--lut-ffn-table-dropout=0` (default 0.2). `0.0` is supported by the cartridge (the dropout mask is skipped when the rate is 0).

**References** (wandb `spiky-nanochat`, group `nanochat_lut_ffn`):
- v2 `d24-lutffn-confn1-1xh100-s0-full-v2-9811cced`, steps 0-1500 (dropout 0.2).
- `d24-lutffn-confn1-1xh100-s0-v2repro500-2959ed04`, steps 0-500 (dropout 0.2, seed-identical recipe, the repro of v2).
- Dense `d24-dense-1xh100-s0` for context.

## Setup
- Launcher `nanochat/runs/lut_ffn_d24_nodropout1500_1xh100.sh` (wrapper over `lut_ffn_d24_1xh100.sh`): LUT-FFN confidenceLUT read_top_n=1, r=768, tph=64, nap=8, h16/d48; `--no-lr-rescale`; d24, fp8, global batch 2^20 (dbs 16 x accum 32), seed 42, ClimbMix.
- **Full-run schedule:** `NUM_ITERATIONS=5568` (LR/WSD as in the 5.5K-step run), stop with `--max-steps=1501` (the extra step so val bpb, CORE and the checkpoint at step 1500 run).
- Checkpoints every 500 steps (pruned to the newest 2). Val bpb every 250 steps, CORE every 500 (in-training subsample).
- wandb: run name = folder name, tags `nodropout1500,table_dropout=0`. Smoke: `SMOKE=1`, tag `smoke`, 12 steps.
- Note: the v2 seed-identical init does not depend on dropout, so step-0 val bpb should match repro500 closely; the first differences come from the removed dropout masks.

## Expected cost
~14.2 s/step: about 6 h for 1501 steps plus evals. Needs one long-run approval. Disk: ~11 GB per checkpoint.

## Acceptance criteria (judged in Slack by Anatoly)
Compare steps 0-1500 against v2 (dropout 0.2): train loss (10-step smoothed), val bpb at 250/500/.../1500, CORE at 500/1000/1500. Caveat: no seed replicate exists, and the v2 reference has a loss spike around steps 196-225 that repro500 did not show, so differences of a few hundredths are within noise; say so plainly in the verdict.

## COMPARISON (steps 0-1500, new run `d24-lutffn-confn1-1xh100-s0-nodropout1500-8df9b1fa` vs v2 `...-full-v2-9811cced`, repro500 `...-v2repro500-2959ed04`, dense `d24-dense-1xh100-s0-4bad20ea`)

Data: wandb histories (`train/loss`, `val/bpb`, `core_metric`, `train/dt`), `compare_first1500.csv` (raw and 10-step trailing-mean losses), `compare_first1500.png`. The new run finished (steps 0-1500, commit `437ba995`, see `run_config.json`). repro500 only covers steps 0-500.

| metric | new (no dropout) | v2 (dropout 0.2) | repro500 (dropout 0.2) | dense | new - v2 |
|---|---|---|---|---|---|
| train loss, mean steps 1-10 | 9.842 | 9.838 | 9.841 | 9.665 | +0.004 |
| train loss, mean steps 91-100 | 4.670 | 4.635 | 4.667 | 4.403 | +0.034 |
| train loss, mean steps 241-250 | 3.515 | 3.633 | 3.510 | 3.311 | -0.118 |
| train loss, mean steps 491-500 | 3.132 | 3.188 | 3.132 | 2.976 | -0.056 |
| train loss, mean steps 991-1000 | 2.962 | 2.961 | - | 2.778 | +0.002 |
| train loss, mean steps 1491-1500 | 2.851 | 2.879 | - | 2.711 | -0.027 |
| val bpb, step 0 | 3.1714 | 3.1808 | 3.1714 | 3.1808 | -0.0094 |
| val bpb, step 250 | 1.0599 | 1.0814 | 1.0560 | 0.9924 | -0.0215 |
| val bpb, step 500 | 0.9606 | 0.9662 | 0.9603 | 0.9022 | -0.0056 |
| val bpb, step 750 | 0.9195 | 0.9237 | - | 0.8651 | -0.0042 |
| val bpb, step 1000 | 0.8987 | 0.9004 | - | 0.8455 | -0.0017 |
| val bpb, step 1250 | 0.8831 | 0.8853 | - | 0.8321 | -0.0022 |
| val bpb, step 1500 | 0.8735 | 0.8749 | - | 0.8229 | -0.0014 |
| CORE, step 500 (in-training subsample) | 0.0959 | 0.0988 | 0.0973 | 0.1228 | -0.0029 |
| CORE, step 1000 (in-training subsample) | 0.1435 | 0.1489 | - | 0.1714 | -0.0054 |
| CORE, step 1500 (in-training subsample) | 0.1588 | 0.1418 | - | 0.1807 | +0.0170 |
| sec/step, mean steps 11-1500 (`train/dt`) | 13.98 | 14.54 | 14.21 (steps 11-500) | 8.32 | -0.56 (-4%) |
| peak memory allocated / reserved (MiB) | 63,608 / 65,460 | 64,377 / 65,458 | 64,378 / 65,460 | 52,673 / 53,474 | -770 / +2 |

CORE is the in-training evaluation on a subsample (`core_metric_max_per_task=500`): the four columns are comparable to each other only, not to full-CORE numbers (v2's full CORE is in `../d24-lutffn-confn1-1xh100-s0-full-v2/`). The dense run's step-0 bpb equals v2's (3.1808), the new run's equals repro500's (3.1714). "sec/step" for the new run is lower than v2's, but the machines differ (v2 ran on another host), so it is not a dropout effect on its own.

**First divergence from v2.** The 10-step-smoothed difference to v2 first exceeds 0.01 (absolute) at **step 16** (negative sign; first positive exceedance at step 58). 76% of steps 10-1500 are outside the 0.01 band, so this criterion is not informative here: the band is narrower than the scatter.

**Mean difference over steps 500-1500, excluding the v2 spike window 196-225** (the window lies before step 500, so the exclusion changes nothing): raw new-minus-v2 loss **+0.007** (naive standard error 0.001, which ignores autocorrelation and is too small). By sub-window: 500-750 -0.005, 750-1000 +0.006, 1000-1250 +0.015, 1250-1500 +0.012. Over steps 226-500, the first stretch after the spike, it is -0.039 (v2 is still recovering from the spike there). The smoothed difference stays within [-0.058, +0.085] over steps 500-1500.

**Trend of new-minus-v2 over steps 500-1500.** It grows: a linear fit on the raw per-step difference gives +0.023 per 1000 steps (naive standard error 0.003; smoothed fit +0.026), from about -0.005 at 500-750 to about +0.012-0.015 at 1000-1500, i.e. the new run goes from slightly ahead of v2 to slightly behind it in train loss. The val bpb gap, on the other hand, shrinks (-0.0056 at 500, -0.0042 at 750, -0.0017 at 1000, -0.0022 at 1250, -0.0014 at 1500), with the new run still marginally lower at 1500. The two measures disagree in sign at the end (train loss at 1491-1500 is -0.027, but the 500-1500 mean is positive), so the trend is small and not clearly different from drift of a single run.

**Same-code control (0-500).** The new run and repro500 share code, seed and data order and differ only in table dropout. Their raw losses coincide at the start (difference below 1e-4 for steps 0-4) and stay close: mean new-minus-repro500 -0.002, per-step sd 0.008, largest single-step gap 0.05 (step 178). val bpb at 250: 1.0599 vs 1.0560; at 500: 0.9606 vs 0.9603; CORE-500 0.0959 vs 0.0973. By contrast repro500-minus-v2 has per-step sd 0.096 (mostly the v2 spike). So in steps 0-500, removing table dropout moves the run far less than the difference between the two code versions does. The new run shows no spike at 196-225 (nor does repro500).

**Verdict.** Removing table dropout gives no measurable change in train loss or val bpb vs v2 beyond the run-to-run noise estimate. Against v2 the new run is within the +-0.04 band after the spike (smoothed difference -0.06..+0.09 over steps 500-1500, mean +0.007); val bpb is 0.0014-0.0056 lower from step 500 on, which is smaller than the 0.0025 (step 500) and 0.025 (step 250) differences already seen between repro500 and v2. The difference at step 250 (-0.0215 vs v2) is the v2 spike recovery, which repro500 shows as well (-0.0254), not an effect of dropout. The same-code control against repro500 (steps 0-500) agrees: no difference detectable there. The one measure that is outside what the 500-step data would predict is the slow drift of train loss to +0.01-0.015 above v2 at steps 1000-1500, with the new run's CORE-1500 (0.159) above v2's (0.142) in the opposite direction; with a single run per setting, no seed replicate and in-training subsample CORE, none of these can be attributed to dropout. Table dropout also saves memory (-770 MiB allocated), which indicates it was active in v2/repro500. Caveat: v2 has a loss spike at 196-225 that neither repro500 nor the new run has, so a seed replicate of v2 would be needed to estimate noise past step 500.
