# d24-lutffn-confn1-1xh100-s0-v2repro500

**Question.** Does the current `research/lut_nanochat` code, on a fresh machine (worker `ds-h100-anatoly`), reproduce the first 500 steps of the reference run `d24-lutffn-confn1-1xh100-s0-full-v2`?

**Reference.** `results/d24-lutffn-confn1-1xh100-s0-full-v2/` (wandb `spiky-nanochat`, run `d24-lutffn-confn1-1xh100-s0-full-v2-9811cced`). Its exact configuration is in that folder's `run_config.json`.

## Setup (identical to v2, except for the early stop)
- Launcher: `nanochat/runs/lut_ffn_d24_repro500_1xh100.sh`, a thin wrapper over `lut_ffn_d24_1xh100.sh`. It sets the LUT-FFN recipe (confidenceLUT, read_top_n=1, r=768, tph=64, nap=8, h16/d48), `--no-lr-rescale`, d24, fp8, global batch 2^20 (dbs 16 x grad-accum 32), seed 42, ClimbMix.
- **Full-run schedule:** `NUM_ITERATIONS=5568`, so the LR/WSD schedule is the one of the 5.5K-step run. The run only stops early with `--max-steps=501`.
- Why 501 and not 500: evals happen at the top of the loop, so val bpb, CORE and the step-500 checkpoint need the loop to reach step 500.
- Checkpoints at steps 250 and 500 only (`SAVE_EVERY=250`).
- `SKIP_BASE_EVAL=1`: no standalone `base_eval` (no horizon checkpoint exists). The small change to `baseline_d24_1xh100.sh` is a default-off hook.
- wandb: project `spiky-nanochat`, group `nanochat_lut_ffn`, run name = this folder name, tag `repro500`. Smoke run: `SMOKE=1`, tag `smoke`, 12 steps, run name `...-v2repro500-smoke` (to be deleted by Anatoly).
- `WANDB_BASE_URL=https://api.wandb.ai` (the forge URL is only the UI).

## Expected cost
~14.5 s/step, so ~2 h for 500 steps plus evals. Checkpoint disk: ~11 GB each.

## Acceptance criteria (to be judged in Slack by Anatoly)
Compare steps 0-500 against the reference from wandb: train/loss (smoothed), val/bpb at 250 and 500, CORE at 500, tokens/s and memory. Bitwise equality is not expected: the fused-kernel code was refactored after v2 ran (v2 ran at commit `9f52b1fb`, which now lives only on `keep/pre-rebase-ef98e323`), and fp8/atomics are non-deterministic. Divergence beyond run-to-run noise at the same step is the thing to flag.

## COMPARISON (steps 0-500, new run `d24-lutffn-confn1-1xh100-s0-v2repro500-2959ed04` vs reference `...-full-v2-9811cced`)

Data: wandb histories (`train/loss`, `val/bpb`, `core_metric`, `train/dt`), `compare_v2_first500.csv`, `compare_v2_first500.png`. Both runs finished; the new run ran from commit `3c53f98a` (see `run_config.json`).

| metric | ref (v2) | new (repro500) | new - ref |
|---|---|---|---|
| train loss, mean steps 1-10 | 9.838 | 9.841 | +0.003 |
| train loss, mean steps 91-100 | 4.635 | 4.667 | +0.032 |
| train loss, mean steps 241-250 | 3.633 | 3.510 | -0.123 |
| train loss, mean steps 491-500 | 3.188 | 3.132 | -0.056 |
| val bpb, step 0 | 3.1808 | 3.1714 | -0.0095 |
| val bpb, step 250 | 1.0814 | 1.0560 | -0.0254 |
| val bpb, step 500 | 0.9662 | 0.9603 | -0.0059 |
| CORE, step 500 (in-training, `core_metric_max_per_task=500` subsample, both runs) | 0.0988 | 0.0973 | -0.0015 |
| sec/step, mean steps 11-500 (`train/dt`) | 14.50 | 14.21 | -0.29 (-2%) |
| tok/s, mean steps 11-500 | 72,308 | 73,784 | +2% |
| peak memory (allocated / reserved, MiB) | 64,377 / 65,458 | 64,378 / 65,460 | ~0 |

Step-500 CORE in both runs is the in-training evaluation on a subsample of at most 500 examples per task, not the full CORE eval, so it is only comparable to each other (reference full-CORE numbers are in `../d24-lutffn-confn1-1xh100-s0-full-v2/`). Val bpb at step 0 already differs by 0.0095 (initial train loss 10.3980 vs 10.3974), so the two initialisations are not identical.

**First divergence.** On the 10-step trailing-mean loss the curves first differ by more than 0.01 (absolute) at **step 16**; they come back inside 0.01 repeatedly, and 83% of steps 10-500 are outside the 0.01 band. Largest gap 0.48 at step 206.

**What drives the large gaps.** The reference has a loss spike at roughly steps 196-225 (raw loss ~4.1 vs ~3.65 for the new run around step 200; smoothed peak +0.48), which the new run does not have. That spike accounts for the -0.12 at steps 241-250 and the val bpb gap at 250 (the reference has only just recovered from the spike at step 250). Outside that spike the curves track within about +-0.04 (smoothed), with the new run ahead by about 0.015 over steps 300-500 on average, and -0.056 in the last window.

**Verdict.** This is not shown to be a reproduction within noise. Step-to-step scatter of the loss around its local trend is only ~0.010-0.015 per run, and the window differences (+0.032 at steps 91-100, -0.123 at 241-250, -0.056 at 491-500) are several standard errors of that scatter (standard errors of the per-step differences 0.006, 0.012, 0.005); the final val bpb is 0.006 lower and 0.025 lower at step 250. Both runs have the same data order and seed, but with one run per code version there is no seed-replicate to measure true run-to-run noise, so this cannot be settled from these two runs alone; a loss spike such as the reference's may itself be run-to-run variation (unverified), in which case the new run's lower losses would just reflect it not having one. By step 500 the new run is slightly ahead (val bpb 0.960 vs 0.966, CORE-500 0.097 vs 0.099, i.e. equal within the subsample noise), with no sign of a regression, but the early trajectory (step 16 onwards, and especially around step 200) differs more than the per-step noise. Throughput and memory match (+2% tok/s, same peak memory). A seed-replicate of either version would be needed to call this a reproduction.
