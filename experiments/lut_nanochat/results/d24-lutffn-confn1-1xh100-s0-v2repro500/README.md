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

## Status
Setup only (this PR). Not run yet. After approval: smoke -> full run -> results pushed to this folder (`train.log`, `pin_config.json`, `run_config.json`, `manifest.json`, comparison plot).
