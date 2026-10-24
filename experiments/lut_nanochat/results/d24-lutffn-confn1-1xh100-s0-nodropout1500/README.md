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

## Status
Setup only (this PR). After approval: smoke -> full run -> results.
