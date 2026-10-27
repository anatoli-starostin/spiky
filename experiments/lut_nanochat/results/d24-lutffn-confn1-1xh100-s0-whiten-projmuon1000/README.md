# d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000

**Question.** Does the whiten (Muon-analogue) table optimizer help when *everything else* is Muon too? Earlier `../d24-lutffn-confn1-1xh100-s0-whiten500/` (tables on whiten, projections on AdamW) showed no gain over AdamW in 500 steps. Hypothesis (Anatoly): the idea may only work if applied everywhere - Muon on the attention/block matrices (already the case in all LUT runs), Muon on the ProjectionMHL projections, plus whitening on the LUT tables.

## Setup
Identical to `../d24-lutffn-confn1-1xh100-s0-whiten500/` (v2 recipe, `--no-lr-rescale`, `NUM_ITERATIONS=5568` schedule, `--max-steps=1001`, seed 42, batch 2^20, fp8) plus **`--lut-ffn-proj-fp8-muon`**: ProjectionMHL compress/decompress GEMMs in fp8 (fp32 masters), projection *weights* on Muon at matrix-lr (0.02, no LR rescale); biases and confidence scalars stay AdamW. Tables: `--lut-table-optimizer=whiten` (same untuned RMS 0.3 / beta 0.9 / eps 1e-3). Launcher: `nanochat/runs/lut_ffn_d24_whiten_projmuon1000_1xh100.sh`; wandb group `nanochat_lut_ffn`, tag `whiten-projmuon1000`.

**Checkpoints** at steps 250, 500, 750 and 1000 are kept on the worker (`SAVE_EVERY=250`, `KEEP_CKPTS=4`, not committed); continue to 5568 with the same launcher and `TRAIN_EXTRA="--max-steps=5569"`.

## Confound and planned control (B)
`--lut-ffn-proj-fp8-muon` also changes the projections' precision (fp8 vs fp32 island) and the optimizer, so a gain here cannot be attributed to the whitened tables alone. Control B (same flag, AdamW tables) is to be run afterwards (a separate experiment) if this run looks interesting.

## Baselines
`../d24-lutffn-confn1-1xh100-s0-full-v2/` (AdamW, first 1000 steps from its wandb history; repro500 only covers 500), `../d24-lutffn-confn1-1xh100-s0-v2repro500/` (AdamW, to 500), `../d24-lutffn-confn1-1xh100-s0-whiten500/` (tables only), reference v2. Noise caveat as in the repro500 README (single seed; AdamW repro vs v2 differ up to 0.12 loss at step ~200).

## Acceptance criteria (to be judged in Slack by Anatoly)
train/loss windows, val/bpb at 250/500/750/1000, CORE at 500 and 1000, s/step, memory, stability vs the baselines above. Result pending.
