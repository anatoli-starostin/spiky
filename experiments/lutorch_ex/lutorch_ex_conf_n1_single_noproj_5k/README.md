# lutorch_ex_conf_n1_single_noproj_5k — fork of lutorch_ex_conf_n1_16k: no input projection + single probes, 5K schedule

Fork of `../lutorch_ex_conf_n1_16k_1125` (W&B qwnyrzbh) with the **input side** changed; everything else identical
(ConfidenceLUT read_top_n=1 via `CART=conf_n1`, tph64, nap8, d_out 48, 6 blocks, cell-TV lambda=10, head-dropout 0.2,
seed 1, optimizer/LR/WD/batch/seq/tokenizer/data, nanochat 92d63d4 with stub rustbpe, same spiky.lutorch overlay as the 16K run).

## Geometry (flags in `train_multi.py`, helpers in `noproj_geometry.py`; default = old behaviour)
`NO_INPUT_PROJ=1 ANCHOR_MODE=single`:
* `LUTSpec(h_in=1, h_out=8, tph=64, nap=8, d_in=384, d_out=48, anchor_mode="single")` — fan-out: all 8 groups x 64 tables read the
  **same whole 384 stream**; `ProjectionMHL(..., compress=False)` (identity input side; the decompress 8*48=384 -> 384 is kept).
  (In the 16K run each head had its own compress 384->48; heads were not stream slices.) input_dim 384 = 1*384, output 8*48 = 384.
* Single probes: each table's 8 address bits are `z[a] > 0` on 8 coordinates of the 384 stream drawn per table from the seed (nap 8 <= d_in 384).
* Accepted as is by the ConfidenceLUT (n=1) trainer class; no library change. `test_noproj_geometry.py`: flags off == old model
  (identical state dict and output), flags on = geometry above (passes, GPU).

Schedule: **n_steps=5000 as a full schedule** (warm-up 10% = 500 steps, cosine to 0.1x over 5000, eval every 500) — not a cut of the 16K schedule.

## Results
Params **67,350,540** (16K run: 68,237,580; difference = 6 x 147,840 = the six removed compress matrices 384x384+384). Peak mem 7.67 GB.
Speed: **219 ms/step** mean over the logged steps (16K run on RTX 5090: ~170 ms; this is an H100, 5.3 GB-class batch 12x512x1, grad_accum 4), 0.32 h total.

| step | 5k fork (full 5K schedule) | 16K original (mid-schedule) |
|---|---|---|
| 500 | 1.7999 | 2.0125 |
| 1000 | 1.6043 | 1.7528 |
| 1500 | 1.5114 | 1.6082 |
| 2000 | 1.4496 | 1.5093 |
| 2500 | 1.3994 | 1.4453 |
| 3000 | 1.3641 | 1.3871 |
| 3500 | 1.3382 | 1.3462 |
| 4000 | 1.3206 | 1.3190 |
| 4500 | 1.3090 | 1.2953 |
| 5000 | 1.3021 | 1.2790 |

Final val bpb 1.3021 @5000. The 16K column is the original run's value at the same step under *its* (still high-LR) schedule, so it is not
a like-for-like 5K baseline (a 5K-schedule baseline of the original model was not run).

## Reproduce
`bash run.sh` (this folder; paths for the machine it ran on: H100 `ds-h100-anatoly`). `config.json` = 16K config with n_steps=5000, exp_name changed
(`lut_read_top_n: 2` there is unused for the cartridge, as in the 16K README).

* W&B: https://forge.coreweave.com/wandb/deepspike-inc/tally_transformer/runs/lutorch_ex_conf_n1_single_noproj_5k (group lutorch_ex; val bpb logged as `val_bpb` every 500, live; final values checked against metrics.csv by `verify_wandb.py`).
* Checkpoint (model+opt, step 5000): s3://deepspike-data/anatoly/lutorch_ex/lutorch_ex_conf_n1_single_noproj_5k/ckpt.pt

## Deviations
Only the intended input-side change and the 5K schedule. Hardware differs from the 16K run (H100 vs RTX 5090). Metric key is `val_bpb` (the tracker of the original run), not `val/bpb`.
