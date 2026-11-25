# lutorch_ex_conf_n1_single_noproj_5k16s — conf n=1, single probes, no input projection; first 5000 steps of the 16K lr schedule

Repeat of `../lutorch_ex_conf_n1_single_noproj_5k` with ONE change: the lr schedule is the **16K schedule cut at step 5000**
(`n_steps=16000` schedule: warm-up 10% = 1600 steps, peak 3e-4, cosine to 0.1x over 16000; the run stops after step 5000;
lab convention for 5K runs). Everything else identical (geometry `LUTSpec(h_in=1,h_out=8,tph=64,nap=8,d_in=384,d_out=48,anchor_mode="single")`,
compress=False, decompress kept, `NO_INPUT_PROJ=1 ANCHOR_MODE=single`, conf n=1, cell-TV 10, head-dropout 0.2, seed 1, batch/seq/data, eval every 500,
same environment, H100). Code change: `STOP_AT_STEP` env flag in `../train_multi.py` (default = N_STEPS = old behaviour); `val/bpb` is logged next to `val_bpb`.

## Schedule check (from W&B `train/lr`)
| step | lr (W&B) | 16K formula |
|---|---|---|
| 500 | 9.375e-05 | 9.375e-05 |
| 1000 | 1.875e-04 | 1.875e-04 |
| 1600 | 3.000e-04 (peak) | 3.000e-04 |
| 5000 | 2.645e-04 | 2.645e-04 |

## Curve (val bpb)
| step | this run (5K of 16K schedule) | full 5000-step schedule (`../lutorch_ex_conf_n1_single_noproj_5k`) | 16K original (qwnyrzbh, other geometry) |
|---|---|---|---|
| 500 | 1.9856 | 1.7999 | 2.0125 |
| 1000 | 1.7375 | 1.6043 | 1.7528 |
| 1500 | 1.6040 | 1.5114 | 1.6082 |
| 2000 | 1.5163 | 1.4496 | 1.5093 |
| 2500 | 1.4576 | 1.3994 | 1.4453 |
| 3000 | 1.4078 | 1.3641 | 1.3871 |
| 3500 | 1.3675 | 1.3382 | 1.3462 |
| 4000 | 1.3402 | 1.3206 | 1.3190 |
| 4500 | 1.3177 | 1.3090 | 1.2953 |
| 5000 | **1.2997** | 1.3021 | 1.2790 |

**The full-5000-step-schedule run (final 1.3021) is NOT comparable with the other 5K cells** (warm-up 500 and full decay to 0.1x within 5000 steps,
so it is far ahead mid-run); the comparable 5K number for this geometry is this run's 1.2997. Versus the 16K original (same schedule, but the original
has per-head compress 384->48 instead of the fan-out input side) this run is 0.027 better at 500, ~0.015 better at 1000, and 0.007-0.021 worse from 2000 on
(final +0.021 at 5000); the schedule itself matches exactly (lr table), the differences are the input-side change plus H100-vs-RTX 5090 numerics.

## Run
Params 67,350,540, peak mem 7.67 GB, **219 ms/step** mean (0.32 h) on H100.
* W&B: https://forge.coreweave.com/wandb/deepspike-inc/tally_transformer/runs/lutorch_ex_conf_n1_single_noproj_5k16s (group lutorch_ex; `val_bpb` and `val/bpb` every 500; checked against metrics.csv by `verify_wandb.py`, see `verify.log`).
* Checkpoint (step 5000): s3://deepspike-data/anatoly/lutorch_ex/lutorch_ex_conf_n1_single_noproj_5k16s/ckpt.pt
* Reproduce: `bash run.sh`.
