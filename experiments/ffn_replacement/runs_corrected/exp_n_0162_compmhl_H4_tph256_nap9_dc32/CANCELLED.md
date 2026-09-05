# CANCELLED — too large for gpustar, deferred to nebius-h100

Stopped deliberately on 2026-09-05, not a failure. Anatoly's call: the nap9 CompressionMHL
models are too big to be training on the RTX 5090; the large configs move to nebius-h100 and
gpustar switches to the smallest cell (`exp_n_0164_compmhl_H4_tph256_nap8_dc32`).

**Step reached: 600 of 16,000** (3.75%). Last logged eval, on the fixed protocol
(bs48 x 100, skip 12):

| step | train loss (ema) | val bpb |
|---|---|---|
| 200 | 9.378390 | 2.595468 |
| 400 | 7.591211 | 2.111455 |
| 600 | 6.621668 | 1.925812 |

Loss was falling normally — nothing was wrong with the run. The training process (PID 724400)
was sent SIGTERM and exited cleanly; GPU memory returned to the desktop baseline (2,060 MiB,
4% util) with no orphaned workers. No checkpoint was written (the trainer saves only at the
end), so there is nothing to resume from; a future run on nebius starts from scratch.

## What is kept here

| file | what |
|---|---|
| `config.json` | the full config as launched (device_batch 6 / grad_accum 8) |
| `train.py` | fork of `../../train_fixed.py` — fixed eval protocol, unchanged |
| `metrics.csv` | the 3 eval rows above |
| `train.log` | run log to step 600 (gitignored) |
| `train_oom_bs12.log` | the first launch, which OOMed at step 1 (gitignored) |

## The bs12 → bs6 story, for whoever picks this up on nebius

`exp_n_0118`'s `device_batch_size: 12` is an 80 GB H100 setting. The nap9 soft-backward
buffer is `[tokens, H*tph=1024, 2^nap=512]` fp32 = 12.9 GiB at bs12 and is held twice, so a
step needs ~39.5 GiB against the 5090's 31.35 GiB. It died at step 1 with
`Tried to allocate 12.00 GiB … 1.79 GiB is free`. At device_batch 6 / grad_accum 8 (effective
batch unchanged at 24,576) it ran fine at 22.2 GiB and 0.833 steps/s. **On an H100, bs12 is
the right setting and this note does not apply.**

Verified before launch, from the trainer's own log: **129,970,956 params**
(6·4·256·512·32 = 100,663,296 LUT tables + 592,908 compress/decompress projections
+ 28,714,752 base). Projection FLOPs `H*384*d_c` = 49,152 vs vanilla's `384*384*4` = 589,824,
i.e. **0.0833x**.
