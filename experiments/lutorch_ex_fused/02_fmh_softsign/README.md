# 02_fmh_softsign  —  fused cartridge `fmh_softsign`

Cartridge: `FusedManifestoHardLUT(spec, weight="softsign")`
(all: `seed=1000, weight_init_std=1e-3, table_dropout_rate=0.2`; wrapped as
`ProjectionMHL(cart, d_model=384).float()`, replacing every block FFN.)

Old abl47 counterpart: **fss_hard (FusedSoftSignHard)**, old final val_bpb **1.15647**.
(For 05/06 the old run used the *reference* ConfidenceLUT; this is its fused twin — equivalence-tested
to match the reference to tolerance, so expect the same val_bpb within run noise.)

Geometry/schedule: the shared abl47 champion topology (see ../README.md and config.json): 6-block
MinimalGPT, n_embd 384, n_head 6, seq_len 512, vocab 32768, untied unembedder; 48k steps, lr 3e-4
cosine (0.1*lr floor) warmup 0.1, wd 0.1, grad_accum 4, device_batch 12 (24,576 tok/step), seed 1,
eval_every 500; LUTSpec h8/tph64/nap8/d48 pairs; cell-TV λ=10, table_dropout 0.2.

Launch — **ONLINE-or-die wandb** (hard requirement): run as a BARE `python train.py` on a CUDA box
with the lutorch_ex extension built, OUTBOUND NETWORK, and wandb logged in. `wandb.init` runs FIRST
(before data/model/GPU); with no network (e.g. inside the sbox cage) or no wandb login it HARD-FAILS
in seconds — no offline fallback. NOT via `sbox`. NOT launched yet.
    cd 02_fmh_softsign && setsid nohup python train.py > stdout.log 2>&1 &
    # short sanity: SMOKE_STEPS=40 python train.py
Artifacts land here: metrics.csv, summary.json, ckpt.pt (resume), stdout.log.
