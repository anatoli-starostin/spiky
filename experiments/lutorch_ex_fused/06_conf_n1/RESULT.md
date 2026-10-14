# 06_conf_n1 -- RESULT

**Cartridge:** `FusedConfidenceLUT(read_top_n=1, beta_init=2.0, gamma_init=1.0, read_tau_init=0.5, read_tau_learnable=True)`
(config cartridge=`conf_n1`; seed=1000, weight_init_std=1e-3, table_dropout 0.2). The abl47 confidence
REFERENCE arm (top-1 read). Champion topology (h8/tph64/nap8/d48, 6-block MinimalGPT, n_embd 384), fp32
(TF32 off), cell-TV lambda=10 (separate backward), grad_accum 4, 48000 steps, 68,237,580 params. train.py =
the verified reference trainer (md5 6ebb49ec...).

**Final:** best **val_bpb = 1.0969796** (step 48000; monotone to the end). Final train_loss 3.5941 (ema 3.6116).
Wall 1.99 h (solo), mean ~138 ms/step, peak 7.69 GB. Clean exit, wandb finished (exit 0), 0 NaN/Inf.
Run `lutorch_ex_fused_abl47_conf_n1_1012_231716`.

**Reproducibility (3 seeds):** 1.096980 / 1.096822 / 1.094599 (seeds 1 / 2 / 3; mean 1.09613, sigma 0.001331).
The seed replicates 06_conf_n1_s2 / _s3 (and the init-ablation _oldbias) live in ../aux_runs/. This run is the
Delta-reference ("0") for the whole sweep comparison table in ../README.md.

**Comparison (ALL on the full 170-shard corpus):** conf_n1 (top-1) is beaten by conf_n2 (top-2, 1.086532,
-0.010448), the dense vanilla baseline (00_vanilla, 1.089362) and qconf_n2 (1.089864); it beats its own
quantised twin 07_qconf_n1 (1.099281, +0.0023) and every manifesto arm (1.1095-1.1378).

**NOT comparable to abl47-family numbers** (the old conf_n1 reference 1.11737 was 4-shard/~7-epoch on the
pre-expansion corpus). See ../aux_runs/README.md.
