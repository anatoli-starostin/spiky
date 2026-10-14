# 05_conf_n2 -- RESULT

**Cartridge:** `FusedConfidenceLUT(read_top_n=2)` (config cartridge=`conf_n2`). Fused CUDA confidence
read, top-2 cells. Geometry: 6-block MinimalGPT, n_embd 384, h8/tph64/nap8/d48, r=384=n_embd,
anchor_mode="pairs", fp32 (TF32 off), table_dropout 0.2, cell-TV lambda=10 (separate backward), grad_accum 4,
48000 optimizer steps, 68,237,586 params. train.py byte-identical to the verified 06_conf_n1 trainer (md5 6ebb49ec...).

**Final:** best **val_bpb = 1.0865318** (step 48000; monotone to the end). Final logged train_loss 3.5749
(ema 3.5818). Wall time 3.60 h, mean ~253 ms/step, peak 7.69 GB. Clean exit, wandb finished (exit 0),
0 NaN/Inf. Run `lutorch_ex_fused_abl47_conf_n2_1014_014318`.

**Comparison (ALL on the full 170-shard corpus -- comparable to each other):**
- This run (conf_n2) 1.0865318 is the **BEST arm of the whole sweep**. It beats the top-1 reference
  06_conf_n1 (1.096980) by **-0.010448** (about -7.2 sigma below the conf_n1 seed cluster) -- adding the
  second confidence cell is a real, large win, and it FLIPS the earlier "n=1 is best" reading.
- It also edges the dense vanilla baseline (00_vanilla, 1.089362) by 0.002830 and its quantised twin
  (08_qconf_n2, 1.089864) by 0.003332.

**NOT comparable to abl47-family numbers** (abl47 trained ~7 epochs over only 4 shards, pre-expansion
corpus; this trains <1 epoch over the 170-shard corpus downloaded 2026-10-07). See ../aux_runs/README.md.
