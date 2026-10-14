# 07_qconf_n1 -- RESULT

**Cartridge:** `FusedQuantisedConfidenceLUT(quant_mode="p2_int8", read_top_n=1)` (config cartridge=`qconf_n1`).
Fake-quant (pow2-int8 straight-through) confidence read, top-1 cell; no extra learnable params. Champion
topology (h8/tph64/nap8/d48, 6-block MinimalGPT, n_embd 384), fp32 island, table_dropout 0.2, cell-TV lambda=10,
grad_accum 4, 48000 steps, 68,237,580 params. train.py = the verified reference (6ebb49ec...) plus ONE additive
`build_cartridge` branch + the import -> md5 `f8ded6fe...` (by explicit authorization; everything else byte-identical).

**Final:** best **val_bpb = 1.0992805** (step 48000; monotone). Final train_loss 3.6047 (ema 3.6171). Wall
3.77 h, mean ~264 ms/step, peak 7.69 GB. Clean exit, wandb exit 0, 0 NaN/Inf. Run `lutorch_ex_fused_abl47_qconf_n1_1014_052615`.

**Comparison (170-shard corpus):**
- vs its full-precision twin 06_conf_n1 (1.096980): quantisation costs **+0.002301** bpb (about +2.4 sigma) --
  small, as expected for pow2-int8 fake-quant on the top-1 confidence read.
- Behind conf_n2 / vanilla / qconf_n2 (all ~1.086-1.090); it is the weakest of the confidence-family arms.

**NOT comparable to abl47-family numbers.** See ../aux_runs/README.md.
