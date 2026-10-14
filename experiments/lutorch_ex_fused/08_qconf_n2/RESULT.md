# 08_qconf_n2 -- RESULT

**Cartridge:** `FusedQuantisedConfidenceLUT(quant_mode="p2_int8", read_top_n=2)` (config cartridge=`qconf_n2`).
Fake-quant (pow2-int8 straight-through) confidence read, top-2 cells; no extra learnable params. Champion
topology, fp32 island, table_dropout 0.2, cell-TV lambda=10, grad_accum 4, 48000 steps, 68,237,586 params.
train.py md5 `f8ded6fe...` (reference + one additive branch).

**Final:** best **val_bpb = 1.0898642** (step 48000; monotone). Final train_loss 3.5845 (ema 3.5914). Wall
3.85 h, mean ~270 ms/step, peak 7.69 GB. Clean exit, wandb exit 0, 0 NaN/Inf. Run `lutorch_ex_fused_abl47_qconf_n2_1014_052618`.

**Comparison (170-shard corpus):**
- vs its full-precision twin 05_conf_n2 (1.086532): quantisation costs **+0.003332** bpb -- slightly more
  quant-sensitive than conf_n1 (+0.0023), consistent with the top-2 read carrying more signal to lose.
- qconf_n2 (1.089864) lands essentially **TIED with the dense vanilla baseline** (00_vanilla, 1.089362;
  delta +0.000502) and still beats full-precision 06_conf_n1 (1.096980) by -0.007116.

**NOT comparable to abl47-family numbers.** See ../aux_runs/README.md.
