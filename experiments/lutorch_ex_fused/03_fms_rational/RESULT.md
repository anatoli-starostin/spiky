# 03_fms_rational — RESULT

**Cartridge:** `FusedManifestoSoftLUT(weight="rational")` (config cartridge=`fms_rational`). Fused CUDA manifesto
soft-read, rational neighbour-weight. Geometry: 6-block MinimalGPT, n_embd 384, h8/tph64/nap8/d48, r=384=n_embd,
anchor_mode="pairs", fp32 (TF32 off), table_dropout 0.2, cell-TV λ=10 (separate backward), grad_accum 4, 48000
optimizer steps, 68,237,568 params. train.py byte-identical to the verified 06_conf_n1 trainer (md5 6ebb49ec…).

**Final:** best **val_bpb = 1.1186331** (step 48000; last evals 47000=1.1192, 47500=1.1189, 48000=1.1186).
Final logged train_loss 3.663. Wall time 3.83 h, mean ~264 ms/step, peak 7.69 GB. Clean exit, wandb finished
(exit 0), 0 NaN/Inf. Run `lutorch_ex_fused_abl47_fms_rational_1013_211158`.

**Comparison (ALL on the full 170-shard corpus — comparable to each other):**
- 06_conf_n1 (FusedConfidenceLUT read_top_n=1, reference): 1.096980 (3-seed cluster mean ≈ 1.09613, σ ≈ 0.001331).
- This run (fms_rational) 1.1186331 is **+0.0217 vs 06_conf_n1**, ≈ +16.9σ above the conf_n1 seed cluster.
- Soft-vs-Hard (rational): soft (this, 1.1186) beats hard (01_fmh_rational, 1.1324) by 0.0138 — the soft read helps.

**NOT comparable to abl47-family numbers.** abl47 (2026-10-04) trained ~7 epochs over only 4 shards (~170M
tokens, pre-expansion corpus); every run here trains <1 epoch over ~27 unique shards of the 170-shard corpus
downloaded 2026-10-07. Cross-corpus comparison is invalid (see ../aux_runs/README.md).
