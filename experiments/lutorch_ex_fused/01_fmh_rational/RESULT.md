# 01_fmh_rational — RESULT

**Cartridge:** `FusedManifestoHardLUT(weight="rational")` (config cartridge=`fmh_rational`). Fused CUDA manifesto
hard-read, rational neighbour-weight. Geometry: 6-block MinimalGPT, n_embd 384, h8/tph64/nap8/d48, r=384=n_embd,
anchor_mode="pairs", fp32 (TF32 off), table_dropout 0.2, cell-TV λ=10 (separate backward), grad_accum 4, 48000
optimizer steps, 68,237,568 params. train.py byte-identical to the verified 06_conf_n1 trainer (md5 6ebb49ec…).

**Final:** best **val_bpb = 1.1324248** (step 48000; last evals 47000=1.1330, 47500=1.1328, 48000=1.1324).
Final logged train_loss 3.697. Wall time 3.69 h, mean ~258 ms/step, peak 7.69 GB. Clean exit, wandb finished
(exit 0). Run `lutorch_ex_fused_abl47_fmh_rational_1013_165608`.

**Comparison (ALL on the full 170-shard corpus — same corpus, so these ARE comparable to each other):**
- 06_conf_n1 (FusedConfidenceLUT read_top_n=1, the reference): val_bpb **1.096980**.
- 3-seed conf_n1 cluster: 1.0969795755941025 / 1.0968218702576613 / 1.0945992497236325 (mean ≈ 1.09613, sample σ ≈ 0.001331).
- This run (fmh_rational) 1.1324248 is **+0.0355 vs 06_conf_n1**, ~27σ above the conf_n1 seed cluster — i.e. the
  hard manifesto arm is clearly worse than the confidence arm at this geometry, consistent with the historical
  ordering (confidence < manifesto-hard).

**NOT comparable to abl47-family numbers.** abl47 (2026-10-04) trained ~7 epochs over only 4 shards (~170M
tokens, the pre-expansion corpus); every run here trains <1 epoch over ~27 unique shards of the 170-shard corpus
downloaded 2026-10-07. Cross-corpus comparison is invalid (see ../aux_runs/README.md).
