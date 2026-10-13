# 04_fms_softsign — RESULT

**Cartridge:** `FusedManifestoSoftLUT(weight="softsign")` (config cartridge=`fms_softsign`). Fused CUDA manifesto
soft-read, soft-sign neighbour-weight. Geometry: 6-block MinimalGPT, n_embd 384, h8/tph64/nap8/d48, r=384=n_embd,
anchor_mode="pairs", fp32 (TF32 off), table_dropout 0.2, cell-TV λ=10 (separate backward), grad_accum 4, 48000
optimizer steps, 68,237,580 params. train.py byte-identical to the verified 06_conf_n1 trainer (md5 6ebb49ec…).

**Final:** best **val_bpb = 1.1095368** (step 48000; last evals 47000=1.1103, 47500=1.1098, 48000=1.1095).
Final logged train_loss 3.632. Wall time 3.95 h, mean ~274 ms/step, peak 7.69 GB. Clean exit, wandb finished
(exit 0), 0 NaN/Inf. Run `lutorch_ex_fused_abl47_fms_softsign_1013_211158`.

**Comparison (ALL on the full 170-shard corpus — comparable to each other):**
- 06_conf_n1 (FusedConfidenceLUT read_top_n=1, reference): 1.096980 (3-seed cluster mean ≈ 1.09613, σ ≈ 0.001331).
- This run (fms_softsign) 1.1095368 is **+0.0126 vs 06_conf_n1**, ≈ +10.1σ above the conf_n1 seed cluster — the
  BEST of the four manifesto arms and the closest non-confidence cartridge to the conf_n1 reference.
- Soft-vs-Hard (softsign): soft (this, 1.1095) beats hard (02_fmh_softsign, 1.1378) by 0.0283 — soft read helps a lot.
- Note the weight ordering flips with the read: under HARD, rational < softsign (01<02); under SOFT, softsign < rational
  (04<03). So fms_softsign is the strongest manifesto variant here.

**NOT comparable to abl47-family numbers.** abl47 (2026-10-04) trained ~7 epochs over only 4 shards (~170M
tokens, pre-expansion corpus); every run here trains <1 epoch over ~27 unique shards of the 170-shard corpus
downloaded 2026-10-07. Cross-corpus comparison is invalid (see ../aux_runs/README.md).
