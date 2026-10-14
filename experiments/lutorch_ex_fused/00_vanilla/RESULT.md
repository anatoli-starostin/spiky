# 00_vanilla -- RESULT

**FFN:** dense MLP (`Linear(384->1536) -> GELU -> Linear(1536->384)`, no bias) replacing the LUT cartridge --
the parameter-light baseline for the fused sweep (ffn_type=`dense`). Same backbone/recipe as the LUT arms
(6-block MinimalGPT, RoPE + SDPA, untied head, fp32/TF32 off, 48000 steps, grad_accum 4, 24,576 tok/step,
seed 1, vendored fixed_eval) but NO cell-TV / table-dropout / faithful-init (not applicable to a dense FFN).
35,792,640 params (~52% of the LUT arms 68.24M). train.py md5 `c2703f78...` (the reference with the LUT
machinery removed and the dense MLP added; see its header).

**Final:** best **val_bpb = 1.0893617** (step 48000; monotone). Final train_loss 3.5502 (ema 3.5536). Wall
3.21 h, mean ~222 ms/step, peak 14.32 GB (dense activations exceed the LUT island). Clean exit, wandb exit 0,
0 NaN/Inf. Run `lutorch_ex_fused_abl47_vanilla_1014_021241`. Reproduces the arch/recipe of the canonical anchor
`exp_n_0216_vanilla48k_gelu_ckpt2k_seed1` (val_bpb 1.1151 on the OLD 4-shard corpus) on the 170-shard corpus.

**Comparison (170-shard corpus) -- the headline finding:**
- The dense baseline (1.0893617, HALF the params) is the **2nd-best run of the entire sweep**, beaten only by
  full-precision conf_n2 (1.086532, by 0.002830). It BEATS 06_conf_n1 (1.096980) by -0.007618 and both
  quantised arms.
- So on this corpus/topology, **only the top-2 confidence LUT (conf_n2) clears a parameter-light dense MLP**;
  conf_n1, the quantised arms, and every manifesto variant do not.

**NOT comparable to abl47-family numbers.** See ../aux_runs/README.md.
