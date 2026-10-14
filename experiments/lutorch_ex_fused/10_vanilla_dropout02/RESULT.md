# 10_vanilla_dropout02 -- RESULT

**FFN:** dense MLP (Linear 384->1536 -> GELU -> nn.Dropout(0.2) -> Linear 1536->384, bias=False) -- the
00_vanilla baseline FFN with a STANDARD dropout INSIDE the MLP, right after the GELU (on the 1536-dim hidden
units, before the down-projection); attention untouched. The dense analogue of the fused arms table dropout
(which acts inside the FFN substitute). ffn_type=dense, dropout 0.2. Everything else byte-identical to 00_vanilla
(shared trainer md5 065358e5...): 6-block MinimalGPT, RoPE+SDPA, untied head, fp32/TF32 off, 48000 steps,
grad_accum 1, device_batch 48, 24,576 tok/step, seed 1, vendored fixed_eval, ckpt 2000. 35,792,640 params
(dropout adds none).

**Final:** best **val_bpb = 1.1054547** (step 48000; monotone). Final train_loss 3.6670 (ema 3.6774). Wall
1.76 h (solo), mean ~121 ms/step, peak 14.55 GB. Clean exit, wandb finished (exit 0), 0 NaN/Inf.
Run `lutorch_ex_fused_abl47_vanilla_drop02_1014_115545`.

**Headline (controlled A/B vs 00_vanilla):** 10 differs from 00_vanilla ONLY by the dropout -- same seed 1,
same init (dropout has no params), same data order (loader deterministic from token 0), same 35,792,640 params.
Result: dropout 0.2 **HURTS** by **+0.016093** bpb (1.105455 vs 00_vanilla 1.089362). Expected and informative:
at ~1.18B tokens (<1 epoch of the 170-shard corpus) the model is data-bound, not overfitting, so dropout
removes signal rather than curbing overfit. This also suggests the fused arms table dropout 0.2 is, at best,
neutral here. The p=0.1 twin 09_vanilla_dropout01 is prepared + committed but was NOT run.

**NOT comparable to abl47-family numbers.** See ../aux_runs/README.md.
