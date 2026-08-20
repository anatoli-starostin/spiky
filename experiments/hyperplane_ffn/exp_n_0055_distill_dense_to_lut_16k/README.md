# exp_n_0055 — distillation test: can a LUT imitate the dense FFN's real activations?

> **PHASE-1 RESULT: raw frozen swap-in val_bpb = 1.3553413 (16k, 0.76 h). A distilled LUT does NOT reproduce the
> dense FFN when frozen in.** +0.1587 vs dense (1.196646), and notably **+0.1268 WORSE than END-TO-END LUT
> training exp_n_0052 (1.2285517)**. Per-block imitation MSE plateaus at non-zero floors that grow with depth:
> b0 0.0025, b1 0.0037, b2 0.0082, b3 0.0097, b4 0.0140, **b5 0.0263** (last FFN ~10× the first). So a LUT of this
> capacity cannot fully represent the dense FFN (non-zero MSE), AND those imitation errors compound badly when the
> rest of the network is frozen and can't co-adapt — which is why frozen-distill (1.355) is worse than
> co-adapted end-to-end LUT (1.229). **Phase 2** (finetune attention+unembedder with the LUTs frozen) decomposes
> how much of that 1.355→1.229 gap is pure co-adaptation vs irreducible LUT-function error.

## Setup
- **Frozen dense baseline:** exp073 (tied vanilla dense FFN, 1.1966461, 23.2M params) loaded from its checkpoint
  and frozen. Verified on load: reproduces val_bpb 1.1966461 exactly.
- **Targets:** for each of the 6 blocks, a forward hook on `block.mlp` captures the FFN sublayer INPUT
  (`h = ln2(x)`) and OUTPUT (`mlp(h)`, pre-residual), both `[N, 384]`.
- **LUTs:** one `CompressionMultiHeadLUT` per block, hyperparameters reproduced **1:1 from exp_n_0052** (batched
  control): H8 / d48 / tph64 / nap6, `joint_head_compression=false`, **`batched_multi_head_input=true`**,
  hard forward, `use_bf16=false`, `initial_weights_noise=0.001`, `learnable_temps=true`, seed `1000+block`.
  Per-LUT 1,868,546 params; 6 LUTs = 11,211,276 trainable.
- **Recipe:** the STANDARD hard-forward / soft-backward (STE) path (NOT soft-forward). Regress input→output with
  `MSE`. AdamW (0033 grouping: LUT tables+temps→nodecay wd0, compress/decompress→decay wd0.1), lr 3e-4, cosine
  decay + 10% warmup, **16000 steps** (matched to the standard rung).

## Online streaming distillation
Each step runs a fresh training batch through the frozen dense model; the 6 hooks capture all blocks' (in,out)
simultaneously, and each block's LUT trains on `MSE(lut(in), out.detach())` (grads only into the LUTs, one
backward per block to keep the STE surrogate memory bounded). This streams `n_steps × B×T ≈ 393M` token-vectors
per block (vs caching a fixed ~multi-GB dataset) — a stronger test (no memorization) that also matches the
standard-rung token budget. Shared modules `fast_multi_head_lut.py` / `compression_mhl.py` untouched.

## Eval
After training, all 6 LUTs are wrapped in a `LUTAdapter` (`[B,T,C]→reshape→CMHL→[B,T,C]`) and swapped into the
frozen dense model in place of `block.mlp`; whole-model val_bpb is measured on the clean 245,760-token val set
(eval_steps 10 × bs 48 × seq 512). Deliverables: per-block learning curves (MSE vs step), swap-in val_bpb +
delta vs 1.196646, per-block final MSE.

## Smoke (300 steps, full model size)
Dense loads and reproduces 1.1966461 exactly; hooks capture all 6 blocks; all 6 LUTs train with MSE dropping
(b0 0.056→0.012 … b5 0.220→0.070); swap-in+eval produces a sane bpb (1.525, undertrained). **Block 5 fits
notably worse** than blocks 0–4 (final-layer FFN is harder for a LUT to imitate) — one to watch in the full run.
