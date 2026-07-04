---
name: project-rope-lut-lm
description: "RoPE on q,k inside LUT-LM attention beats additive per-layer learned pos_emb by ~0.058 bpb @ 8K (exp321 = 1.5933 vs exp303 = 1.6509). New LUT-LM SOTA at 8K. Gap to vanilla+RoPE is still ~0.046 bpb."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# RoPE inside LUT-LM (2026-05-14, exp321)

**Fact:** Replacing the additive per-layer learned `pos_emb` (dim E=64, added to qk_joint LUT input) with standard half-rotation RoPE applied on (q, k) post-q_norm/k_norm right before SDPA gives **−0.0576 bpb** on the 8K nanochat LUT-LM benchmark (601.9 M params).

| run | pos encoding | residual_lut | val_bpb @ 8K | params |
|---|---|---|---|---|
| exp303 (prior SOTA) | additive learned per-layer | dense | 1.6509 | 602.1 M |
| **exp321** | **RoPE on q,k (base=10000)** | dense | **1.5933** | 601.9 M |
| exp319 | RoPE on q,k (MinimalGPT) | n/a | 1.5468 | 23.2 M |

The gap is consistent across the entire training run, not a final-step artifact. Already visible at step 200 (Δ −0.091) and stable through step 8000 (Δ −0.058).

**Implementation** (`nanochat_exps/exp321_dense_rope/train.py`):
- `RotaryEmbedding(head_dim=d_qk=64, max_seq_len=context_size=512, base=10000)` at the Model level; `cos, sin` registered as non-persistent buffers.
- `apply_rope(q, k, cos[:T], sin[:T])` called between the q/k reshape-to-[B,H,T,d_qk] and `F.scaled_dot_product_attention`. v is NOT rotated.
- `LUTBlock` drops the additive `pos_emb` term — `qk_joint(x_flat)` runs on raw x.
- All other hyperparameters identical to exp303.

**Why RoPE matters more for LUT-LM than for the vanilla transformer (qualitative):** the additive learned pos_emb feeds the qk_joint LUT input, which means position information has to survive being argmax-quantized through a 6-anchor-pair LUT to reach q and k. RoPE injects position AFTER the LUT, into a smooth float space where SDPA already lives — no quantization loss on the positional signal.

**Capacity gap to vanilla + RoPE:** exp319 (23 M, vanilla MinimalGPT + RoPE) still beats exp321 (602 M) by **+0.0465 bpb**. The LUT-LM dual-stream architecture has a structural disadvantage vs an FFN-equipped vanilla transformer at the same training budget. Closing this gap is the open problem; RoPE was a "free" gain that helped both. See [[project_rope_vs_learned_pos_emb]] for the vanilla baseline.

**How to apply:**
- All new LUT-LM forks should use RoPE on q,k post-q_norm/k_norm. Drop the additive per-layer pos_emb pattern entirely.
- For models with `d_qk != 64`, RoPE still works as long as `d_qk` is even.
- When porting a vanilla baseline to compare LUT-LM, use the RoPE recipe from `nanochat_exps/exp319_minimal_gpt_rope/train.py` (parameter-free; cos/sin are non-persistent buffers).

**Next step (open):** capacity-vs-vanilla gap is +0.045 bpb at 26× more LUT-LM params. Approaches worth trying:
- joint qkv LUT (exp322): share lookup tables across q,k,v projections; if it works, halve qk+v params (or use the savings for more capacity elsewhere).
- bigger residual_lut now that RoPE has shaved 0.05 bpb off the floor.
- FFN injection inside LUT-LM blocks (small dense MLP between attention and residual_lut).
