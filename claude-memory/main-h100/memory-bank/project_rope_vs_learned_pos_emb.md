---
name: project-rope-vs-learned-pos-emb
description: "RoPE beats learned absolute pos_emb by ~0.08 bpb on MinimalGPT @ 8K (exp001 vs exp319). Motivates trying RoPE inside the LUT-LM stack (see [[project_dual_stream_lut_rope_plan]])."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# RoPE vs learned absolute pos_emb — MinimalGPT (vanilla baseline)

**Fact:** Swapping learned absolute `nn.Embedding(seq_len, n_embd)` (additive on tok_emb input) for standard half-rotation RoPE on q,k before SDPA improved val_bpb by **−0.0788** on the nanochat MinimalGPT baseline @ 8000 steps.

| exp | pos encoding | val_bpb @ 8K | params |
|---|---|---|---|
| exp001 | learned absolute (additive) | 1.6256 | 23.4M |
| exp319 | RoPE (base=10000, half-rotation) | **1.5468** | 23.2M |
| | | Δ = **−0.0788** | |

The gap was already visible at step 200 (RoPE −0.015) and stable through the run; not a transient warmup effect.

**Why:** The Anatoli's [[project_lut_bandwidth]] memory tracks Inference bandwidth — RoPE costs zero extra params, the cos/sin tables are buffers. So RoPE is strictly better at this scale (smaller, faster, lower bpb).

**How to apply:** When a vanilla-transformer baseline is needed (e.g. for LUT-LM comparison, or to set a 8K bpb target), use exp319's RoPE recipe, not exp001's learned pos. Always quote the *RoPE-enabled* vanilla bpb (1.5468) when comparing against LUT-LM ablations at the same depth/width/step count.

Implementation reference (exp319_minimal_gpt_rope/train.py):
- `RotaryEmbedding(head_dim=64, max_seq_len=512, base=10000)` instantiated at the Model level; `cos, sin` registered as non-persistent buffers.
- `apply_rope(q, k, cos[:T], sin[:T])` slotted between the qkv→head_dim reshape and `F.scaled_dot_product_attention`. v is NOT rotated.
- Standard half-rotation form: `cos/sin` of shape `[T, head_dim]` built as `torch.cat([freqs, freqs], dim=-1)` paired with `rotate_half(x) = cat([-x2, x1])`.

**Next step (queued, awaiting approval):** apply RoPE inside the LUT-LM attention path (exp303/exp318 LUTBlock — drop additive learned `pos_emb_E`, RoPE on q_vec/k_vec post-q_norm/k_norm before SDPA). d_qk=64 matches MinimalGPT's head_dim so same RoPE module is drop-in. See [[project_dual_stream_lut_rope_plan]] (TODO).
