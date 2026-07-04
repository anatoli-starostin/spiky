---
name: tiny-lut-sota-exp362
description: "Stacking noise=0 + lut_lr=1e-3 on top of bs=96 (exp349) gives tiny-LUT-LM SOTA at 8K: val_bpb=1.4296 @ 43.1M. Both bs=16 findings transferred linearly to bs=96."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Tiny LUT-LM SOTA @ 8K: exp362 = 1.4296 (2026-05-15)

## Configuration
- E=48, D=384, H=6, d_qk=64, d_v=16, 6 layers, ~43.1 M params
- LUTs: qkv_lut (NAP=6, tph=16), v_lut (NAP=8, tph=32), out_proj (NAP=8, tph=128 uniform), residual_lut (NAP=6, tph=64)
- All-soft backward, RoPE, post-norm E-stream + LN before unembed
- bs=96, total_batch=49152, 8000 steps (~393 M tokens)
- **noise=0.0** (per exp352: noise=0 beats noise=0.002 by −0.006 at tiny scale)
- **lut_lr=1e-3** (per exp353: 1e-3 best on LUT param group; non-LUT stays at adam_lr=3e-4)

## Result
- val_bpb = **1.4296** @ step 8000, 1.016 h on H100
- Δ vs exp349 (same shape, default noise+LR) = **−0.0182**

## Why this matters
Both bs=16 optimizer findings (noise=0, lut_lr=1e-3) **transfer linearly** to bs=96 — sum of individual gains ≈ 0.013 at bs=16, observed 0.018 at bs=96. Suggests they're orthogonal effects, not redundant or saturating.

## Tiny-LUT-LM SOTA progression today (2026-05-15)
| Exp | Settings | val_bpb | Params |
|---|---|---|---|
| exp340 | bs=16, noise=0.002, lr=3e-4 | 1.6366 | 43.1 M |
| exp348 | + bs=48 | 1.5234 | 43.1 M |
| exp349 | + bs=96 | 1.4478 | 43.1 M |
| **exp362** | + noise=0 + lut_lr=1e-3 | **1.4296** | 43.1 M |

Gap to vanilla bs=16 (exp328 = 1.3882 @ 23.2 M) = **0.0414** with 1.86× params.

## Use this config as base for further forks
For any future tiny-LUT-LM fork, start from exp362's recipe (bs=96 + noise=0 + split lut_lr=1e-3) unless explicitly varying one of those axes.
