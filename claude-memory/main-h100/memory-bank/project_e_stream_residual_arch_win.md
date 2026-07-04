---
name: e-stream-residual-arch-win
description: "exp418 — adding an E-stream residual (x_lut + out_proj output) with GPT-2 style ln_pre/ln_post to the exp365 LUT-LM saves 60 mb at matched compute (43.06M, bs=16, 8K). Big arch win that should apply to all LUT-LM forks."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# E-stream residual + pre/post LN — 60 mb arch win (2026-05-17, exp418)

## Motivation
Routing entropy analysis of exp364's checkpoint (`routing_entropy.csv`) showed:
- L5 out_proj: 97% of rows dead, mean entropy 0.61 bits (out of 8), top-1 mass 87% → near-total collapse
- L4 out_proj: 35% dead, mean H=4.45 — also collapsing
- L5 residual_lut: 29% dead, mean H=4.80
- Earlier layers (L0–L3): mostly healthy (87–95% utilization)

Hypothesis: late-layer collapse is driven by a missing **E-stream residual**. In the prior exp365/exp364 design:
```
x_lut_next = LN(out_proj(SDPA(qkv_lut(x_lut))))   # x_lut REPLACED, no residual
```
Every layer is a hard-argmax bottleneck in the forward path. Gradients to early layers must traverse all subsequent LUTs (≥12 argmax operations for L0 in a 6-layer model) → vanishing gradient signal → late-layer routing collapses (locked) while early layers can't escape bad init.

## Change (exp418 vs exp365/exp364)
```python
# Per LUTBlock:
self.ln_pre  = nn.LayerNorm(E)       # pre-norm before qkv/v_lut input
self.ln_post = nn.LayerNorm(E)       # post-norm on the residual sum, fed to residual_lut

# Forward:
x_pre  = self.ln_pre(x_flat)
qkv_out = self.qkv_lut(x_pre); ...
v_lut_out = self.v_lut(x_pre); ...
out_e = self.out_proj(SDPA(q,k,v))
x_lut_next = x_flat + out_e          # ← E-stream RESIDUAL (was: out_e_norm replacing x)
r_in   = self.ln_post(x_lut_next)
r_out  = self.residual_lut(r_in)
```

Two new LayerNorms per block (576 extra params total over 6 layers) and one new add op. Removed the `ln_e` that previously wrapped out_e directly.

## Result
| Run | Arch | bs | Steps | Params | Final bpb |
|---|---|---|---|---|---|
| exp365 | exp365 baseline | 16 | 8K | 43.06M | 1.6215 |
| **exp418** | **+ E-residual + ln_pre/post** | **16** | **8K** | **43.06M** | **1.5611** |

**Δ = −60.4 mb** at identical compute, params, and batch size.

## Trajectory
| Step | exp418 | exp365 | Δ (mb) |
|---|---|---|---|
| 200 | 2.2895 | 2.2975 | −8 |
| 600 | 1.9426 | 1.9665 | −24 |
| 1000 | 1.8372 | 1.8786 | −41 |
| 1400 | 1.7784 | 1.8288 | −50 |
| 2000 | 1.7225 | 1.7804 | −58 |
| 3000 | 1.6675 | 1.7264 | −59 |
| 4000 | 1.6297 | 1.6904 | −61 |
| 6000 | 1.5818 | 1.6418 | −60 |
| 8000 | **1.5611** | **1.6215** | **−60** |

Gap opens by step 600 and is stable around −60 mb from step 1000 onwards.

## How to apply
- **Add the E-residual to ALL future LUT-LM forks**. It's nearly free (576 extra params, 1 add op per layer) and saves 60 mb at bs=16. Likely larger savings at lower batch sizes (where gradient flow matters more).
- **Use GPT-2 pre-norm + post-norm structure**: ln_pre before the qkv/v_lut input, ln_post after the residual sum before residual_lut. The original `ln_e` wrapping out_proj output is replaced.
- The previous `q_norm`/`k_norm` (post-qkv_lut for q,k) stays — those normalize per-head SDPA queries/keys and are still needed since LUT outputs aren't bounded.
- exp418's arch should become the new baseline; rerun the bandwidth-quality U-curve, batch-scaling, and architecture sweeps with it to confirm the savings generalize.
- The 60-mb improvement is roughly the same magnitude as bs=16 → bs=32 batch scaling. Cheaper to get via architecture than via more compute.

## Outstanding tests
1. Will exp418's improvement compose with bs=64/128/192 scaling? (probably yes; needs verification)
2. Does the E-residual reduce L5 out_proj's routing collapse? (run entropy analysis on exp418 checkpoint)
3. Does it help even more if combined with `lut_lr=1e-3` or noise=0.002 (current exp418 has lut_lr=adam_lr=3e-4 since it's a direct exp364 fork — see config.json)?

## Notes
- exp418's config retains exp364's choices including `out_tph_per_layer` and `out_proj_multi_nap` (not set), so out_proj is a single TinyMHLut at NAP=8 tph=128 per layer.
- 0.37h training time at bs=16, 8K steps. Two LayerNorms are negligible cost.
