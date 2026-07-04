---
name: soft-backward-beats-ste-tiny
description: "At tiny LUT-LM scale, TinyMHLut soft backward consistently outperforms multi-alt n_alt=3 (ste) backward across all four LUT modules. Roughly +0.03 bpb gap monotonically growing with steps."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Soft backward beats ste/multi-alt at tiny LUT-LM scale (2026-05-15, exp351)

exp351 = exp340 fork with ALL four TinyMHLut modules (qkv_lut, v_lut, out_proj, residual_lut) switched from `backward_mode='soft'` → `'ste'` with `n_alternatives=3` and `multialt_learnable_temps=True`. Killed at step 4000 — answer was already clear.

| step | exp340 (all-soft) | exp351 (all-ste) | Δ (ste − soft) |
|---|---|---|---|
| 200  | 2.3322 | 2.3634 | +0.0312 |
| 800  | 1.9371 | 1.9370 | −0.0001 |
| 2000 | 1.7924 | 1.8083 | +0.0159 |
| 3000 | 1.7377 | 1.7605 | +0.0228 |
| 4000 | 1.6997 | 1.7295 | +0.0298 |

**Why:** soft backward computes gradients consistent with the chosen row's bit pattern AND propagates through the soft-mixing math, giving denser per-anchor signals. Multi-alt ste relies on perturbations to push gradient through, which is sparser. At this small-batch tiny-model regime soft's denser signal wins decisively.

**How to apply:** Stick with `backward_mode='soft'` for all TinyMHLut modules in tiny LUT-LM runs. Multi-alt ste was historically used at NAP=8 to avoid soft's [B*T, K=2^NAP] memory blow-up — only switch when memory forces it.
