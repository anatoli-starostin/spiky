---
name: exp731-fastmhl-hard-densek-sota
description: "FastMHL(forward='hard', backward='dense_K') trained natively = 1.2178 hard val, new deployment SOTA, beats exp729 (hard+ball NAP+1) by 18.2 mb. Wider K-row bwd helps at native-hard training."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp731 is the new hard-inference SOTA @ 16k bs=24×2

**Final: hard val_bpb = 1.2178 at step 16000, E=384 NAP-bump arch (276.83 M params), 4.89 h wallclock.**

| Run | Recipe | Hard val | vs exp731 |
|---|---|---|---|
| exp724 | TinyMHLut hybrid_smooth fwd + K-row dense bwd | 1.2611 | +43.3 mb |
| exp726 | TinyMHLut NAP+1 fwd + (NAP+1) autograd bwd | 1.3399 | +122.1 mb |
| exp729 | FastMHL hard fwd + NAP+1 ball bwd | 1.2360 | +18.2 mb |
| exp730 | FastMHL hybrid_smooth fwd + NAP+1 ball bwd | 1.2810 | +63.2 mb |
| **exp731** | **FastMHL hard fwd + K-row dense bwd** | **1.2178** | **0 (NEW SOTA)** |

## Trajectory checkpoints (exp731)

| step | val_bpb | lead vs exp729 |
|---|---|---|
| 200 | 2.4056 | (warm-up noise) |
| 4000 | 1.3781 | +20.3 mb |
| 8000 | 1.2808 | +18.6 mb |
| 12000 | 1.2370 | +18.2 mb |
| 16000 | 1.2178 | +18.2 mb |

Lead pinned at +18-20 mb through almost the entire run after warm-up ended at step 1600. No peak-LR pinch.

## Why dense_K wins at native-hard training

At hard fwd, every (b, t) picks exactly one row from K=2^NAP. There's no soft→hard distribution shift either way. The role of backward is to keep LUT rows usefully trained:

- **NAP+1 ball bwd** (exp729): gradient flows through main_idx + the NAP single-bit-flip neighbors. Other rows of the LUT never see gradient signal.
- **K-row dense bwd** (exp731): softmax-weighted gradient through ALL K rows, sharpest mass at main but non-zero everywhere. Every row gets trained.

At E=384 with K=128 (out_proj NAP=7), ball reaches only 8/128 = 6.25% of rows per step. dense_K reaches all 128. Over 16k × 2 (grad_accum) = 32k backward passes, this difference compounds into more useful row training.

## When to use which backward

- **hard fwd + dense_K bwd**: deployment-bound runs targeting hard inference. Best quality at +5-8% wall-clock vs ball.
- **hard fwd + ball NAP+1 bwd**: when memory-constrained (Z_full einsum at K=128 needs ~1.6 GB intermediate) or NAP > 8 (ball wins on speed at larger K).
- **hybrid_smooth fwd + anything**: don't, for deployment-targeted runs. exp730 and exp724 both showed sharp soft pdf → hard inference can't follow → ~+0.07 gap.

## How to apply

- Use `FastMultiHeadLUT(forward_mode='hard', backward_mode='dense_K')` from
  `src/spiky/lutorch/fast_multi_head_lut.py` for any new LUT-LM run targeting hard-mode deployment.
- The K-row dense bwd cost scales with K, so this advice is tested at K∈{64,128}. At NAP≥9 (K=512) the K-wide Z einsum may be memory-prohibitive; revisit ball there.
- exp731's checkpoint at `nanochat_exps/exp731_FastMHL_hard_dense_K/checkpoint.pt` is the current deployment SOTA artefact.

Cross-refs: [[fastmhl-hard-ball-deployment-sota]] (superseded), [[fastmhl-hybrid-smooth-fwd-dispatch]].
