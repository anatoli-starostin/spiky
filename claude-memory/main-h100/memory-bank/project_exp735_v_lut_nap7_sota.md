---
name: exp735-v-lut-nap7-sota
description: "exp735 = exp731 with v_input_nap 6 -> 7 = 1.2138 hard val @ 16K, new deployment SOTA, beats exp731 by 4.0 mb at +37.7M params. v_lut benefits from wider K at d_v=64."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp735 is the new hard-inference SOTA @ 16k bs=24×2

**Final: hard val_bpb = 1.2138 at step 16000, 314.6 M params, 5.53 h wallclock.**

| Run | Recipe | Hard val | vs exp735 | Params | Wallclock |
|---|---|---|---|---|---|
| exp731 | FastMHL hard + dense_K, v_lut NAP=6 | 1.2178 | +4.0 mb | 276.8 M | 4.89 h |
| **exp735** | **same recipe + v_input_nap 7** | **1.2138** | **0 (NEW SOTA)** | **314.6 M** | **5.53 h** |

Single architectural change: **v_input_nap 6 -> 7**, doubling v_lut K from 64 to 128.

## Short-horizon (4K) progression that motivated the full run

| Run | NAPs | val @ 4K | params | wallclock |
|---|---|---|---|---|
| exp732 | v=6, resid=6, emb_resid=6 (baseline) | 1.3912 | 276.8 M | 1.225 h |
| exp733 | v=7, resid=6, emb_resid=6 | 1.3857 (-5.5 mb) | 314.5 M | 1.374 h |
| exp734 | v=7, resid=7, emb_resid=7 | 1.3862 (+0.5 vs 733) | 358.6 M | 1.316 h |

exp734 showed that residual + emb_resid LUTs do NOT benefit from wider K
at this scale once v_lut is already at NAP=7 — the gain saturates with
just the v_lut change. Spending another +44 M params on residual stream
LUTs is wasted.

## Why v_lut at NAP=7 helps

v_lut: H=6, tph=256, K, n_out=d_v=64.
- At NAP=6, K=64: 6.29 M params per layer
- At NAP=7, K=128: 12.58 M params per layer (+6.29 M/layer × 6 = +37.7 M)

n_outputs = d_v = 64 < 128, so v_lut continues to use the index_add wgrad
path (the bmm-wgrad dispatch correctly skips it). Only the K-wide
intermediate chain (D, E, F phases) grows.

The wider K gives v_lut twice as many anchor cells per (b, t), letting it
represent more diverse value-projection mappings before the attention sum.
At d_v=64 the per-cell payload is small, so doubling K is param-cheap
compared to widening out_proj (d=384) or residual (D=384).

## Trajectory checkpoints (exp735)

| step | val_bpb | lead vs exp731 |
|---|---|---|
| 200 | 2.3985 | (-7 mb, warmup) |
| 1000 | 1.6785 | -11 mb |
| 2000 | 1.5024 | -7 mb (warmup ends) |
| 4000 | 1.3690 | -9 mb |
| 8000 | 1.2751 | -6 mb |
| 12000 | 1.2331 | -4 mb |
| 16000 | **1.2138** | **-4 mb (final)** |

Lead grew through warmup, peaked at ~10 mb early-training, then steadily
narrowed to ~4 mb by 16K as exp731 had more steps to recover. Net win:
4 mb.

## How to apply

- For any deployment-bound LUT-LM run at E=384 NAP-bump architecture,
  use **v_input_nap=7** (not 6). Other NAPs unchanged: qkv=4, residual=6,
  emb_resid=6, out=7.
- Use `FastMultiHeadLUT(forward_mode='hard', backward_mode='dense_K')`
  for all LUT modules (inherited from exp731).
- Recipe documented at `nanochat_exps/exp735_v_lut_nap7_16k/`.
- Checkpoint at `nanochat_exps/exp735_v_lut_nap7_16k/checkpoint.pt`.

## What was tested and rejected on the way

- exp734 (v=resid=emb_resid all at NAP=7): +44 M extra params, 0 bpb benefit.
  Residual stream LUTs at NAP=6 are already sufficient; wider K there is wasted.

## Open extensions to test

- v_input_nap=8 (K=256, +75 M more): does the v_lut win continue at NAP=8?
  Memory cost roughly doubles again; might be the next worthwhile NAP bump.
- out_input_nap: already at 7. Going to 8 would add 75 M (largest single
  module) — worth testing on a 4K baseline first.
- qkv_input_nap currently 4. Going up unlikely to help since qkv_lut already
  caps at K=16 (very small per-cell).

Cross-refs: [[exp731-fastmhl-hard-densek-sota]] (immediate predecessor),
[[fastmhl-hard-ball-deployment-sota]] (superseded twice now),
[[fastmhl-wgrad-bmm]] (current wgrad optimization).
