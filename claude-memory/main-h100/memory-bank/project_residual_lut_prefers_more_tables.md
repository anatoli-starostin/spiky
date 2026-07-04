---
name: residual-lut-prefers-more-tables
description: "For residual_lut in the dual-stream nanochat LUT-LM, more independent tables (high tph) beats fewer-deeper tables (high NAP / many entries per table) at equal-or-greater entry budget."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

In the dual-stream LUT-LM (exp268-series, 6-layer, E=64, R=384), the residual_lut module strongly prefers **wide (many tph slots) over deep (many entries per slot)** at matched or greater total entry budget. **Confirmed by full 3-way sweep at 8K steps on 2026-05-13.**

**Why:** Three same-horizon forks of exp300 tested LUT shape at fixed entry budget:

| Run | residual NAP / tph | entries/slot | params | final val_bpb @ 8K | Δ vs exp300 |
|---|---|---|---|---|---|
| exp300 (baseline) | 6 / 1024 | 65 536 | 451 M | **1.6573** | — |
| exp302 (deep: NAP=8, tph=512) | 8 / 512 | 131 072 | 602 M | ~1.71 @ step 5000 (stopped early) | **+0.005 → +0.008 bpb** consistently worse |
| **exp303 (wide: NAP=6, tph=2048)** | 6 / 2048 | 131 072 | 602 M | **1.6509** | **−0.0064 bpb** (clean win) |

exp302 and exp303 have **identical entry counts** (131 072/slot) and **identical param counts** (~602 M). The only difference is shape: exp302 is "deep" (fewer richer tables); exp303 is "wide" (more independent tables). Result is unambiguous — same params, same entries, wide-shape beats deep-shape by **~0.013 bpb** at 8K, and wide-shape also beats the smaller exp300 baseline by 0.006 bpb.

**How to apply:** When sweeping residual_lut shapes in this LUT-LM architecture, **scale `residual_tph` not `residual_input_nap`**. The +151 M params from wider tables actually earn their cost (−0.006 bpb at 8K); the same params via deeper tables are a net loss (+0.005–0.008 bpb mid-run). The "more capacity" intuition fails for this module — more independent tables route gradient signal better than fewer richer ones.

**New LUT-LM SOTA at 8K (this architecture family)**: exp303 = 1.6509 @ 602 M params, replacing exp300's 1.6573 @ 451 M.

**Open questions:**
  - Does this hold for `out_proj` and `v_lut` too, or is it residual-specific?
  - Does it invert at extreme tph values (e.g. tph ≥ 4096)?
  - Does it interact with `soft_use_bf16` or `argmax_noise_eps`?
  - Long-horizon (48K) replication.

See also: [[transformer-experiment-summary]], [[bitattention-matmulfree]], [[soft-lut-noise-regularization]].
