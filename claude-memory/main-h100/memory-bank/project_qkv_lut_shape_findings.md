---
name: qkv-lut-shape-findings
description: "qk_joint and v_lut tph doubling are both net-negative on top of exp303 SOTA, but v widening hurts much less than qk widening. Differs from residual_lut which benefits from widening."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

In the dual-stream LUT-LM (exp268-series, 6-layer, E=64, D=384), doubling `tph` on the **attention input LUTs** (qk_joint, v_lut) at fixed NAP is net-negative on top of the exp303 SOTA. The harm gradient is striking: **qk widening hurts ~2× as much as v widening at the same param cost**.

| Run | Change | Params | Final val_bpb | Δ vs exp303 |
|---|---|---|---|---|
| exp303 (SOTA) | — | 602 M | **1.6509** | — |
| **exp310** | `v_tph` 256 → 512 (NAP=8) | 678 M (+76) | **1.6556** | +0.005 |
| **exp309** | `qk_tph` 256 → 512 (NAP=6) | 678 M (+76) | **1.6615** | +0.011 |

Both add the same param count (~+76 M). v_lut widening costs ~0.005 bpb, qk_joint widening costs ~0.011 bpb. Trajectories were clearly distinguishable from step 1400 onward: exp310 trailed by ~+0.003-0.005, exp309 trailed by ~+0.007-0.011 throughout mid/late training.

**Contrast with [[residual-lut-prefers-more-tables]]:** residual_tph 1024 → 2048 *helped* by −0.006 bpb at +151 M (exp303 vs exp300). The wide-beats-deep rule does NOT generalize to QKV LUTs — residual_lut is the only module in this architecture that benefits from `tph` scaling.

**How to apply:**
- Keep `qk_tph=256, qk_input_nap=6, v_tph=256, v_input_nap=8` as defaults.
- If you must spend +76 M, the cheapest place is v_lut (smallest penalty observed); residual_tph already scaled, out_proj saturated.
- Param-efficiency ranking for spending the next +N M on exp303-family (8K horizon):
  1. `residual_tph` widening — *helpful* (~−0.006 bpb / +151 M)
  2. `out_proj` L0/L1 boost — *helpful* (~−0.005-0.006 bpb / +16.7 M each, best per-M)
  3. `v_tph` widening — mildly harmful (~+0.005 bpb / +76 M)
  4. `qk_tph` widening — clearly harmful (~+0.011 bpb / +76 M)
  5. `out_proj` everywhere widening / tapering — strongly harmful

**Open:**
  - Does `v_input_nap` shrink (8 → 6) free up budget cheaply?
  - Long-horizon (48K) replication: do the small +0.005 penalties stay or shrink?

See also: [[residual-lut-prefers-more-tables]], [[outproj-per-layer-schedule]], [[transformer-experiment-summary]].
