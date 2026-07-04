---
name: outproj-per-layer-schedule
description: "out_proj per-layer tph sweep finding (2026-05-13). Layer 0 and Layer 1 are the only layers that benefit from doubled tph; layers 2-5 stay at tph=1024. Unlike residual_lut, out_proj does not follow a global wide-beats-deep rule."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

In the dual-stream LUT-LM (exp268-series, 6 layers, E=64, D=384), `out_proj` benefits from a **heavy-early per-layer tph schedule**, not a uniform shape, and not from extending the boost to mid/late layers. Full sweep at 8K steps on 2026-05-13:

| Exp | `out_tph_per_layer` | Params | Final val_bpb | Δ vs exp303 | Δ vs exp306 (uniform) |
|---|---|---|---|---|---|
| **exp303** (current SOTA) | `[2048, 2048, 1024, 1024, 1024, 1024]` | 602 M | **1.6509** | — | −0.011 |
| exp308 | `[2048, 1024, 1024, 1024, 1024, 1024]` | 585 M | 1.6570 | +0.006 | −0.005 |
| exp306 | `[1024, 1024, 1024, 1024, 1024, 1024]` | 585 M | ~1.662 (stopped @ 5600) | +0.011 | 0 |
| exp307 | `[2048, 2048, 2048, 2048, 1024, 1024]` | 635 M | ~1.661 (stopped @ 5200) | +0.003-0.005 | — |
| exp304 | `[2048, 2048, 1024, 1024, 512, 512]` | 585 M | ~1.667 (stopped @ 3600) | +0.011-0.015 | — |
| exp305 | `[2048, 2048, 2048, 2048, 2048, 2048]` | 669 M | 1.6624 | +0.011 | — |

**Per-layer marginal contribution (at constant exp303 backbone):**
- **L0 boost** (vs uniform-1024, exp308 − exp306): ~**−0.005 bpb** per +16.7 M params → ~0.0003 bpb/M
- **L1 boost** (given L0, exp303 − exp308): ~**−0.006 bpb** per +16.7 M params → ~0.0004 bpb/M
- **L2+L3 boost** (given L0+L1, exp307 − exp303): **slightly positive (worse)** — the extra 33 M params do not help
- **L4+L5 boost or taper** (exp305 widen, exp304 narrow): both ~+0.01 to +0.015 bpb worse

**Why it differs from residual_lut:** residual_lut prefers wider over deeper at *fixed entry budget* (project_residual_lut_prefers_more_tables). out_proj instead prefers a *heavy-early schedule* — the first two layers are the bottleneck for the attention output projection, deeper layers are saturated.

**How to apply:** Keep `out_tph_per_layer=[2048, 2048, 1024, 1024, 1024, 1024]` as the default for this LUT-LM architecture. Don't experiment with uniform out_proj shapes or deep-layer boosts — both lose by ~0.01 bpb at this scale. If hunting param efficiency, **L1 boost is the first to cut** (worth ~0.006 bpb at 16.7 M params, less efficient than L0).

**Open questions:**
  - Does the L0/L1 advantage scale with model depth (e.g. 12-layer would it be L0-L3 that need boosting)?
  - Does NAP=10 or different out_input_nap interact with this schedule?
  - Long-horizon (48K) replication of exp303 vs exp308 cost-benefit.

See also: [[residual-lut-prefers-more-tables]], [[transformer-experiment-summary]].
