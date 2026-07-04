---
name: exp408-smaller-dims-cost
description: exp408 — shrinking E=48→32 and d_v=16→8 in exp407 arch costs ~33 mb bpb at bs=16. Quality drops faster than param savings.
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp408 — smaller E and d_v at bs=16 (2026-05-17)

## Setup
Fork of exp407 (cleaned LUT-LM base). Changed E=48→32 and d_v=16→8. All other dims unchanged (D=384, d_qk=64, H=6, num_layers=6). bs=16, 8K steps, soft backward.

## Result
| Run | Params | LUT params | Final bpb | Time | Δ vs exp407 |
|---|---|---|---|---|---|
| exp407 (E=48, d_v=16) | 43.06M | 28.9M | 1.6038 | 0.225h | reference |
| **exp408 (E=32, d_v=8)** | **36.74M** | **23.1M** | **1.6372** | 0.212h | **+0.0334 bpb** |

Trade: −14.7% params (−6.3M), +33.4 mb bpb. Quality drops faster than params; pretty bad rate.

## Trajectory (gap was monotone in middle and late phases)
| Step | exp408 | exp407 | Δ (mb) |
|------|--------|--------|--------|
| 200 | 2.2942 | 2.2957 | −1.5 |
| 400 | 2.0660 | 2.0544 | +11.6 |
| 1000 | 1.8934 | 1.8675 | +25.9 |
| 2000 | 1.7881 | 1.7626 | +25.5 |
| 4000 | 1.7024 | 1.6727 | +29.7 |
| 6000 | 1.6568 | 1.6239 | +32.9 |
| 8000 | **1.6372** | **1.6038** | **+33.4** |

Gap opened by step 400 and stayed in the +25 to +34 mb band throughout. No convergence in late phase.

## Conclusion
Cutting E and d_v in half is **not free**. At bs=16, the smaller dims cost +33 mb for 14.7% params saved — bad bpb-per-param trade.

Hypothesis: E=48 may already be at or below the bottleneck for the x_lut carry; shrinking it further (E=32) loses representational capacity faster than the param count drops. d_v=8 also halves attention head width to 8 elements per head, which may be too narrow for the SDPA softmax to discriminate.

## How to apply
- Don't shrink E below 48 or d_v below 16 in the LUT-LM base architecture at bs=16. The quality cost outpaces the param savings.
- If memory budget is the constraint, prefer trimming `out_tph` or `residual_tph` (these dominate per-layer LUT params) over E/d_v.
- For a future ablation: test E=64 (slightly bigger) — would tell us if E=48 is already saturating.
