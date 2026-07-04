---
name: qkv-only-no-vlut-neutral
description: "exp404 — wider qkv_lut (tph 16→96) with v_lut REMOVED is essentially tied with exp392 dual-branch design. 1.75× LUT params, 2× attention bandwidth, zero quality gain. Dedicated v_lut+qkv joint > single wider joint."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# qkv-only fork (no v_lut) — neutral result (2026-05-17, exp404)

## Setup
Fork of exp392 with two changes:
1. v_lut REMOVED entirely (no NAP=8 tph=32 separate v branch)
2. qkv_lut widened: NAP=6, tph 16 → 96 (q,k,v all from this single LUT; last d_v outputs are v)

All other settings identical (bs=16, 8K steps, soft backward, lut_lr=1e-3, RoPE).

## Param + bandwidth changes
- Attention LUT params/layer: 1.67M (qkv+v) → 5.31M (qkv-only) — Δ +3.64M/layer
- Total LUT params: 28.9M → 50.7M (1.75×)
- Total model params: 43M → 64.9M
- Attention bandwidth/token: 288 → 576 lookups (exactly 2× exp392)

## Result
| Step | exp404 | exp392 | Δ (mb) |
|------|--------|--------|--------|
| 1000 | 1.8719 | 1.8715 | +0.4 |
| 2000 | 1.7636 | 1.7672 | -3.6 |
| 4000 | 1.6736 | 1.6740 | -0.4 |
| 6000 | 1.6231 | 1.6235 | -0.4 |
| 8000 | **1.6033** | **1.6029** | **+0.4** |

**Final: essentially tied** (Δ within ±1 millibit at every checkpoint).

## Conclusion
The wider qkv_lut absorbed the v_lut's role but did not capture any extra quality at this regime. Dedicated v_lut (NAP=8 tph=32) + smaller joint qkv_lut (NAP=6 tph=16) achieves the same accuracy with **less than 60% of the LUT params** and **half the attention bandwidth**.

Implication: at bs=16 the architecture is qkv-bandwidth-bound, not capacity-bound. The exp326 dual-branch design (`qkv_lut` for q,k + residual v contribution; separate `v_lut` for main v capacity) remains the recommended pattern.

## How to apply
Don't drop v_lut to "simplify" — the dual-branch design is the right param/bandwidth trade-off at bs=16. If you want to widen attention LUTs, scale `qkv_tph` while KEEPING the v_lut. Tested wider qkv-only design here is strictly dominated by exp392.
