---
name: e-vs-dv-attribution
description: "exp407/exp408/exp409 ablation — at bs=16, shrinking d_v hurts more per param than shrinking E. E=48→32 alone costs +10.6 mb (~3 mb/M params lost); adding d_v=16→8 costs +22.8 mb more (~8.6 mb/M params, 3× worse rate)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# E vs d_v attribution ablation (2026-05-17, exp407 / exp408 / exp409)

## Three-way comparison at bs=16
| Run | E | d_v | Params | Final bpb | Δ vs exp407 | Δ params |
|---|---|---|---|---|---|---|
| exp407 | 48 | 16 | 43.06M | **1.6038** | reference | — |
| **exp409** | **32** | **16** | **39.39M** | **1.6144** | **+10.6 mb** | **−3.67M (−8.5%)** |
| exp408 | 32 | 8 | 36.74M | 1.6372 | +33.4 mb | −6.32M (−14.7%) |

## Attribution
- **E=48→32 alone** (exp407→exp409): **+10.6 mb** at −3.67M params → **2.9 mb / M params lost**
- **d_v=16→8 on top** (exp409→exp408): **+22.8 mb** at −2.65M extra → **8.6 mb / M params lost** (3× worse rate)
- **Combined** (exp407→exp408): +33.4 mb at −6.32M

**d_v dominates the cost.** The attention-head dimension matters more than the LUT carry-stream width per param trimmed.

## Trajectory comparison (gap monotone throughout)
| Step | exp407 | exp409 (E↓) | exp408 (E↓ + dv↓) |
|------|--------|--------|--------|
| 1000 | 1.8675 | 1.8752 (+8 mb) | 1.8934 (+26 mb) |
| 2000 | 1.7626 | 1.7737 (+11 mb) | 1.7881 (+25 mb) |
| 4000 | 1.6727 | 1.6868 (+14 mb) | 1.7024 (+30 mb) |
| 6000 | 1.6239 | 1.6347 (+11 mb) | 1.6568 (+33 mb) |
| 8000 | **1.6038** | **1.6144 (+11 mb)** | **1.6372 (+33 mb)** |

exp409 holds a stable +10-14 mb gap to exp407 throughout. exp408 (further d_v cut) opens an additional ~20 mb gap.

## Conclusion / how to apply
1. **Keep d_v ≥ 16** for the LUT-LM base. Cutting head width is the most expensive way to shrink the model — 8 dims/head appears too narrow for SDPA to discriminate cleanly.
2. **E=32 is a small but real regression vs E=48** (~10 mb). If you need to shrink, prefer cutting E over d_v. But neither change is param-efficient compared to cutting `out_tph` or `residual_tph`.
3. Both factors compound additively in this regime — no cross-term observed (10.6 + 22.8 ≈ 33.4 holds exactly).
4. For shrinks at bs=16, distinguish two budgets:
   - **Activation memory** (peak GPU RAM): v_lut dominates (~3 GB peak at tph=128). Halving v_tph → big memory win.
   - **Parameter count**: out_proj and residual_lut tie at 1.57M/layer each (33% of layer's LUT each), well above v_lut's 786K (16%). For pure param cuts, halving out_tph or residual_tph saves ~4.7M total (~−11%), vs halving v_tph saves only ~2.4M (~−5.5%).
