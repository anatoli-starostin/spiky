---
name: v-lut-nap-trade-neutral
description: exp407 — v_lut NAP=8 tph=32 → NAP=6 tph=128 (param-matched) is essentially tied with exp392. 4× v_lut bandwidth buys zero quality at bs=16 in the cleaned-up base architecture.
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# v_lut NAP-for-tph trade — neutral at bs=16 (2026-05-17, exp407)

## Setup
Cleaned-up fork of exp392 (config trim + dead-branch removal). One architectural change: v_lut NAP=8 tph=32 → **NAP=6 tph=128** (param-matched at 786,432/layer; v_lut bandwidth/token/layer rises 192 → 768 = 4×).

Other changes: out_proj uses plain `TinyMHLut(NAP=6, tph=512)` instead of single-component `TinyMultiNapMultiHeadLut([[6,512]])` — functionally identical, just simpler code.

All other hyperparams identical to exp392.

## Result
| Run | Final bpb | Params | Time |
|---|---|---|---|
| exp392 (NAP=8 v_lut tph=32) | 1.6029 | 43.06M | ~2h |
| **exp407 (NAP=6 v_lut tph=128)** | **1.6038** | 43.06M | 0.23h |

**Δ = +0.0009 (~+1 millibit)** — essentially tied; within noise band at every step 200–8000.

## Trajectory (every 1000 steps)
| Step | exp407 | exp392 | Δ (mb) |
|------|--------|--------|--------|
| 1000 | 1.8675 | 1.8715 | −4.0 |
| 2000 | 1.7626 | 1.7672 | −4.6 |
| 4000 | 1.6727 | 1.6740 | −1.3 |
| 6000 | 1.6239 | 1.6235 | +0.4 |
| 8000 | 1.6038 | 1.6029 | +0.9 |

exp407 led by ~3-5 mb in early-to-mid phase, but they converged in late phase. Final tied.

## Conclusion
The wide-beats-deep heuristic that worked for residual_lut and out_proj does **NOT** generalize to v_lut at bs=16. At fixed parameter budget, NAP=8 with fewer wider tables is as good as NAP=6 with 4× more tables. The 4× attention-bandwidth cost of NAP=6 v_lut buys zero quality.

This matches the `project_qkv_lut_shape_findings.md` finding that wide-beats-deep is **residual_lut-specific** and doesn't generalize to attention-input LUTs.

## How to apply
- Keep `v_lut` at NAP=8 with the smaller tph (e.g. 32) for bs=16 bandwidth-efficient builds. Don't pay 4× v_lut bandwidth without arch evidence at the new batch size.
- The cleaned-up exp407 train.py / config layout is the new clean template for future LUT-LM forks (no STE plumbing, no multi-nap wrapper, no vestigial qk_* keys, no pos_emb plumbing).

## Code changes
None to library code. exp407 itself is a cleaned-up train.py removing:
- `TinyMultiNapMultiHeadLut` import + local module file
- `_TINY_MULTIALT_KWARGS` and STE branches in `_make_qkv_joint` / `_make_v` / `_make_out`
- `out_proj_multi_nap` and `out_tph_per_layer` branches in `_make_out`
- `qk_input_nap` / `qk_tph` fallback patterns
- `_POS_EMB_*` plumbing and `pos_emb_params` from optimizer groups
- 12 unused config keys
