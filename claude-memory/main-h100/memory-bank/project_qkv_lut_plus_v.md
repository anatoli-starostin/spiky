---
name: project-qkv-lut-plus-v
description: "A joint qkv_lut that provides Q,K and an ADDITIVE shallow v-branch (on top of an unchanged NAP=8 v_lut) is the new LUT-LM SOTA at 8K — exp326 = 1.5887 (vs exp321 = 1.5933, Δ=−0.0046). Pure-joint variants (replacing v_lut entirely) all lost."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Joint qkv_lut with additive v-branch (2026-05-14, exp326)

**Fact:** Replacing exp321's `qk_joint` (NAP=6, tph=256, n_out=2*d_qk=128) with a wider joint `qkv_lut` (NAP=6, tph=256, n_out=2*d_qk+d_v=160), then routing:
- `q, k ← qkv_lut(x)[..., :2*d_qk]`
- `v ← v_lut(x) + qkv_lut(x)[..., 2*d_qk:]`   ← additive contribution from joint table

(v_lut is KEPT unchanged at NAP=8, tph=256, n_out=d_v=32) gives a new LUT-LM SOTA at 8K of **1.5887 bpb** @ 620.8 M params (training time 0.911 h, +0.030 h vs exp321).

| run | qkv config | params | val_bpb @ 8K | Δ vs exp321 |
|---|---|---|---|---|
| exp303 (prior dense SOTA, learned pos) | qk_joint + v_lut, no RoPE | 602.1 M | 1.6509 | +0.0576 |
| exp321 (RoPE SOTA) | qk_joint (NAP=6, t=256) + v_lut (NAP=8, t=256), RoPE | 601.9 M | 1.5933 | 0 |
| **exp326** | qkv_lut (NAP=6, t=256, n_out=160) + v_lut + RoPE, v additive | **620.8 M** | **1.5887** | **−0.0046** |
| exp322 (pure joint, no v_lut) | joint qkv, NAP=6, t=256 | 545 M | trailing exp321 by ~+0.022 @ s=3400 (stopped) | — |
| exp323 (pure joint, deeper) | joint qkv, NAP=8, t=256 | 828 M | +0.012 @ s=2000 (stopped) | — |
| exp324 (pure joint, wider) | joint qkv, NAP=6, t=512 | 640 M | +0.012 @ s=4400 (stopped) | — |
| exp325 (pure joint, widest) | joint qkv, NAP=6, t=1024 | 828 M | +0.012 @ s=5800 (stopped) | — |

**Key insight:** the pure-joint variants (exp322–exp325) that REPLACED v_lut all lost, even at much higher param counts. The win in exp326 comes from:
1. **Keeping the dedicated NAP=8 v_lut** so v has its deep-table capacity.
2. **Adding** a parallel shallow (NAP=6, table_dim=64) v contribution from the shared qkv table — this lets q/k/v share anchor decisions WITHOUT giving up v's deep-NAP path.

So the v branch of qkv_lut acts as a **shallow residual** to v_lut, not a replacement. Mirrors the residual_lut design (small additive stream alongside a deep main path).

**Implementation** (nanochat_exps/exp326_qkv_lut_plus_v/train.py): see LUTBlock.forward. q/k from `qkv_lut(x)[..., :d_qk]` and `[..., d_qk:2*d_qk]`; `v = v_lut(x) + qkv_lut(x)[..., 2*d_qk:]`. Param impact: per-layer LUT params went from 25.2 M (exp321: qk 12.6 + v 12.6) to 28.3 M (exp326: qkv_lut 15.7 + v_lut 12.6), so +18.6 M total over 6 layers — modest spend for −0.0046 bpb.

**How to apply:**
- New LUT-LM forks should keep both `qkv_lut` and `v_lut`, sum their v outputs. Don't replace `v_lut` with the joint table; gives up its NAP=8 capacity.
- The shared qkv table should be NAP=6 (matches q,k traditional depth); deepening it (NAP=8 in exp323) did not help.
- Capacity gap to vanilla+RoPE (exp319 1.5468 @ 23 M) is now **+0.042 bpb** at 27× more params. Still open — but exp326 is the cleanest 8K LUT-LM baseline yet.

**Comparison vs the failed joint-only forks (exp322–exp325):** the pattern is clear — anchor-sharing between q/k/v helps as a *residual* signal, not as the sole source of v. The shallow joint table cannot replicate the deep (NAP=8) entries v_lut uses for its own routing.
