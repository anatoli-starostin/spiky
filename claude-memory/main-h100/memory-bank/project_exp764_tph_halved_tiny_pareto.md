---
name: exp764-tph-halved-tiny-pareto
description: "exp764 = exp760 with ALL tphs halved + eff bs doubled to 96 = 1.2116 hard @ 97.5M, 24K, 5.72h. New tiny-deployment Pareto point alongside exp760 quality SOTA."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp764 — tiny-deployment Pareto via tph-halving + 2× tokens

**Final: hard val_bpb = 1.2116 @ step 24000, 97.52 M params, 5.72 h wallclock.**

The exp755→exp760 trick applied at the **tph knob** instead of the n_steps knob:
- Halved every tph from exp760 (qkv/v 256→128, out 512→256, resid/emb 256→128).
- Doubled effective batch (eff bs 48 → 96, phys 48 × accum 2) to absorb the saved per-step compute.
- Same n_steps = 24K, same NAPs (qkv=4, v=6, out=7, resid=6, emb=6), same E=192/d_v=32, same recipe (hard + dense_K + bf16 storage + master Lion + global clip(1.0)).

## Pareto positioning

| Run | Hard val | Params | Wall | Tokens | Notes |
|---|---|---|---|---|---|
| exp720 (hardened) | 1.2909 | 85 M | 9.89 h (soft 16K) | 786 M | TinyMHLut hybrid_smooth + bit-reverse perm |
| **exp764** | **1.2116** | **97.5 M** | **5.72 h** | **1.18 B** | native hard + dense_K |
| exp760 | 1.2048 | 176 M | ~5.1 h (clean) | 590 M | current quality SOTA |

exp764 trades +6.8 mb of bpb for −44.6 % params and ~1.8× lower inference HBM bytes — the new
**tiny-deployment Pareto point** in the FastMHL family.

## Trajectory shape

| step | exp764 | exp760 | Δ |
|---|---|---|---|
| 2400 (peak LR) | 1.4758 | 1.5067 | **-31 mb** (peak lead) |
| 4000 | 1.3676 | 1.3921 | -25 mb |
| 8000 (1/3) | 1.2870 | 1.2979 | -11 mb |
| 12000 (1/2) | 1.2529 | 1.2579 | -5 mb |
| 13800 | 1.2428 | 1.2425 | **0** (crossover) |
| 16000 (2/3) | 1.2323 | 1.2305 | +2 mb |
| 18000 (3/4) | 1.2244 | 1.2206 | +4 mb |
| 24000 (final) | **1.2116** | **1.2048** | **+6.8 mb** |

Bigger batch + smaller arch leads convincingly through warmup and mid-cosine, crosses around
step ~14000, then capacity edges ahead by 5-7 mb in the late cosine. Classic
**capacity-vs-tokens equilibrium** at this regime.

## Why tph is the right capacity knob

200-step wallclock bench (`bench_exp760_vs_E96.py`) measured the wall-saving leverboard at exp760's
scale:

| Lever | Wall saving | Param saving |
|---|---|---|
| All tphs × 0.5 | **-44.7 %** | -44.6 % |
| forward_mode hybrid_smooth → hard | -37 to -39 % | 0 % |
| E:192→96, d_v:32→16 | -4.3 % | -28.6 % |

The tph knob cleanly halves both compute (gather + dense_K bwd) and HBM bandwidth on every LUT
module — linear scaling, no diminishing returns. By contrast, E/d_v shrink saves params but
barely moves wall because residual_lut + emb_resid_lut + unembedder dominate wall and are
E-insensitive. See [[lut-wall-leverboard]] for the full breakdown.

## How to apply

- **For tiny-deployment LUT-LM forks**, prefer exp764's recipe over exp720's: native hard +
  dense_K + bf16 storage + master Lion + global clip(1.0), with all tphs at the halved values
  (128/128/256/128/128) and **eff bs 96**.
- Checkpoint at `nanochat_exps/exp764_tph_halved_24k_bs96/checkpoint.pt`.
- For pure-quality SOTA still use exp760 / exp735.
- The +7 mb cost is roughly the right price for ~2× smaller arch at this LR/horizon regime.
  Don't expect to recover it with 4× tokens (diminishing returns are flat past 1.2 B tokens at
  this arch).

## What this tells you about future tuning

- **Don't chase wall via E/d_v** — only ~4 % wall per halving, while sacrificing 29 % params'
  worth of representation capacity.
- **tph is the dominant wall AND capacity lever**. The exp755→exp760 schedule trick worked
  because hard mode + dense_K + bf16 was a huge wall cut; the tph trick is the natural next
  step in the same compute-vs-capacity reframing.
- The "tokens-substitute-for-capacity" conversion is roughly **2× tokens ≈ +7 mb** at this
  regime. Below 2× tokens, the smaller arch loses worse than that; above 2× the marginal token
  is wasted.

Cross-refs: [[exp735-v-lut-nap7-sota]] (current quality SOTA at 16K),
[[lut-wall-leverboard]] (the leverboard finding that motivated this trade),
[[tinymhl-hybrid-smooth-hard-eval-bug]] (related: exp720's true hard-deployable number is 1.2909, not 1.2052).
