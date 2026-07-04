---
name: project_e192_full_arch_sota
description: exp585 — E=192/d_v=32 on the exp567 proven full-arch recipe beats exp567 by 14.3 mb; current LUT-LM SoTA at this bandwidth class. E-widening works on full arch but NOT on dual-head variants.
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Headline:** **exp585 = 1.4625 bpb @ 155.7M params, 0.733h. New LUT-LM SoTA at ~500 Mbits/tok bandwidth class, beating exp567 (1.4768) by −14.3 mb.**

**Architecture**: exact exp567 recipe at wider E:
- E=192 (was 96), d_v=32 (was 16) — preserves H·d_v=E invariant, out_proj stays square.
- D=384 unchanged.
- Per-layer residual_lut NAP=6 tph=128 (E=192 → D=384) — exp567 pattern.
- Untied Linear(D=384, V=32768) head — exp567 pattern, NO dual head.
- Everything else byte-for-byte exp567.

**Trajectory vs exp567** (gap = exp585 − exp567):

| step | exp585 | exp567 | gap |
|--|--|--|--|
| 200 | 2.2510 | 2.2561 | −0.005 |
| 1000 | 1.7728 | 1.7820 | −0.009 |
| 2000 | 1.6522 | 1.6622 | −0.010 |
| 3000 | 1.5889 | 1.5986 | −0.010 |
| 4000 | 1.5450 | 1.5554 | −0.010 |
| 5000 | 1.5090 | 1.5229 | −0.014 |
| 6000 | 1.4859 | 1.4995 | −0.014 |
| 7000 | 1.4705 | 1.4911 | −0.021 |
| **8000** | **1.4625** | **1.4768** | **−0.0143** |

Lead held constant ~10 mb through steps 1-4K, then widened to 14-21 mb in the late phase. exp585's slope was ~5-15% steeper than exp567's in steps 5K-7K. Win is real and **growing through training**, not transient.

**Resources:**
- Params: **155.72M** (+52.7M vs exp567's 103M)
  - tok_emb_E: +3.15M (V·E grows)
  - v_lut × 6L: +11.80M (d_v doubles)
  - out_proj × 6L: +37.74M (n_out=E doubles) ← dominant cost
- Trunk bandwidth: **96 Mbits** (+24 vs exp567's 72)
- Total bandwidth: **499 Mbits/tok** (+24 vs exp567's 475; head Linear(D=384, V) unchanged at 403 Mbits)
- Wall-clock: 0.733h (1.14× exp567's 0.642h)

**Per-token convergence position:**
- exp585 vs tied vanilla (exp328 = 1.3882): gap **+0.0743** (was +0.089 for exp567)
- exp585 vs untied vanilla (exp476 = 1.4143): gap **+0.0482** (was +0.0625 for exp567)
- Closes the LUT-vs-vanilla gap by ~15 mb on both reference lines.

**Critical context — E-widening works on full arch only:**

| variant at E=192 (or wider) | head topology | residual_lut | final bpb | comments |
|--|--|--|--|--|
| **exp585 (this)** | **single Linear(D=384, V) untied** | **per-layer x6** | **1.4625** | **SoTA** |
| exp583 | tied dot + Linear(D, V) DUAL | per-layer x6 | 1.5200 | dual TIED hurts |
| exp582 | tied dot + Linear(D, V) DUAL | 1 final post-L6 | proj ~1.53 | dual TIED hurts |
| exp575 | tied dot only | none | proj ~1.54 | minimal arch caps lower |
| exp581 (E=384) | tied dot only | none | proj ~1.51 | E-widening on minimal helps less |

The E-widening **compounds with the proven full-arch recipe** but doesn't help (or hurts) when combined with:
- Dual heads (tied or untied — see also [[project_head_arch_doesnt_stack]])
- Minimal-arch stripping (no residual_lut, tied-only)

This locks in: the productive lever on the LUT-LM family at this scale is **scaling exp567's exact recipe** (more E, more D, more residual_tph) NOT topology changes. Per [[project_lut_lm_residual_width_sota]] the D-widening sweep (exp562/565) also worked on full arch.

**Bandwidth-loss Pareto position vs vanilla family:**

| | params | trunk (Mbits) | total (Mbits) | bpb |
|--|--|--|--|--|
| exp328 (tied vanilla, E=384) | 23M | 340 | 742 | 1.3882 |
| exp476 (untied vanilla, E=384) | 36M | 340 | 742 | 1.4143 |
| **exp586 (next E step)** | **261M** | **146** | **549** | **1.4509** |
| **exp585 (this)** | **156M** | **96** | **499** | **1.4625** |
| exp567 (prior LUT SoTA) | 103M | 72 | 475 | 1.4768 |

exp585 vs vanilla: **3.5× cheaper trunk, 1.5× cheaper total bandwidth, +0.074 bpb gap to tied vanilla**.

## Update 2026-05-26 — exp586 (E=384/d_v=64) advances SoTA further

**exp586 = exp585 + E doubled to 384 / d_v doubled to 64 (same H·d_v=E rule). Final: 1.4509 @ 261.10M, 0.959h. Beats exp585 by −11.6 mb, beats exp567 by cumulative −25.9 mb.**

Trajectory tracked exp585 with tiny gap through warmup (≈0 at step 200/800/1000), then opened steadily through mid-late phase: lead grew 0 → 6 → 7 → 9 → 10 → 12 mb. Slope late-phase matched vanilla's −0.017/k at step 6-7K.

**Diminishing-return per-param efficiency:**

| E doubling | Δparams | Δbpb | bpb/M |
|--|--|--|--|
| E=96 → 192 (exp567 → exp585) | +52M | −14.3 mb | **−0.275 mb/M** |
| E=192 → 384 (exp585 → exp586) | +105M | −11.6 mb | **−0.110 mb/M (2.5× less efficient)** |

**Gap to vanilla closed by ~26 mb in total** (vs tied vanilla 1.3882):
- exp567 (E=96): +0.089
- exp585 (E=192): +0.074
- **exp586 (E=384): +0.063**

vs untied vanilla 1.4143: gap is now **+0.037** — closer than any LUT-LM ever.

**Trunk bandwidth scaling: 72 → 96 → 146 Mbits as E doubles** (still 2.3× cheaper than vanilla's 340 trunk at E=384). Total bandwidth 549 Mbits / 26% cheaper than vanilla's 742.

Next probe direction questions:
- E=512 (further E-sweep): diminishing returns predict ~5-8 mb more.
- D widening (exp562/565 path proven on full arch, never combined with E=384).
- bs scaling (exp364 family showed bs=192 gave huge gain; never applied to exp585/586 scale).
- Trunk sparsification (reduce tph proportionally to lower per-Mbits cost at this E).

## Update 2026-05-26 — exp587 (residual_tph 128→256 on exp586) extends SoTA

**exp587 = exp586 + residual_tph doubled to 256. Final: 1.4471 @ 279.97M, 0.986h. Beats exp586 by −3.8 mb. Cumulative −29.7 mb vs exp567.**

Single config change vs exp586 (`residual_tph: 128 → 256`). +18.87M params (residual_lut 3.15M → 6.29M per layer × 6), +9.4 Mbits trunk bandwidth, +1.7% total bandwidth.

Trajectory:

| step | exp587 | exp586 | exp587−exp586 |
|--|--|--|--|
| 200 | 2.2475 | 2.2511 | −0.004 |
| 1000 | 1.7668 | 1.7722 | −0.005 |
| 2000 | 1.6415 | 1.6487 | −0.007 |
| 3000 | 1.5785 | 1.5857 | −0.007 |
| 4000 | 1.5331 | 1.5388 | −0.006 |
| 5000 | 1.4972 | 1.5020 | −0.005 |
| 6000 | 1.4731 | 1.4773 | −0.004 |
| 7000 | 1.4565 | 1.4607 | −0.004 |
| **8000** | **1.4471** | **1.4509** | **−0.0038** |

Lead held ~5-7 mb through mid-phase (steps 1K-4K) then drifted down to 4 mb late.

**Per-param efficiency of the residual_tph lever** declines with backbone size:
- exp530 (E=64 backbone): −9.4 mb at +18.9M = **−0.50 mb/M**
- exp587 (E=384 backbone): −3.8 mb at +18.87M = **−0.20 mb/M** (~40% of prior efficiency)

Gap-to-vanilla progression after session:

| ref | exp567 gap | exp587 gap | closed by |
|--|--|--|--|
| exp328 (tied vanilla, 1.3882) | +0.089 | **+0.0589** | 30 mb |
| exp476 (untied vanilla, 1.4143) | +0.063 | **+0.0328** | 30 mb |

## Full session SoTA progression (2026-05-26)

| exp | knob | params | trunk Mbits | final | Δ vs prev |
|--|--|--|--|--|--|
| exp567 | E=96 baseline | 103M | 72 | 1.4768 | — |
| exp585 | E 96→192 + d_v 16→32 | 156M | 96 | 1.4625 | −14.3 mb |
| exp586 | E 192→384 + d_v 32→64 | 261M | 146 | 1.4509 | −11.6 mb |
| **exp587** | **residual_tph 128→256** | **280M** | **155** | **1.4471** | **−3.8 mb** |

Diminishing returns: each step gives progressively less per-param improvement. The "scale exp567 recipe" path has produced 30 mb of SoTA improvement but each step is now sub-5 mb.

Files: `/home/starost/spiky/nanochat_exps/exp585_E192_full_arch/`, `/exp586_E384_full_arch/`, `/exp587_E384_restph256/`.
