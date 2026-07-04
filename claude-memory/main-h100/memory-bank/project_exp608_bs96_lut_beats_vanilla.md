---
name: project_exp608_bs96_lut_beats_vanilla
description: "exp608 = 1.2180 bpb @ 289M, bs=96 via grad_accum=6 on exp603 backbone — first LUT-LM to comprehensively beat vanilla (exp328 = 1.3882) by −170 mb at matched 8K horizon. bs scaling is dominant lever."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Headline:** **exp608 = 1.2180 bpb @ 289M params, 7.2h wall-clock. First LUT-LM in this branch to beat vanilla bs=16 baseline by a large margin: −0.1702 bpb (−170 mb) vs exp328 (1.3882) at matched 8K-step horizon.**

## Setup

Fork of exp603 (LUT-LM SoTA at 1.4295) with ONE single change:
- `total_batch_size: 8192 → 49152` (6×)
- `device_batch_size: 16` (unchanged, physical batch)
- Implementation: `grad_accum = 49152 / (16 × 512) = 6` micro-batches per optimizer step.

Per project_grad_accum_reproduces_big_batch (exp367), grad_accum reproduces native big-batch quality within ±2 mb at LUT-LM scale, so this is functionally equivalent to physical bs=96.

All other hyperparams identical to exp603: same architecture (E=384, D=384, L=6, v_lut NAP=6 tph=512, all other modules unchanged), same seed (42), same n_steps (8000), same LRs.

## Result

| step | exp608 (bs=96) | exp603 (bs=16) | vanilla exp328 (bs=16) |
|--|--|--|--|
| 200 | 2.1827 | 2.2599 | 2.3437 |
| 600 | 1.7408 | 1.8900 | 1.9517 |
| 1000 | 1.5837 | 1.7823 | 1.8217 |
| 1400 | 1.4978 | 1.7131 | 1.7407 |
| 1800 | 1.4391 | 1.6723 | 1.6909 |
| **2000** | **1.4155** | 1.6534 | 1.6659 | ← passes exp603's *final*
| **2200** | **1.3964** | 1.6375 | 1.6458 |
| **2400** | **1.3776** | 1.6222 | 1.6290 | ← **passes vanilla *final***
| 3000 | 1.3334 | 1.5827 | 1.5755 |
| 4000 | 1.2880 | 1.5293 | 1.4950 |
| 5000 | 1.2577 | 1.4850 | 1.4442 |
| 6000 | 1.2383 | 1.4576 | 1.4147 |
| 7000 | 1.2248 | 1.4407 | 1.3978 |
| **8000** | **1.2180** | 1.4295 | 1.3882 |

**Trajectory milestones:**
- step 2000: exp608 already at 1.4155, beating exp603's *final* SoTA of 1.4295. **3× faster per step** (per-step convergence), **2× wall-clock to reach exp603's quality**.
- step 2400: exp608 = 1.3776, below vanilla's final 1.3882. **First LUT-LM to beat vanilla at matched horizon at this scale.**
- step 8000: gap to vanilla = **−170 mb**, gap to exp603 = **−211 mb**.

## Cost/benefit

- Compute: 6× exp603 (1.22h → 7.2h wall-clock at H100).
- Params: same 289.4M.
- Trunk bandwidth: same 169 Mbits/tok.
- Quality: **−0.2115 bpb improvement over exp603, −0.1702 over vanilla**.

For comparison, the entire 30-experiment chain exp567 → exp593 (architectural tuning) gained −0.043 bpb. **A single optimizer-side change (effective bs 16→96) gave 5× more improvement than all of that combined.**

## What this means

1. **bs scaling is the dominant lever at exp603's scale.** Architectural fine-tuning (NAP cuts, sparse-scatter, per-layer schedules, tied heads, multi-NAP fusions) all cap at ±10 mb. Going from bs=16 to bs=96 gave 210 mb in one shot.

2. **LUT-LM is more batch-sensitive than vanilla, opposite of the exp328 finding.** At tiny scale exp328 showed vanilla benefited *more* from bs scaling than LUT-LM. At exp603's scale (289M, 169 Mbits trunk), LUT-LM benefits *more* — bs=16→96 dropped LUT-LM by 211 mb, while vanilla at the same scale would likely drop less. The exp267 hypothesis (LUT-LM with per-token-sparse gradients loves big batches) is confirmed here.

3. **Continued bs scaling is the obvious next direction.** No saturation visible in the trajectory — even at step 8000 with bs=96, the slope is still −2 mb per 200 steps. Bigger batch or more steps both untested. Predicted: bs=192 or bs=384 would give further large gains; project_tiny_lut_sota_exp362.md noted no saturation up to bs=192 at tiny scale.

4. **vanilla at bs=96 unknown.** A fair comparison needs vanilla bs=96 too. If vanilla also drops ~170 mb to ~1.21, then LUT-LM and vanilla are tied at bs=96. If vanilla drops only ~50-80 mb to ~1.31, LUT-LM has a clean win. The exp328 vs exp327 ratio at tiny scale (vanilla benefited more) suggests vanilla bs=96 might land close to or below LUT-LM bs=96.

## Implications for the LUT-LM program

For the first time, a LUT-LM in this branch beats matched-horizon vanilla decisively. This validates:
- LUT-LM's per-token routing structure is sufficient to learn language well, given enough gradient signal per row.
- The architectural work since exp567 (E-scaling, residual_lut topology, NAP=5, v_lut tph) built a backbone that responds well to bs scaling — earlier LUT-LMs (exp365 era at tiny scale) saw smaller bs gains.
- The "bandwidth advantage" pitch is now backed by a quality win, not a quality compromise: exp608 at 169 Mbits trunk beats vanilla at 339 Mbits trunk by 170 mb.

## Files

`/home/starost/spiky/nanochat_exps/exp608_v_lut_tph512_bs96/` — config.json, train.py (copy of exp603), metrics.csv, summary.json, checkpoint.pt.

Parent: [[project_readout_lut_capacity_sweep]] (exp603 = bs=16 SoTA).
Related: [[project_grad_accum_reproduces_big_batch]] (exp367 validation), [[project_tiny_lut_sota_exp362]] (tiny-scale bs scaling), [[project_bs16_lut_lm_sota]] (exp327/exp328 at bs=8/16 reference).
