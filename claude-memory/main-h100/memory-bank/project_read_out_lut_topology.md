---
name: project_read_out_lut_topology
description: "exp588 — single read_out_lut at end of stack (tph=1536) beats exp587's 6 per-layer residual_lut (tph=256 each) by −12.7 mb at identical params and bandwidth. Per-layer D-stream injections were redundant work, not depth-multiplying capacity."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Headline:** **exp588 = 1.4344 bpb @ 279.97M params, 1.027h. New LUT-LM SoTA. Beats exp587 (1.4471) by −0.0127 at identical params and trunk bandwidth.**

## The topology swap

exp587 (prior SoTA) had a per-layer residual_lut in each of 6 LUTBlocks: TinyMHLut(soft, NAP=6, tph=256, E→D), output summed into a D-stream accumulator `x_resid` across all 6 layers, then ln_final(D) → Linear(D, V).

exp588 removes the per-layer residual_lut entirely. The LUTBlock now returns only the updated E-stream `x_lut`. After the 6-layer stack, a single `read_out_lut` (TinyMHLut(soft, NAP=6, **tph=1536**, E→D) — same NAP, 6× the tables) reads the final E-stream once. Then ln_final(D) → Linear(D, V).

By design this is param-matched and bandwidth-matched:
- residual_lut params: 6 layers × tph=256 × 64 rows × D = 37.7M total → read_out_lut: 1 × tph=1536 × 64 × D = 37.7M total. **Same.**
- residual_lut trunk bandwidth: 6 × (256 × 384 × 4B) = 18.87 Mbits/tok → read_out_lut: 1 × (1536 × 384 × 4B) = 18.87 Mbits/tok. **Same.**
- Total params: 279.97M (vs 279.97M for exp587). **Same to a digit.**
- Total trunk bandwidth: 155 Mbits. **Same.**

Only the topology differs.

## Trajectory (exp588 vs exp587)

| step | exp588 | exp587 | exp588 − exp587 |
|--|--|--|--|
| 200 | 2.2569 | 2.2475 | +0.0094 |
| 400 | 1.9876 | 1.9744 | +0.0132 |
| 600 | 1.8824 | 1.8739 | +0.0085 |
| 1000 | 1.7737 | 1.7668 | +0.0069 |
| 2000 | 1.6449 | 1.6415 | +0.0034 |
| 3000 | 1.5782 | 1.5785 | −0.0003 |
| 4000 | 1.5310 | 1.5331 | −0.0021 |
| 5000 | 1.4903 | 1.4972 | −0.0069 |
| 6000 | 1.4617 | 1.4731 | −0.0114 |
| 7000 | 1.4447 | 1.4565 | −0.0118 |
| **8000** | **1.4344** | **1.4471** | **−0.0127** |

**Phases:**
- Steps 200-1000: exp588 trails by +0.005 to +0.013 (warmup penalty — single concentrated head has more to integrate, slower start).
- Steps 1500-2500: gap closes to ~+0.002 to +0.003.
- Step 3000: crossover. exp588 ahead.
- Steps 4000-8000: lead grows monotonically from −0.002 to **−0.013**. Late-phase slope ~5-10% steeper than exp587.

The lead is real and **grows through training**, not a transient. exp588 at step 6800 already matched exp587's final 1.4471 — i.e., **same quality at 15% less compute**.

## Why the simpler topology wins

The per-layer residual_lut was injecting `r_l = residual_lut_l(LN(x_lut_l))` into a sum `x_resid = Σ_l r_l`. Several mechanisms explain why one concentrated read-out outperforms this:

- **D-stream was a passive accumulator, not a true stream.** No layer ever *reads* x_resid; only the final unembedder consumes it. So x_resid wasn't carrying inter-layer information forward — it was a sum of 6 independent projections of intermediate x_lut states.
- **Final E embedding already aggregates all attention work.** The E-stream is updated additively across layers (`x_lut += out_e_l`), so the final `x_lut` already contains the whole layer-by-layer integration. A single read-out from it captures everything the 6 distributed read-outs were summing — just without the early-layer noise being baked into the sum.
- **Capacity concentration helps.** At fixed total residual_lut params, one tph=1536 lookup table can encode finer-grained partitions of the final E manifold than 6 disjoint tph=256 tables forced to fire from progressively-less-trained intermediate x_lut states. The intermediate states (L1, L2 outputs) carry less signal than the final, and their residual_lut contributions were essentially "noise summed in" until late training.
- **No gradient routing through 6 paths.** In exp587 the read-out loss gradient splits across 6 residual_lut modules; in exp588 it concentrates onto one. With Lion and bs=16, denser per-row gradient updates train the LUT rows better.

This reframes the *dual-stream residual* design: x_resid as a per-layer accumulator was **redundant work**. The actual lever was always concentrating the read-out at the end.

## Bandwidth-loss position

| | params | trunk Mbits | total Mbits | bpb |
|--|--|--|--|--|
| exp328 (tied vanilla) | 23M | 340 | 742 | 1.3882 |
| exp476 (untied vanilla) | 36M | 340 | 742 | 1.4143 |
| **exp588 (this SoTA)** | **280M** | **155** | **558** | **1.4344** |
| exp587 (prior SoTA) | 280M | 155 | 558 | 1.4471 |
| exp567 (E=96 baseline) | 103M | 72 | 475 | 1.4768 |

Gap-to-vanilla progression after exp588:
- vs tied vanilla (1.3882): +0.0462 (closed by 12.7 mb from exp587's +0.0589)
- vs untied vanilla (1.4143): **+0.0201** (closed by 12.7 mb from exp587's +0.0328)

**LUT-LM is now within 20 mb of untied vanilla** at 2.2× cheaper trunk bandwidth.

## Implications

1. **Per-layer residual_lut is not a depth-multiplier — it was redundant capacity.** Cumulative session lesson: when scaling the full arch via E (exp585/586) or residual_tph (exp587), we were spending those gains *despite* the per-layer residual_lut, not *because* of it. Removing it and concentrating the budget at the end is a net win at exp587's scale.

2. **The "D-stream as a parallel residual" picture is wrong at this scale.** No layer reads from D; calling it a stream was overstating the topology. It was always a sum of read-outs.

3. **The recipe simplifies.** LUTBlock now only has qk_lut, v_lut, out_proj. The whole residual_lut placement is one module at the model level, after the layer stack. ln_post inside the block disappears.

4. **Open question for future work:** does an actual D-stream (where layers *read* and *write* x_resid) recover and exceed the read_out_lut alone? Or is the LUT-LM happy without any persistent state beyond the additive E-stream?

5. **Lever ranking after exp588:** read_out concentration > residual_tph > E-widening > everything else tried (heads, anchor learning, multi-NAP, dual heads at E=96).

## Files

`/home/starost/spiky/nanochat_exps/exp588_read_out_lut/` — config.json, train.py, metrics.csv, summary.json, checkpoint.pt, loss.png.

Parent: `[[project_e192_full_arch_sota]]` (exp567/585/586/587 SoTA chain).
