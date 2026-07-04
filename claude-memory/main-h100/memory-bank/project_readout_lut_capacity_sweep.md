---
name: project_readout_lut_capacity_sweep
description: "Read_out LUT capacity-axis disambiguation (exp588-593). Three independent capacity knobs (atom count, atom width, mixture depth) cut independently from exp588 (1.4344 SoTA). Mixture depth dominates; NAP=5 is the sweet spot at fixed depth — exp593 = 1.4337 at half the read_out params."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

## Headline

**exp593 = 1.4337 bpb @ 261.10M, 1.036h. New LUT-LM SoTA.** NAP=5 read_out (32 rows/table) ties exp588 (NAP=6, 64 rows/table) and is nominally ahead in every late-phase eval, at **half the read_out_lut params** (37.7M → 18.87M, −6.7% of model).

## The capacity-axis disambiguation

Starting from exp588 (single read_out_lut: H=1, NAP=6, tph=1536, n_out=D=384), the read_out has three independent capacity knobs:

- **atom count** = tph × K = how many distinct row-vectors the LUT can address
- **atom width** = n_out per table = how rich each fetched row is
- **mixture depth** = tph = how many table contributions are summed into each output coordinate

Six configurations probed (all single read_out at end, same train.py except for read_out cfg):

| exp | knob | atoms | width | depth | read_out params | trunk bw | gap @ step 2600 | final |
|--|--|--|--|--|--|--|--|--|
| exp588 | dense baseline | 98K | 384 | 1536 | 37.7M | 155 Mb | — | 1.4344 |
| exp589 | 6 heads × 64 | 98K | 64 | 256 | 6.29M | 139 Mb | +0.029 (killed @ 3K) | (n/a) |
| exp590 | sparse-scatter | 98K | 64 | 256 | 6.29M | 139 Mb | +0.029 (killed @ 3.6K) | (n/a) |
| exp591 | tph 1536→512 | 32K | 384 | 512 | 12.58M | 142 Mb | +0.030 (killed @ 1K) | (n/a) |
| exp592 | NAP 6→4 | 24K | 384 | 1536 | 9.43M | 155 Mb | +0.020 (killed @ 2.6K) | (n/a) |
| **exp593** | **NAP 6→5** | **49K** | **384** | **1536** | **18.87M** | **155 Mb** | **+0.003** | **1.4337** |

## Findings

### 1. Mixture depth dominates at this scale.

The 6× depth reduction (1536 → 256) in exp589 and exp590 costs ~0.028-0.030 bpb regardless of topology (multi-head vs sparse-scatter). Same param/bandwidth cut, same loss penalty — the topology of *how* you cut depth doesn't matter. What matters is the per-output-coordinate summation depth.

### 2. Atom count is slack down to 24K (at fixed depth).

exp592 (NAP=4, 24K atoms, depth 1536) was +0.020 — *less* damaging than the depth-cut variants at 98K atoms × 256 depth. exp593 (NAP=5, 49K atoms, depth 1536) basically ties exp588 (98K atoms). Conclusion: with depth held at 1536, you need ~25-50K atoms minimum but going beyond that has diminishing returns.

### 3. NAP=5 is the gradient-coverage sweet spot.

At bs=16 × ctx=512, each step fires ~12.6M rows across the read_out's `tph × K` rows:

| NAP | K | atoms | tokens/row/step |
|--|--|--|--|
| 6 (exp588) | 64 | 98K | 128 |
| **5 (exp593)** | **32** | **49K** | **256 (2×)** |
| 4 (exp592) | 16 | 24K | 512 — but atom ceiling kicks in |

NAP=5 doubles per-row gradient coverage vs NAP=6 (256 vs 128 tokens/row), giving cleaner Lion sign updates. The gain compounds late-phase as cosine LR drops 10× and update precision matters more. Trajectory matches: exp593 had no gap through warmup, opened −0.003 mb lead at steps 4K-5K, sustained through step 8K final.

### 4. Atom count × depth has a Pareto frontier; NAP=5 sits at the inflection.

- **Below the frontier (more atoms, less depth):** exp589/590 with 98K × 256 lose 28 mb.
- **Below the frontier (less atoms, less depth):** exp591 with 32K × 512 lost ~30 mb.
- **On/above the frontier (atoms preserved, depth preserved):** exp593 = exp588.

The frontier is fundamentally about *how many summands accumulate per output dim* at the model's working precision.

## Implications for future LUT-LM tuning

- **Default `readout_input_nap = 5`** going forward (was 6). Free −18.87M / −6.7% param reduction at no quality cost.
- The gradient-coverage argument generalizes: any LUT module whose `tph × K` exceeds `bs × ctx × n_lookups_into_module / target_coverage` is wasting capacity on rarely-fired rows. For bs=16 at this scale, target coverage ≥ 200 tokens/row → cap `K × tph` at ~50K per "logical decoder" slot.
- The "depth dominates" finding should be re-checked on per-layer modules (out_proj, residual_lut). Out_proj's tph=1024 / NAP=6 (K=64) → ~65K atoms per layer × 6 layers = 393K total atoms in out_proj. Per-row coverage at bs=16: 8192 tokens × 1024 lookups / 65K rows = 128/row. Same regime as read_out. **Hypothesis: out_proj NAP=5 (tph=1024, K=32) would also be a free win, saving another ~75M params.** Worth testing next.

## Trajectory log (exp593 vs exp588, key points)

| step | exp588 | exp593 | gap |
|--|--|--|--|
| 200 | 2.2569 | 2.2602 | +0.003 |
| 1000 | 1.7737 | 1.7743 | +0.001 |
| 2000 | 1.6449 | 1.6474 | +0.003 |
| 3000 | 1.5782 | 1.5794 | +0.001 |
| 4000 | 1.5310 | 1.5289 | **−0.002** ← crossover |
| 5000 | 1.4903 | 1.4876 | −0.003 |
| 6000 | 1.4617 | 1.4602 | −0.002 |
| 7000 | 1.4447 | 1.4433 | −0.001 |
| **8000** | **1.4344** | **1.4337** | **−0.0007** |

Tied through 3K, exp593 ahead 4K onward as LR decay amplifies per-row gradient quality differences.

## Gap to vanilla after exp593

- vs exp328 tied vanilla (1.3882): **+0.0455** (was +0.0462)
- vs exp476 untied vanilla (1.4143): **+0.0194** (was +0.0201)

## Files

- `/home/starost/spiky/nanochat_exps/exp588_read_out_lut/` (1.4344 prior SoTA, dense NAP=6)
- `/home/starost/spiky/nanochat_exps/exp589_readout_multihead/` (killed @ 3K)
- `/home/starost/spiky/nanochat_exps/exp590_readout_sparse/` (killed @ 3.6K)
- `/home/starost/spiky/nanochat_exps/exp591_readout_tph512/` (killed @ 1K)
- `/home/starost/spiky/nanochat_exps/exp592_readout_nap4/` (killed @ 2.6K)
- `/home/starost/spiky/nanochat_exps/exp593_readout_nap5/` — **SoTA**

Parent: [[project_read_out_lut_topology]] (exp588's single-readout-at-end finding).

## 2026-05-27 follow-on: out_proj and qkv sweeps (exp594–exp600)

The NAP=5 read_out win prompted a sweep of the same axis on every other LUT module. **All of these failed to generalize the NAP=5 read_out win.** Recording the negative results so future sessions don't re-explore the same dead ends.

### Out_proj sweep (exp594–exp597) — all worse than exp593

| exp | knob | atoms/L | width | depth (=tph) | tokens/row/step | result |
|--|--|--|--|--|--|--|
| exp593 baseline | NAP=6 tph=1024 | 65K | 384 | 1024 | 128 | 1.4337 |
| exp594 | NAP=5 tph=1024 | 32K | 384 | 1024 | 256 | +0.018 mid (killed) |
| exp595 | NAP=6 tph=1024 sparse n_out=64 | 65K | 64 → scatter | ~170 | 128 | +0.016 mid (killed) |
| exp596 | NAP=7 tph=512 | 65K | 384 | 512 | 64 | +0.016 mid (killed) |
| exp597 | NAP=5 tph=2048 | 65K | 384 | **2048 (2×)** | 256 | +0.005 mid (killed) |

**Out_proj finding**: NAP=6 + tph=1024 is the local optimum. Both the K=64→32 cut (exp594, lost expressivity) and the K=64→128 + half-tph (exp596, lost gradient coverage) failed by similar amounts. Sparse-scatter exp595 also failed despite preserving atoms (mixture depth dominated). **Even doubling tph at NAP=5 (exp597) only got within +5 mb at +0 params** — bandwidth-for-quality at out_proj is much less efficient than at read_out. **Why out_proj is different from read_out**: out_proj output enters the residual stream `x_lut += out_e` and compounds through 6 layers; any imprecision propagates. Read_out fires once at the end, so its imprecision is one-shot.

### qkv sweep (exp598–exp600) — all worse than exp593

| exp | knob | result | notes |
|--|--|--|--|
| exp598 | qk_lut tph 256→512 (per-dim input coverage 32→64 uses) | +0.011 mid (killed @ 4K) | bigger qk doesn't converge — likely temperature/Lion-step mismatch with doubled mixture |
| exp599 | qk_lut H=6→1 (vanilla-equivalent shared routing, single matrix sliced into heads) | **+0.0115 final** | per-head routing in qk_lut is structurally load-bearing; LUTs can't reuse the vanilla "one matrix, sliced" trick |
| exp600 | Unified qkv multi-NAP H=1 (NAP=4 tph=256 + NAP=6 tph=320, n_out=1152=H·(2·d_qk+d_v)) | **+0.0295 final @ +104M params** | combines exp599's H=1 collapse with multi-NAP depth; mixture depth uplift can't offset 6× routing entropy loss |

**qkv finding**: per-head routing diversity in attention input projections is REQUIRED. exp599 nailed this — single-head with same params/bandwidth still loses 11.5 mb. exp600 extended the failure mode by adding v to the shared routing, and the +104M param budget didn't help. **Vanilla attention's "W_q is one matrix sliced into heads" pattern does not port to LUT-LM** because vanilla's per-head computation is just an output partition of a dense matmul; LUT-LM's per-head routing is a structural choice that informs WHICH rows fire, not just which slots get written.

### Out of the 7 negative results, the single positive: exp593 (NAP=5 read_out) remains SoTA.

Pareto-relevant takeaways:
- **Don't cut tph or atom count on per-layer LUTs** (out_proj, qkv) — they compound through depth.
- **Don't collapse multi-head LUTs to H=1** — per-head routing diversity matters in attention input projections.
- **Bandwidth-for-quality is cheap on read_out, expensive on per-layer modules.**
- The next direction for bandwidth/quality is likely **scaling exp593's recipe (E, D, residual_tph) further**, NOT topology changes on existing modules.

## Files (continued)

- `/home/starost/spiky/nanochat_exps/exp594_out_proj_nap5/` (killed @ 5.2K)
- `/home/starost/spiky/nanochat_exps/exp595_out_proj_sparse/` (killed @ 2.8K)
- `/home/starost/spiky/nanochat_exps/exp596_out_proj_tph512_nap7/` (killed @ 4.4K)
- `/home/starost/spiky/nanochat_exps/exp597_out_proj_nap5_tph2048/` (killed @ 6.4K)
- `/home/starost/spiky/nanochat_exps/exp598_qkv_tph512/` (killed @ 4K)
- `/home/starost/spiky/nanochat_exps/exp599_qkv_singlehead/` (final 1.4452 @ 261M)
- `/home/starost/spiky/nanochat_exps/exp600_unified_qkv_multinap/` (final 1.4632 @ 365M)
- `/home/starost/spiky/nanochat_exps/exp601_tied_head/` (killed @ 3K, +0.048 widening)
- `/home/starost/spiky/nanochat_exps/exp602_v_lut_nap5_tph512/` (killed @ 1K, +0.034 widening — v_lut atoms load-bearing)
- `/home/starost/spiky/nanochat_exps/exp603_v_lut_nap6_tph512/` — **SoTA = 1.4295 @ 289.4M**
- `/home/starost/spiky/nanochat_exps/exp608_v_lut_tph512_bs96/` — bs=96 SoTA = 1.2180 @ 289.4M (loses to vanilla bs=96 = exp609 1.1812 at matched compute)
- `/home/starost/spiky/nanochat_exps/exp609_vanilla_rope_bs96_8k/` — vanilla bs=96 = 1.1812 @ 23.2M (matched-compute reference)
- `/home/starost/spiky/nanochat_exps/exp610_E96_mhlut_smooth/` — exp567 fork with MultiHeadLut(smooth, n_alt=3) (killed @ 1.8K, +0.013 vs TinyMHLut soft)
- `/home/starost/spiky/nanochat_exps/exp611_hybrid_smooth/` — **NEW bs=16 LUT-LM SoTA = 1.4048 bpb @ 289M, 1.73h. Beats exp603 by −0.0247 at matched arch.** backward_mode='hybrid_smooth' added to TinyMHL. Math iterated through 3 versions: (a) `u = 0.5/(1+|d|/T_soft)` → +0.009 vs exp603 (ad-hoc, wrong T placement); (b) `u = 0.5/(T_soft+|d|)` → +0.012 worse (T_soft<1 makes u>0.5, alt dominates main); (c) **`u = sigmoid(−2|d_min|/(T_sel·(T_soft+|d_min|)))` = exact top-2 softmax over main+alt** → **−0.025 vs exp603 (new SoTA)**. The (c) formula is mathematically the exact top-2 softmax probability that approximates soft-mode's full K-row softmax forward, derived analytically from `ts[main]−ts[alt] = 2|p[p_star]|` where `p[i] = d[i]/(T_soft+|d[i]|)`. Uses BOTH learnable temperatures. Forward: `out = (1−u)·W[main] + u·W[alt]` where alt = main XOR (1<<argmin|d|). Weight grad: 2-row scatter `(1−u)·grad → main, u·grad → alt` (chain rule of forward). Input grad: still delegated to soft K-row surrogate (`_soft_lut_bwd_body`), mixing two maths. Open: try fully self-consistent backward (only chain rule through u, no soft K-row surrogate) — especially with n_alternatives=NAP (full Hamming-1 ball, no topk needed). Code: `_hybrid_smooth_lut_fwd_body`, `_TinyMHLutHybridSmooth`, `_hybrid_smooth_weight_grad` in `src/spiky/lutorch/tiny_multi_head_lut.py`. **Gap to untied vanilla (exp476=1.4143): −0.0095 — LUT-LM finally beats untied vanilla at matched bs=16 horizon.**
- `/home/starost/spiky/nanochat_exps/exp604_out_proj_pyramid/` (killed @ 5.4K, pyramid [2048,2048,1024,1024,512,512] lost — late taper to 512 hurts)
- `/home/starost/spiky/nanochat_exps/exp605_out_proj_heavy_early/` (final 1.4325 @ 339.7M, +0.003 — heavy-early [2048,2048,1024×4] lever weaker than at exp303 scale)
- `/home/starost/spiky/nanochat_exps/exp606_vanilla_lut_ffn/` (final 1.4736 @ 167M; vanilla MinimalGPT with FFN replaced by TinyMHLut NAP=6 tph=1024; **+0.085 vs vanilla**)
- `/home/starost/spiky/nanochat_exps/exp607_vanilla_lut_ffn_nap4/` (final 1.4951 @ 54M; same as exp606 but NAP=4 K=16; **+0.107 vs vanilla**)

## 2026-05-27 finding: half-LUT hybrid (vanilla + LUT-FFN) does NOT work (exp606/607)

Both attempts to replace ONLY the FFN of vanilla MinimalGPT+RoPE with a TinyMultiHeadLut lost decisively to both vanilla and full LUT-LM. **Vanilla attention + LUT FFN is strictly worse than either fully vanilla or fully LUT.**

| variant | params | trunk BW | total BW | final bpb | vs vanilla | vs LUT SoTA |
|--|--|--|--|--|--|--|
| vanilla exp328 | 23.2M | 339 Mb | 742 Mb | 1.3882 | — | −0.0413 |
| exp603 (full LUT-LM SoTA) | 289M | 169 Mb | 572 Mb | 1.4295 | +0.041 | — |
| exp606 (vanilla + LUT FFN NAP=6 tph=1024) | 167M | 189 Mb | 591 Mb | **1.4736** | **+0.085** | +0.044 |
| exp607 (vanilla + LUT FFN NAP=4 tph=1024) | 54M | 189 Mb | 591 Mb | **1.4951** | **+0.107** | +0.066 |

**Trajectory pattern**: both LUT-FFN variants START 40-50 mb ahead of vanilla at step 200 (LUT routes quickly to good initial mappings), then VANILLA progressively catches up + overtakes around step 2400-2600 (the dense MLP's `Linear → GELU → Linear` keeps learning structure that the LUT lacks an analog for), and by step 8K vanilla is 85+ mb ahead.

**Why LUT-FFN fails (interpretation)**: 
- The vanilla MLP has a 2-stage nonlinearity: project into 4× wider space, apply GELU (genuine nonlinear gating), project back. The LUT version has ONE routing-then-sum stage — equivalent to a piecewise-constant function with K=64 (or 16) pieces per table.
- A single LUT layer cannot replicate the function class of MLP. It does provide bandwidth savings (3× cheaper trunk per layer), but at the cost of capacity.
- This is consistent with full LUT-LM's success: when ALL the trunk modules (qk, v, out_proj, read_out) are LUT, the routing decisions compose across layers and recover composition power. Half-hybrids break that composition.

**Practical takeaway**: don't mix LUT and dense in the same block. Either go fully LUT (LUT-LM family) or fully vanilla. The half-hybrid loses both ways.

**Net session result (exp601-607)**: 7 attempts, 1 SoTA improvement (exp603 v_lut tph 320→512 = 1.4295), 0 ties, 6 losses. Closed dead-ends: tied head; v_lut NAP cut; out_proj per-layer schedules; vanilla+LUT-FFN hybrid (both NAP options). exp603 remains the SoTA recipe.

## 2026-05-27 finding: v_lut tph 320→512 works (exp603 SoTA)

After all the negative results, the winning lever was simple **+60% tph on v_lut at unchanged NAP=6**. Why this works where everything else failed:

- **No K cut** → atoms preserved at K=64 (exp602 with K=32 lost +0.034, confirming v_lut atoms ARE load-bearing — opposite of read_out's slack atoms).
- **No gradient-coverage penalty** → per-row coverage stays at 128 tok/row/step (tph and rows grow together).
- **Pure mixture-depth + atom-count addition** → both go +60%, no axis traded against another.
- **No per-layer compounding catastrophe** → v_lut output feeds SDPA which mixes across tokens before the next layer composes; imprecision doesn't cascade like out_proj's does.

Cost: +28.3M params (+11%), +14.1 Mbits trunk (+9%), +20% wall-clock (1.04h → 1.22h).
Gain: −0.0042 bpb (−4.2 mb).

Trajectory matched the "bigger LUT warmup penalty then late-phase win" pattern (slow start, crossover step 3400, lead grew to −0.004 final). Same general shape as exp593's NAP=5 read_out, just on a different module.

**Lesson for future tuning:** the "tph scaling at fixed NAP" lever (more tables, same depth-per-table) might be cleanly scalable across multiple modules. Untested elsewhere at this scale: read_out_tph 1536 → 2048, out_proj tph 1024 → 1536. Both predictable wins-or-losses given the new picture.
