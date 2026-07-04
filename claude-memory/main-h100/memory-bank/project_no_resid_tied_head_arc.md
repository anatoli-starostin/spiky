---
name: project_no_resid_tied_head_arc
description: "exp571-575 arc: minimal arch (no residual_lut + tied dot head) — works after LN-on-emb fix; +0.10 bpb above tied vanilla at 22% bandwidth"
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**Arc target:** strip the LUT-LM to bare essentials (no residual_lut / no D-stream / no Linear unembedder; tied dot head on E-stream) — bound the floor of "matmul-light LM" quality.

**Sequence of single-knob forks on exp567's backbone:**

| exp | head architecture | trajectory key points | verdict |
|--|--|--|--|
| **exp571** | raw `ln_final(x_lut) @ tok_emb_E.T`, no LN on emb | 200/800/1000/2000 = 2.81/1.98/1.91/1.78 | killed step 2000; logit scale ~0.57 at init (small emb std), slow warmup |
| **exp572** | `ln_final` applied to BOTH x_lut and tok_emb_E (shared LN); γ=1.0 init | 200/800/1000/2000 = **4.19**/2.14/2.06/1.89 | killed; high logits at init (γ=1, vector √E ≈ 10) caused softmax to peak CONFIDENTLY on RANDOM wrong vocab → average CE = log(V) at step 200, NOT γ-collapse. |
| **exp573** | SEPARATE ln_final + ln_emb (γ=1 both) | 200 = 4.21 | killed; separating LNs doesn't fix the issue — confirms not collapse-symmetry. |
| **exp574** | separate LNs, **ln_emb.γ init=0.1** (logit scale ~1.0 at init) | 200/800/1000/2000/3000/4000/5000 = 2.62/1.94/1.88/1.76/1.70/1.66/1.63 | killed step 5000 for bandwidth; trajectory was tracking with +0.10 constant offset to exp567. Projected final ~1.55. Validated the small-γ-init fix. |
| **exp575** | exp574 + E=192, d_v=32 (preserves H·d_v = E invariant) | 200/800/1000 = 2.37/1.91/1.86 | killed step 1000; ~+0.03 gap to tied vanilla early. Projected final ~1.45-1.48. |

**Key learnings from this arc:**

1. **Tied dot heads on small-E LUT-LMs need careful LN initialization.** The default `nn.LayerNorm(E)` with γ=1, β=0 makes ln(x_lut)·ln(emb_v) have init std ~√E (≈10 at E=96, ≈14 at E=192). Over V=32768 classes this gives **softmax peaked on a random vocab** at init → **CE = log(V) on average** ("uniform-baseline CE achieved via wrong-but-confident peaks, NOT actual uniform distribution"). Fix: `ln_emb_v2d.weight.data.fill_(0.1)` brings init logit scale to ~1.0 → soft softmax that can learn signal. γ stays learnable for the optimizer to grow back if needed.

2. **Separating LNs (exp573) did NOT fix the init issue** — confirms the failure mode isn't γ-collapse from shared parameters; it's just that γ=1 init gives wrong logit scale for vector dim ~10. Both shared and separate LNs at γ=1 land at step-200 CE = log(V).

3. **LUTs are scale-invariant to embedding magnitude** (because `MeanAbsNorm` strips magnitude before every LUT input). So `tok_emb_E` can have small init for residual mixing AND a separately-scaled head-side normalization for logit temperature — perfect decoupling via the existing norm pathway.

4. **No-residual-lut + tied head trades ~0.10 bpb for ~3× bandwidth reduction** vs exp567. exp574 projection 1.55 at 163 Mbits vs exp567 1.4768 at 475 Mbits. The residual_lut + Linear-head pair was contributing ~100 mb of quality but costs ~312 Mbits.

5. **E-widening compounds on this minimal architecture.** exp575 (E=192, d_v=32) cut the gap to tied vanilla from ~+0.07 (exp574 vs exp328) to ~+0.03 at early steps, while staying at 289 Mbits/tok (39% of vanilla). The bandwidth-vs-loss Pareto frontier is achievable with this style; just needs tuning.

6. **Apples-to-apples reference for tied-head LUT-LMs is exp328 (tied vanilla, 1.3882), NOT exp476 (untied vanilla, 1.4143)**. exp476 is the right reference for exp567/exp562/exp565 (untied) but tied-head experiments compare to exp328. Tied vanilla is +0.026 bpb stronger than untied vanilla, so tied LUT-LMs face a tougher target.

**Open follow-ups:**
- Let exp575 (or its successor) run to completion to lock the E=192-class number.
- E=128, E=256 sweep on this minimal arch for the Pareto frontier.
- V2D-tied head test (exp576, queued) — pairwise-sign dominance in head; +bandwidth, ?loss.

**Files:**
- exp571_no_resid_tied/ — raw tied dot, no LN on emb
- exp572_no_resid_tied_lnemb/ — shared LN tied dot (γ=1)
- exp573_no_resid_tied_lnsep/ — separate LNs (γ=1)
- exp574_lnemb_small_gamma/ — separate LNs, γ_emb init=0.1 (the working recipe)
- exp575_E192_dv32/ — exp574 + E=192/d_v=32 widening
