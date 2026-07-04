---
name: exp415-curriculum-nap3-to-6-negative
description: "NAP=3→6 curriculum with bs=4→bs=32 on exp414 arch loses to plain exp413 at matched compute by ~50 mb persistent gap. Killed mid-stage 1. Stage 0 (NAP=3, bs=4) doesn't produce features that stage 1 (NAP=6, bs=32) can refine cheaply."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp415 — NAP=3→6 + bs change curriculum (FAILED, 2026-05-17)

## Setup
2-stage curriculum on the exp414 architecture (E=32, residual_dim=384, 31.53M target params):

- **Stage 0**: NAP=3, 2× tph everywhere (qkv 32, v 256, out 512, res 64), bs=4, 4000 steps, standard cosine + 10% warmup. Total stage 0 = 8.2M tokens.
- **Stage 1**: NAP=6, exp414 target tph (qkv 16, v 128, out 256, res 32), bs=32, 3500 steps, RESET LR, **NO warmup, pure cosine anneal**. Total stage 1 = 57.3M tokens.
- Weight merge between stages: outer-add `parent[a*8|b] = child_A[a] + child_B[b]` from `anchor_tree.merge_weight_tensor` (exp400 framework).
- Adam state carried via AVG merge (×0.5 on m, v) for shape-changed LUT params.
- All four LUTs participate (qkv, v, out, residual).
- Total compute = 65.5M tokens = exactly matches exp413 (bs=16, 8K steps) reference.

## Result
Killed at stage 1 step 1400/3500 (~31M cumulative tokens) — gap to exp413 at matched tokens was +50 mb and not closing.

### Trajectory vs exp413 plain bs=16 at same cumulative tokens
| Cum. tokens | exp415 | exp413 | Δ (mb) |
|---|---|---|---|
| 8.2M (stage 0 end) | 1.8943 | 1.8904 (step 1000) | +4 |
| 11.5M (s1 step 200) | 1.9067 | 1.8416 (step 1400) | **+65** ← jump-up at merge |
| 14.7M (s1 step 400) | 1.8610 | 1.8106 (step 1800) | +50 |
| 18.0M (s1 step 600) | 1.8335 | 1.7821 (step 2200) | +51 |
| 24.6M (s1 step 1000) | 1.7947 | 1.7432 (step 3000) | +51 |
| 31.1M (s1 step 1400) | 1.7632 | 1.7137 (step 3800) | +50 |

## Analysis
1. **Forward IS preserved by the merge** (corrected note from user: forward is HARD argmax, not softmax). Stage 1's argmax over K=64 = concat(stage 0's argmax over K_A=8, stage 0's argmax over K_B=8), and the merged weights satisfy T_C[a*8+b] = T_A[a] + T_B[b], so the merged stage-1 model produces identical output to stage 0 at t=0.
2. **Backward path is NOT preserved**: at NAP=6 the score range is 2× larger (sum of 6 anchor pair contributions vs 3 in stage 0), so the K=64 softmax in backward is sharper than two independent K=8 softmaxes summed — different gradient flow even from "identical" weights.
3. **Stage 1 starts at peak LR with no warmup** → first ~100-200 steps destabilise the merged weights before the eval at step 200 (visible jump-up from 1.8943 to 1.9067).
4. **Stage 0 budget too small for what it's asked to do**: 8.2M tokens at bs=4 means very noisy gradients on a model with 18.1M params and 4.47M LUT entries. The features learned aren't strong enough that stage 1 can refine them cheaply — stage 1 essentially has to relearn from scratch, paying for both stage 0's lost compute AND its bad init.
5. **The only prior curriculum to beat baseline was exp400** (NAP=4→8, out_proj ONLY, matched bs across stages, +2 mb improvement). Generalising to "all modules + NAP=3→6 + bs change" lost it entirely.

## How to apply
- Don't run multi-module curriculum with stage-0 NAP < 4 — too restrictive.
- Don't change batch size between stages — kills the per-token comparison framing.
- Don't skip warmup in stage 2 — the merge transition needs the high-LR phase to NOT be too disruptive. The exp400 recipe used a fresh 10% warmup per stage.
- Curriculum is fragile; the working exp400 recipe (single-module out_proj NAP=4→8, equal stages, same bs, warmup per stage) seems to be on the edge of what works. Don't expect wider applications to be free.

## Files
- `nanochat_exps/exp415_curriculum_nap3_to_6_bs32/{train.py, anchor_tree.py, config.json}`
- Anchor tree helper is generalisable (handles any even NAP), unlike exp400's power-of-2-only `build_anchor_tree`.
