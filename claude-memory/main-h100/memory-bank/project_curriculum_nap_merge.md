---
name: curriculum-nap-merge
description: "NAP curriculum experiments (exp396-exp398) — train smaller NAP first, merge into bigger NAP via additive weight formula. Mechanism works (smooth transitions, function-preserving merge), but final-stage NAP=8 architecture was bloated and undertrained at bs=16. Curriculum DIDN'T beat baseline at this scale."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# NAP curriculum via additive merge — mechanism works, didn't beat baseline (2026-05-17, exp396-exp398)

User's design idea: start with many small-NAP tables (NAP=1 leaves, 8× more tables than target), train fast with good gradient coverage, then progressively MERGE sibling pairs into bigger-NAP parents via the additive formula:
```
parent[bits_A, bits_B, :] = child_A[bits_A, :] + child_B[bits_B, :]
```
This is **function-preserving at the moment of merge** — sum of two children's outputs equals one parent's output.

Anchor tree: top-down sampling from final NAP=8 architecture using `CANONICAL_FULL_COVERAGE`, then recursively split each table's 8 anchor pairs into binary tree → 1024 NAP=1 leaves at start, merge back to 128 NAP=8 root.

## exp396: 4-stage NAP=1→2→4→8 (each 4000 steps)
- Stage 0 (NAP=1, bs=2): 1.9920 final
- Stage 1 (NAP=2, bs=4): 1.9084 final
- Stage 2 (NAP=4, bs=8): 1.8510 final  
- Stage 3 (NAP=8, bs=16, killed early): on track to ~1.77 at full 4000 steps
- **Issue**: per-stage cosine LR had repeated warmup → stage transitions had +0.05 hiccup. User suggested switching to continuous cosine over all 16K steps.

## exp397: 3-stage NAP=2→4→8 with continuous cosine LR
- Skip NAP=1 leaves entirely (first merge is param-preserving anyway, so NAP=1 was redundant — same param count as NAP=2)
- Stage 0 (NAP=2, bs=2, 8000 steps): 1.8871 final
- Stage 1 (NAP=4, bs=8, 4000 steps): killed mid-way, was at 1.83 around step 3600
- **Hiccup pattern**: still +0.045 at stage transitions despite continuous LR. Identified Adam state reset as culprit.

## exp398: Same as exp397 + Adam state carry-over (averaged sum)
- Adam state for LUT params merged via averaged additive formula:
  `parent.m[bits_A, bits_B] = (child_A.m[bits_A] + child_B.m[bits_B]) / 2`
  (sum was tried first in exp398v1, doubled magnitude → worse hiccup)
- Stage 0 final: 1.8858
- Stage 1 final: 1.8261 (modest hiccup +0.043 at start, recovered fast)
- **Stage 2 final: 1.7684** — way slow descent, didn't approach exp365 baseline
- Total: 16000 steps, 112K bs-units (~93% of exp365 compute)

## Why curriculum underperformed at final stage

1. **Architecture bloat**: snapping qkv and residual_lut to NAP=8 (when exp365 had them at NAP=6) created an 87M model — 2× baseline params. More capacity to train, but with bs=16 not enough gradient samples.
2. **LR too low at stage 2**: continuous cosine had decayed LR to ~30% of peak by stage 2 start. The merged model has 8× LUT params relative to stage 1 — the NEW cross-interaction entries need real training, but low LR limited update magnitude.
3. **bs=16 at NAP=8 = original collapse problem**: same gradient-coverage issue that exp365 had, now affecting 8× more params. Stage 2 descent at ~0.001 per 100 steps.
4. **Stage transition hiccup persistent**: Adam carry helped slightly (avg merge: hiccup +0.043 vs no-carry +0.045 vs sum-merge +0.053). Not eliminated; merged weights' "cross-interaction" entries need real training regardless of optimizer state.

## What we learned (mechanism-level)

- **Additive weight merge works** — function-preserving, smooth transitions for stages with same-magnitude param count
- **Adam state carry is delicate**: must AVERAGE (not sum) for m and v to maintain correct magnitude. Sum-merge is wrong by 2× factor.
- **Early stages (NAP=1, NAP=2) train fast** — bs=2 with NAP=1 reaches similar quality to bs=16 NAP=1 with 1/8 the compute (see project_grad_accum_reproduces_big_batch.md)
- **Final stage IS the bottleneck**: the curriculum can only matter if the final stage's architecture is matched to the data + batch budget

## How to apply / what's next

- **DO NOT use this exact curriculum recipe as-is** — it underperforms baseline.
- For future curriculum experiments:
  - **End at NAP=4 or NAP=6, not NAP=8**. Match the empirical sweet spot from exp390/exp391/exp392 (knee at NAP=6 / 4× bandwidth).
  - **OR drastically reduce target tph** for qkv/residual at NAP=8 to keep params matched to exp365 (e.g. qkv tph=4 at NAP=8 instead of 16).
  - **More compute in stage 2** (final stage) — biggest model needs the most training.
  - **Higher LR at stage 2** if continuous cosine: maybe interrupt the schedule with a mini-warmup after each merge.
  - **Larger batch at final NAP=8 stage** to address coverage problem (bs=64 or higher).
- **Mechanism is documented and code is reusable**: `anchor_tree.py` and `inject_merged_adam_state` in exp398/train.py can be adapted.

## Files

- `anchor_tree.py` — `build_anchor_tree` (top-down split), `merge_weight_tensor` (additive merge)
- `exp398_curriculum_adam_carry/train.py` — orchestrator + per-stage trainer + Adam state injection
- exp398 checkpoints saved (stage0.pt, stage1.pt, stage2.pt) for any post-hoc analysis
