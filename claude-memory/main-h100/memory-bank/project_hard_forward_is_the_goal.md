---
name: hard-forward-is-the-goal
description: "Strategic directive (2026-05-20): soft mixture forward is unaffordable (defeats the matmul-free LUT goal). exp428 is the standing SOTA. Research focus = better TRAINING of hard-forward mode, not changing the forward."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Hard forward is the goal; soft forward is off the table (2026-05-20, user directive)

After exp444 (soft mixture forward, 1.4821) and exp445 (decomposition), the user set firm direction:

**We cannot afford the soft forward — it ruins the whole idea.** The premise of LUT-LM is matmul-free inference: the hard forward is a single table lookup per slot (`W[chosen]`), no multiplications, no softmax. A soft mixture forward (`Σ_k sel_soft[k]·W[k]`) reintroduces ~2^NAP lookups + softmax + weighted sum per table — it defeats the matmul-free purpose. See [[bitattention-matmulfree]] / project_bitattention_matmulfree.

## Standing positions
- **exp428 (1.4983 @ 89.4M, bs=16, 8K) is the best effort aligned with the final goal** (cheap hard forward, single lookup).
- **exp444 (1.4821) is an UPPER-BOUND reference only** — what the function can reach if the forward were soft. NOT deployable, NOT a SOTA to chase as a config. Do NOT "default to hard=False" (that earlier advice in the exp444 note is retracted).
- The −16.2 mb exp428→exp444 gap is the prize: it's a FUNCTIONAL gap (smooth blended output is a better function than a discrete lookup), and exp445 proved it is NOT recoverable by training-side weight-gradient tricks at hard forward.

## Research focus going forward
Find better ways to TRAIN the hard-forward model — close the gap toward exp444's quality while keeping hard (single-lookup) inference. Promising directions consistent with the goal:
- **Distillation**: soft-forward teacher (exp444-class) → hard-forward student. Transfers the functional advantage into the hard tables. (Needs a teacher checkpoint — exp444 saved none; a re-run with `torch.save` would be required.)
- **Soft→hard annealing during training**: temperature curriculum that sharpens softmax to argmax by the end, so the learned tables are good even when hardened (distinct from post-hoc hardening, which exp386 showed destroys a soft model by +0.25 bpb).
- **Architecture/bandwidth at hard forward**: richer hard function via more/smaller tables (cf exp390 NAP=4 tph=2048 hard-forward win) — within the matmul-free budget.

Dead ends (do not revisit for the hard-forward goal): soft weight-gradient (exp360, exp445 = neutral); cheap optimizer-side levers (exp353–361, all ±0.01); gradient-space variance reduction (exp368/369). See [[soft-wgrad-neutral-exp445]].
