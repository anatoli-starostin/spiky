---
name: For long-horizon forks, never compare step-by-step to the short-horizon source
description: When a fork only changes n_steps (e.g. 8K -> 48K), the longer warmup makes per-step bpb incomparable to the source. Compare only against same-horizon runs or final result.
type: feedback
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
Don't compare per-step bpb of a long-horizon fork (e.g. 48K) against its
short-horizon source (e.g. 8K) — even if the configs are otherwise identical.

**Why:** `lr_warmup_fraction` is a fraction of n_steps, so a 6x longer run has
a 6x longer warmup. At step N in the 48K run the LR is roughly 6x smaller
than at step N in the 8K source. The bpb at the same step will be
systematically worse in the longer run for the entire warmup phase (and
even early post-warmup), which says nothing about the recipe's quality at
the longer horizon. Per-step comparisons are *intrinsically misleading*.

**How to apply:** When monitoring a 48K (or similarly longer) fork:
  - Compare against same-horizon runs only (e.g. exp235=1.4906 @ 48K,
    exp229=1.4958 @ 48K, exp041=1.3406 vanilla @ 48K).
  - Or compare only at the *same fraction of total steps* (e.g. step
    1600 of 8K vs step 9600 of 48K — both at 20% through training).
  - Final bpb is always a valid comparison.

Mentioned by user 2026-05-11 during exp260 (48K fork of exp257 8K) live
monitoring — I had been reporting "exp260 step N vs exp257 step N" deltas
which were meaningless during the warmup-dominated first ~5000 steps.
