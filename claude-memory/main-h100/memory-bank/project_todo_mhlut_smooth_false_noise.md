---
name: TODO 2026-05-12 — test MultiHeadLut(smooth_mode=False, nap=3) + bernoulli noise on small deltas
description: Extend the low-confidence-noise regularization insight (exp257) from TinyMHLut(soft) to the standard MultiHeadLut(smooth=False) WTA path. Test whether bernoulli noise on near-tied comparisons gives the same regularization benefit.
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
## What to run (queued by user 2026-05-11 evening, for 2026-05-12)

Test configuration:
  - `MultiHeadLut` (NOT TinyMHLut, NOT SoftMultiHeadLUT — the original
    standard module in `src/spiky/lutorch/multi_head_lut.py`)
  - `smooth_mode=False` (hard STE backward, WTA path)
  - `n_alternatives=3`
  - Bernoulli noise on small deltas (analogue of `argmax_noise_eps` from
    TinyMHLut soft mode)

**Why:** Validated 2026-05-11 (exp257) that explicit low-confidence-bit
noise injection in TinyMHLut(soft) recovers the bf16 implicit regularization
worth ~0.013 bpb. The same physical phenomenon — "small |d_i| → argmax is
unstable → bf16 rounding flips it" — presumably also exists for the standard
MultiHeadLut argmax. If the same trick works there, it confirms the
regularization is a *universal LUT lever*, not specific to soft-pipeline
backward.

**Where to add the noise:** in MultiHeadLut's forward index computation
(`d > 0` -> bit-pack). Need to add a configurable `argmax_noise_eps` and
the same forward/backward consistency pattern (bits reconstructed from
saved index in backward, so the same flipped bits are used everywhere).
User clarification (2026-05-11 evening): use the same noise mechanism as
the current exp260 — bernoulli flip at low-confidence positions, applied
identically in forward and backward.

**Comparators:**
  - exp234 (1.6212 @ 8K, vanilla TinyMHLut STE) → check WTA path same horizon
  - exp257 (1.6060 @ 8K, TinyMHLut soft + noise) → ceiling reference
  - If the new run reaches ~1.605 @ 8K, the noise hypothesis is universal.
  - If it stays near exp234, the soft backward is genuinely needed as a
    substrate for the noise.

**File reference:** `src/spiky/lutorch/multi_head_lut.py` (Conv2DLut,
MultiHeadLut, UnfoldConfiguration). The constraint comment says
"Phase 2 constraint: n_alternatives=3, smooth_mode=True everywhere" — note
that smooth_mode=False breaks that convention, which is the point of the
test.

**How to apply:** When user resumes 2026-05-12 (or shortly after), this
is the next planned experiment. Show config and wait for approval before
launching per `feedback_show_exp_description_before_launch.md`.
