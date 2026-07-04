---
name: project_magnitude_leakage_softmax_package
description: "exp500 STE (hard signs + softmax) = 1.5437 — WORSE than clean hard argmax (exp428/475 ~1.496). Soft signs + softmax are an inseparable package; the whole soft apparatus buys only ~0.018 bpb over hard forward, and it's NOT hardenable (magnitude leakage)."
metadata:
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

**exp500 (2026-05-23): fork of exp493 (MatmulMHLut softmax, 1.4806) + STE on the soft sign.**
After `p = d/(T_soft+|d|)`, use HARD signs (±1) on forward, soft-sign gradient on
backward: `p = p + (sign(d) − p).detach()`. Forward `ts` becomes the exact integer
Hamming score → `softmax(ts/T_sel)` is the exact exp-Hamming kernel. Flag `hard_sign_ste`
on `MatmulMultiHeadLut` (threaded into `_matmul_mhlut_fwd_body`), config `hard_sign_ste:true`.

**Result: exp500 = 1.5437 @ 8K, +0.0631 vs exp493.** Lost monotonically from step 2000 on
(per-1000 deltas −0.025 → +0.001 → +0.028 → +0.043 → +0.050 → +0.057 → +0.060 → +0.063,
plateauing ~+0.06). (No checkpoint — forked exp493's pre-fix train.py, hit the same
`lut_optimizer` save-bug NameError; result/trajectory in metrics.csv.)

**The key cross-experiment framing (user's, 2026-05-23):**
| exp | forward | bpb |
|-----|---------|-----|
| exp475 | hard argmax (single row, STE) | 1.4962 |
| exp428 | hard argmax (single row, STE) | 1.4983 |
| **exp493** | **soft signs + softmax** | **1.4806** |
| **exp500** | **hard signs + softmax** | **1.5437** |

1. **The whole soft apparatus buys only ~0.018 bpb** over the honest hard forward
   (exp428 1.4983 → exp493 1.4806). Small.
2. **Soft signs + softmax are an INSEPARABLE PACKAGE.** Breaking it (hard signs + soft
   row-mixture) is worse than EITHER clean extreme — worse than full-soft (exp493) AND
   worse than full-hard argmax (exp428/475) by +0.046.

**Mechanism (grounded):** softmax row-mixing is only useful as the *readout of a
continuous soft-sign coordinate*.
- Soft signs (exp493): `ts` continuous → upstream shapes adjacent-Hamming rows into a
  meaningful interpolation; mixing carries signal.
- Hard signs (exp500): neighbor rows are independently-learned table entries one bit
  away → mixing them is pure noise. exp500's `T_sel` actually ROSE to mean 0.510 (vs
  exp493's 0.374): at T_sel=0.51 each Hamming-1 neighbor gets ~2% of the winning row's
  weight × 6 neighbors ≈ 12%+ of the output blurred onto noise rows. Can't escape via
  T_sel→0 because the soft-sign BACKWARD makes the temp optimizer "think" it's soft and
  raise T_sel for good gradients — a forward(hard)/backward(soft) mismatch.

**MAGNITUDE LEAKAGE (the deeper why exp493 isn't a real LUT).** Probe of `|p|` on the
exp494 checkpoint (= exp493 model), 4 val batches, all 24 modules:
- mean|p|=0.42, median 0.43, **frac|p|>0.9 = 0.00 across EVERY module type** (qkv 0.35,
  v 0.45, out 0.43, residual 0.33). `|p|` histogram is ~UNIFORM on [0, 0.7] then tapers —
  not a pile at 1.0. A sign-router would saturate; instead the model uses every magnitude
  level ~equally (max-entropy continuous feature).
- Independent corroboration: exp493 learned `T_soft ≈ 0.47–0.78`, never →0 (it pays to
  stay soft).
- ⇒ exp493's 1.4806 is a **continuous rational network wearing LUT structure**, NOT a
  discrete lookup. It is fundamentally **not hardenable**: hard signs (T_soft→0) destroy
  the analog channel it's built on. The honest matmul-free / single-lookup number remains
  **exp475 = 1.4962**; the −0.018 "win" of exp493 is bought entirely with magnitude
  leakage that can't survive hardening (a true hard LUT needs BOTH T_soft→0 and T_sel→0).

**Hard-sign gate sweep (exp501-505, 2026-05-23):** re-ran the MatmulMHL gates with hard signs
(`hard_sign_ste`), all vs softmax-hard (exp500 1.5437) and argmax (exp475 1.4962):
relu_norm 1.6621, layernorm 1.6401, signed (killed, +0.32), square_norm k=2 (too diffuse,
winner ~5% weight), square k=4 (winner ~12%, ~+0.13). **softmax beats EVERY gate under hard
signs too** — its exp-Hamming-kernel shape dominates even with its own temperature trap; the
polynomial square kernel is too diffuse (neighbor-shell multiplicity 6@h=1, 15@h=2) and has no
learnable sharpness. No hard-sign MatmulMHL gate beats the honest argmax line (1.4962).
`gate_mode` square (g0^k unnorm) / square_norm (g0^k/Σ) + `gate_power` knob added to the lib.

**exp506 (hard-sign BACKWARD isolation):** exp475 + `set_hard_sign_bwd(True)` — softmax over
INTEGER Hamming scores in the soft backward. exp475's argmax fwd is sign-only so forward AND
weight grads are bit-identical; ONLY the upstream input gradient changes. Cost ~**+0.037** —
the soft-sign magnitude helps even purely as backward gradient shaping, but modestly (the big
win is the soft FORWARD on qkv, see [[project_hybrid_qkv_soft_localization]]).

Probe script: `nanochat_exps/exp494_softmax_temppenalty/probe_magnitude.py`; verify_claims.py
(invertible-kernel + temperature-trap numerics) in exp500. Relates to
[[project_matmul_mhlut_softmax]] (softmax wins the gate sweep — but only with soft signs),
[[project_hybrid_qkv_soft_localization]], and [[project_lut_convergence_bottleneck]].
