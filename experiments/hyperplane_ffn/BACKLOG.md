# hyperplane_ffn — experiment backlog

Ideas captured but **not** built or queued. Nothing here is running. Each entry says
what it is, why it is worth doing, and what it depends on.

---

## Ternary variant: trainable per-hyperplane bias

**Status:** idea only — test after the balanced-ternary init run (exp_g_0032).

> Future ternary variant: trainable per-hyperplane bias. One scalar DOF per
> hyperplane, retained at deployment (unlike the shadow weights which collapse to
> ternary q). Routing becomes `ternary(<w,x> + b)` in training and `ternary(<q,x> + b)`
> at inference — i.e. a trainable per-hyperplane threshold / dead-zone offset. Cheap
> (adds/subs + one bias compare per hyperplane at inference). Note: earlier we
> deliberately DROPPED the bias for the plain ternary class; this is a distinct,
> deliberate re-introduction of just one bias DOF, to test whether letting each
> hyperplane place its dead zone off-center helps once q is sparse.

**Why it is not a reversal of the earlier decision.** `TernaryHyperplaneMultiHeadLUT`
drops the bias on purpose: the routing test is `<q, x> > 0`, the pure anchor-pair
"compare to zero" form, and `hyperplane_bias` survives only as a frozen zero buffer.
This idea re-introduces *one* scalar per hyperplane deliberately and for a specific
reason — not to undo that, but to ask whether an off-centre dead zone is worth its one
DOF once the routing is sparse.

**Why it might matter.** The dead zone is currently symmetric about zero:
`q = 0 ⟺ |<w,x>| ≤ T·ln3`, the same band for every hyperplane. A per-hyperplane bias
lets each one shift that band along its own projection axis, so a hyperplane can be
selective about *which side* it goes quiet on. With `b` the test becomes
`|<q,x> + b| ≤ T·ln3`, and the +1 / 0 / −1 split per hyperplane stops being tied to the
symmetry of the input distribution.

**Cost, and why it survives deployment.** The shadow weights `w` collapse to ternary
`q` and are discarded; `b` does not — it is a single fp scalar per hyperplane that
stays. At inference the projection is still multiplication-free:
`sum(x[i] : q_i=+1) − sum(x[i] : q_i=−1)`, then one compare against `−b` instead of
against `0`. So the added inference cost is one scalar per hyperplane in memory and
nothing in arithmetic. Parameter cost during training is `n_tables × nap` scalars —
24,576 for the exp_g_0030 shape, i.e. +0.007%.

**Depends on / to decide first**
- Run exp_g_0032 (balanced ternary init) first. If the ternary values turn out to be
  frozen regardless of init, an extra bias DOF is not the thing to fix next.
- Whether `b` should be ternarized or quantized too — the note says no, keep it a
  cheap continuous scalar.
- Whether `b` belongs in the no-decay group with the rest of the LUT parameters. The
  existing rule is "the LUT module's parameters are not weight-decayed", which would
  put it there; worth stating explicitly rather than inheriting by accident.

**Implementation sketch.** Additive to `TernaryHyperplaneMultiHeadLUT`: un-freeze
`hyperplane_bias` back into a Parameter behind a constructor flag (default off, so
existing runs are untouched), and let it flow to the inherited autograd Functions,
which already accept and differentiate a bias at that argument position. No new
backward code.
