# exp003 — small sparse net, 80 exc / 20 inh, ~1k synapses

> ## ⚠ THIS RESULT DOES NOT REPRODUCE ON THE CURRENT ENGINE
>
> Everything below was measured before [PR #92](https://github.com/anatoli-starostin/spiky/pull/92)
> fixed four CUDA bugs, one of which corrupted delay readings for odd meta indices. Re-scoring
> this exact checkpoint on the fixed engine (`src/eval_heldout.py`, 10 builds) gives corrected
> tau-b **+0.0000 ± 0.0000 — chance.** 83 % of outputs never fire inside the 96-tick window,
> five of the six dimensions are silent, and only 2.0 distinct first-spike ticks per state
> survive against the 4.75 recorded here.
>
> A ~1k-synapse net is where corrupted delay handling does the most relative damage, so this is
> the run most likely to have been *measuring the bug*. **Treat the headline below — "83× fewer
> synapses, 70 % of the score" — as unverified.** It is a claim worth re-testing, and cheap to:
> the whole run took 215 seconds. Until then it should not be cited. See
> [the chapter README](../README.md#a-warning-about-every-held-out-number-here) and
> `heldout_eval.json`.

**Hypothesis.** exp002 buys score with 97k synapses per genome and nearly two GPU-hours. If
most of that genome is inert — and the weight histogram says three quarters of it is — then a
net two orders of magnitude smaller should keep most of the score. This is the capacity
control: how much of +0.43 is actually capacity, and how much is the loop finding the same
shallow solution either way?

**Setup.** 17 in / **80 exc / 20 inh** / 6 out, fan-outs scaled 10× down
(`{exc→exc 8, exc→inh 2, inh→ 10, in→ 10, →out 10}`), **1,167 synapses per genome**, K=32,
delays 1–20, 300 rounds. `config.json` is reconstructed.

## Result — 83× fewer synapses, 70 % of the score, 31× faster

| | |
|---|---|
| rounds | 300 / 300, `rc=0`, 0 restarts |
| peak corrected tau-b | **+0.3026** at round 279 |
| final EWMA best / mean | +0.2669 / +0.2662 |
| **held-out corrected tau-b** | **+0.2490** (member 0, **1,011 synapses**) |
| pool collapsed at round | **24** |
| synapses per net | 1,167 → 1,009 |
| wall | **215 s** |

Against exp002: **1.2 %** of the synapses, **3.2 %** of the wall clock, and **70 %** of the
peak. Against exp001's baseline (+0.3498) it loses only 0.047 while running 2.5× faster on a
net 96× smaller.

That is the interesting number in this chapter. It says the 97k-synapse net is not using its
capacity — consistent with the 69–88 % dead-weight fractions measured everywhere — and it makes
the small net the right vehicle for iterating on the *mechanism*, since a full 300-round run
costs under four minutes.

## What the checkpoint says

**Collapse is nearly immediate: round 24.** By the end the whole 32-member pool spans
σ = 0.0028 (min +0.2511, max +0.2669) — and it then runs for 276 more rounds. With a small
genome there is simply less for mutation to differentiate, so the pool converges before it has
searched.

**Weight saturation is worse, not better.** 69.2 % dead, **29.7 %** at the 45.0 ceiling
(the highest fraction of any run), median exactly 0.0, exc/inh ratio 5.05. Squeezing the net
did not make the weights continuous; it pushed more of them to the rail.

**The readout is slightly less degenerate here** — and that is worth noting because it cuts
against the small net elsewhere. Tie rate **0.107** (half of exp002's 0.219) and **4.75**
distinct first-spike ticks per state (vs 3.42). Teacher agreement is accordingly *higher* at
**0.6050** than exp002's 0.5738.

But corrected tau-b is *lower* (+0.2491 vs +0.3757), because the own-null is much larger
(0.0945 vs 0.0355): a small net's orderings are closer to constant, so a label shuffle scores
well against them and more of the raw tau (0.3436) is chance. Excluding ties, agreement is
0.6776 — below exp002's 0.7345. **Fewer ties, but a worse ordering underneath them.**

**Delays.** 30.8 % of excitatory synapses carry non-negligible weight — the highest of any run,
though that is partly just having fewer synapses to dilute. |w|-weighted mean delay 4.11.

## Reading

Capacity is not the binding constraint. A ~1k-synapse net gets most of the way, collapses in a
quarter of the rounds, and finishes in under four minutes. **Use this configuration for
mechanism experiments** — anything that changes the readout, the encoding or the fitness should
be screened here before it costs two GPU-hours at K=128.
