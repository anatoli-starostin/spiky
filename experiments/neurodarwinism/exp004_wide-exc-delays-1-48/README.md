# exp004 — excitatory delays widened to 1–48 (96 metas)

**Hypothesis.** The readout is time-to-first-spike, so the only way the network can express a
*rank* is by separating spike times. exp002 measured only 3.42 distinct first-spike ticks per
state across 6 outputs. If the delay range is what limits temporal separation, widening it from
1–20 to **1–48** should give the network more room to spread its outputs and cut the tie rate.

**Setup.** As exp002's architecture (800 exc / 200 inh, ~97k synapses) but at **K=32**, with
`--d-max 48` — **96 synapse metas** instead of 40, one per (delay, sign). 300 rounds.
`config.json` is reconstructed, but **d_max = 48 is measured**: the checkpoint's delays span
[1, 48] over 48 distinct values.

## Result — the extra range was allocated and then left empty

| | |
|---|---|
| rounds | 300 / 300 |
| peak corrected tau-b | **+0.3995** at round 281 |
| final EWMA best / mean | +0.3571 / +0.3540 |
| **held-out** | **none — the run OOM'd at the evaluation, see below** |
| pool collapsed at round | 63 |
| synapses per net | 97,094 → 80,857 |
| wall | 2153 s |

Peak +0.3995 at K=32 is above the K=32 baseline's +0.3498 and below K=128's +0.4308 — so
widening the delays did help somewhat. But the checkpoint shows it did **not** help for the
reason the hypothesis proposed.

**The load did not move outward.** The |w|-weighted mean delay is **5.18 ticks** — against
**5.20** in exp002 at d_max 20. Quadrupling the available range moved the centre of mass by
0.02 ticks. The raw (unweighted) mean delay dutifully doubles to 23.0, because synapses are
*allocated* uniformly across the range, but the ones carrying weight stay at the short end.

**And it made the sparsity worse.** The fraction of excitatory synapses carrying non-negligible
weight fell from 23.2 % (exp002) to **12.0 %** — the lowest of any run — with **88.0 % dead**
and only 10.9 % at the ceiling. Spreading the same synapse budget over 2.4× as many delay bins
halves the number that end up mattering.

**The tie rate barely moved.** 0.2023 here vs 0.2187 in exp002, with 3.71 distinct first-spike
ticks per state vs 3.42. Teacher agreement is the chapter's best at **0.6324** (0.7928
excluding ties), and corrected tau-b **0.3859** — but the mechanism under test contributed
almost none of that.

## Why there is no held-out number

**All 300 rounds completed. The held-out evaluation then died of GPU memory:**

```
RuntimeError: CUDA runtime API error cudaErrorMemoryAllocation
  at native/spiky/misc/misc.cpp:56
  in build_pool -> sp.add_connections   (steady_state.py:731)
```

The evaluation builds a fresh single-member pool for the best genome, and 96 metas at K=32
does not leave room for it alongside the training pool. The supervisor did what it is supposed
to do — resumed from the checkpoint — and hit the identical failure **40 times in a row**
before giving up (`supervisor.log` in this directory records every attempt).

The history is complete and unaffected. **The held-out number is cheap to recover**: evaluate
the saved checkpoint in a fresh process with no training pool resident. Worth doing, since this
run has the highest teacher agreement in the chapter.

## Reading

Delay range is not the constraint. The network does not *use* long delays even when it has
them; it uses the short end and lets the rest go dead. Any future attempt at temporal
separation should act on the readout — the length of the readout window, a graded rather than
first-spike decode, or an evolvable per-output threshold — not on the delay budget.
