# exp002 — K=128 pool

**Hypothesis.** exp001 collapsed to a near-uniform pool by round 92 while still improving.
If diversity is the binding constraint, quadrupling the pool should buy proportionally more
search: more lineages alive, later collapse, higher peak.

**Setup.** Identical to exp001 except **K=128, cull 32 per round**. Same 800/200 architecture,
same ~97k synapses per genome, same delays 1–20. `config.json` is reconstructed.

## Result — the best run of the chapter, and the only clean win

| | |
|---|---|
| rounds | 300 / 300, `rc=0`, 0 restarts |
| peak corrected tau-b | **+0.4308** at round 281 |
| final EWMA best / mean | +0.3765 / +0.3732 |
| **held-out corrected tau-b** | **+0.3742** (member 54, 84,626 synapses) — *in-run, pre-#92 engine* |
| held-out, refit on the fixed engine | **+0.3277 ± 0.0101** over 10 builds (`heldout_eval.json`) |
| pool collapsed at round | 126 |
| synapses per net | 97,100 → 85,228 |
| wall | 6772 s |

Peak rose **+0.0810** over exp001 (+0.3498 → +0.4308) for **12.7× the wall clock**. That is
the largest improvement anything in this chapter produced, and it is also the least surprising
one: it is what you get by paying for it.

Held-out (+0.3742) sits essentially on the final training EWMA (+0.3765), so the pool is not
overfitting the resampled batch stream — the ceiling is real, not memorisation.

> **Refit caveat.** Re-scored on the post-#92 engine over 10 builds this genome gives
> **+0.3277 ± 0.0101**, and exp005 (+0.3122) and exp006 (+0.3125) land within 0.016 of it —
> about one build-noise σ. The K=128 advantage is clear in *training* score and is **not
> resolvable on held-out data** at this precision. See
> [the chapter README](../README.md#a-warning-about-every-held-out-number-here).

## What the checkpoint says

**Diversity is gone, and pool size only postponed it.** Collapse arrives at round 126 instead
of round 92 — 34 rounds later for 4× the population — and the run then continues for another
174 rounds inside the collapsed regime. At the final round, the spread over **all 128 members**
is σ = **0.0022** (min +0.3628, max +0.3765). Selection is choosing between siblings.

**Weights are two-valued.** Median excitatory weight is exactly **0.0**; 76.8 % are dead
(|w| < 0.6) and 20.4 % sit at the 45.0 clip ceiling (1.5 × w_max). Inhibitory weights are all
pinned at −5 by Dale's law. Three quarters of the genome is inert, and mutation is effectively
a switch rather than a dial. exc/inh ratio 4.13.

**Delays go mostly unused.** Only 23.2 % of excitatory synapses carry non-negligible weight;
the |w|-weighted mean delay is 5.20 ticks against a mean of 9.73 over the full 1–20 range.
The load sits at the short end. (exp004 tests whether widening the range helps. It does not.)

**The readout is the bottleneck.** Pairwise ordering agreement with the teacher is **0.5738**,
barely above the 0.5 coin flip — but **21.9 % of output pairs are exact ties**, and the network
produces only **3.42 distinct first-spike ticks per state** across 6 outputs. The teacher never
ties. Excluding ties, agreement is **0.7345**. The ordering the net produces is respectable;
it just cannot produce enough distinct ranks to express one.

## Reading

More pool buys score, but it buys it by brute force and does not touch any of the three
structural pathologies above. exp005 and exp006 are the two attempts to attack the tie problem
directly; both failed.
