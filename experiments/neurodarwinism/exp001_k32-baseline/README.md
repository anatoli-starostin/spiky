# exp001 — K=32 baseline

**Hypothesis.** A steady-state pool of 32 nets, selected only on corrected tau-b against the
teacher's action ordering, climbs at all. Nothing more ambitious: this run exists to establish
that structural + weight mutation under EWMA selection moves the score off the floor.

**Setup.** 17 in / 800 exc / 200 inh / 6 out, ~97k synapses per genome, K=32, cull 8 per round,
alpha 0.3, grace 2 rounds, delays 1–20. See `config.json` — **it is reconstructed**, not
recorded; the run pre-dates this layout.

## Result — yes, and then the engine killed it

| | |
|---|---|
| rounds | **136** (the run did not end on its own) |
| peak corrected tau-b | **+0.3498** at round 114 |
| final EWMA best / mean | +0.2937 / +0.2862 |
| best − mean at the end | 0.0075 |
| pool collapsed at round | **92** |
| synapses per net | 97,094 → 92,401 |
| wall | 534 s |
| **held-out** | **none — see below** |

The score climbs from +0.08 at round 0 to a peak of +0.3498 by round 114, so the loop works.
It is then flat-to-declining for the last twenty rounds, which is the pattern every subsequent
run repeats.

## Why there is no held-out number

**The run hung at round 136 and never reached the held-out evaluation.** The log stops
mid-flight with no error and no completion line, and a second copy of it was preserved beside
the first as `steady_state_stdp_b64_hung.log`, ending at exactly the same round.

This is the `sort_chains_by_synapse_meta` race later fixed in
[PR #92](https://github.com/anatoli-starostin/spiky/pull/92) — two threads sorting one chain,
producing a cycle that the consumer's chain walk spins on forever. Measured at the time at
roughly **1 build in 35 at K=32**, and 6 hangs in 40 builds at K=128. This run drew the short
straw. `supervise_run.py` was written in response, and every experiment from exp002 on used it.

The peak and the trajectory are unaffected; only the final held-out evaluation is missing, and
no checkpoint survives for this run, so it cannot be recovered without a re-run.

## Notes

- No post-hoc weight / delay / readout analytics for this experiment: there is no checkpoint.
  `analytics/common.py` registers it with `ckpt=None` and the checkpoint-dependent modules skip
  it.
- A slower batch-512 attempt at the same configuration was superseded before this one and its
  history was not kept; only an orphan log remains on `exp/walker2d-lut`.
