# exp005 — tie penalty: fitness = τ_b − 0.1 · tie_rate

**Hypothesis.** exp004 showed that giving the network more temporal *room* does not make it
use any. So stop offering room and start charging rent: put the tie rate directly into the
objective. If selection can see ties, it should breed them out.

**Setup.** exp002's architecture at K=32, delays 1–20, `--tie-penalty 0.1`. This run is also
the first to log the extra per-round diagnostics — `pool_std`, `pool_min`, `pool_max`,
`n_lineages`, `tie_rate_mean` — so the collapse is measured here rather than inferred.
`config.json` is reconstructed.

## Result — ties more than doubled, under a penalty designed to suppress them

| | |
|---|---|
| rounds | **208 / 300** (killed, see below) |
| peak corrected tau-b | **+0.3720** at round 175 |
| final EWMA best / mean | +0.3404 / +0.3390 |
| **held-out** | none from the run itself; measured post-hoc: **+0.3122 ± 0.0093** over 10 builds |
| pool collapsed at round | 82 |
| **tie rate, round 0 → 208** | **0.1148 → 0.2557** (max 0.2886) |
| **live lineages, round 0 → 208** | **32 → 1** |
| synapses per net | 97,094 → 86,890 |
| wall | 1225 s |

**The penalty did not work — the tie rate went the wrong way, by a factor of 2.2.** It starts
at 0.115, rises throughout, and ends at 0.256, peaking at 0.289. A fitness term of
−0.1 · tie_rate is worth at most −0.029 at the observed rates, while the tau-b term ranges over
~0.4; selection simply paid the fine. The peak (+0.3720) also came in *below* the K=32
baseline's trajectory at the same rounds, so the penalty cost score without buying separation.

**Ties are not a behaviour selection can price — they are a structural limit.** Whatever the
network is doing to produce more score, it produces more ties as a side effect, and the two
cannot be traded off at this weight. A much larger coefficient would just optimise the penalty
instead of the task.

## The diversity number this run made visible

`n_lineages` — the number of distinct founding ancestors still represented in the pool — goes
**32 → 1**. Every member of the final pool descends from one round-0 net. The `best − mean`
gap (0.0013) and the final pool σ (0.0018) were always consistent with collapse, but this is
the direct measurement: the pool is one lineage wearing 32 costumes.

exp006 reproduces it exactly (32 → 1). This is the strongest evidence in the chapter that the
steady-state loop, as configured, is a hill-climber with 32 restarts rather than a population
search.

## Why the run stopped at 208

**Terminated externally to free the GPU for exp006.** Not a crash: the supervisor log records
the launch and the warm-up and then simply ends — no nonzero-exit line, no `RUN COMPLETE`. The
checkpoint at round 208 is intact, but no held-out evaluation ever ran.

One has since been measured from that checkpoint with `src/eval_heldout.py` on the fixed
engine: **+0.3122 ± 0.0093** over 10 builds (raw τ_b +0.3484, own null +0.0362, tie rate 0.174,
3.98 distinct ticks/state; `heldout_eval.json` has all ten draws). That is **within one
build-noise σ of exp002's refit +0.3277 and exp006's +0.3125** — so on held-out data the tie
penalty neither helped nor measurably hurt. What it demonstrably did was double the tie rate,
which is the finding above and does not depend on the held-out number.

## Reading

Do not re-run this with a bigger coefficient. The result is not "0.1 was too small"; it is
that ties and score move together, so any penalty strong enough to matter will trade away the
thing being measured. exp006 attacks the same problem from the mechanism side instead — and
fails differently.
