# Evolution ideas

## Compositional warm-starting of evolution (Anatoly's idea, 2026-07)
Hypothesis: evolution may PRESERVE COMPOSITIONALITY.

Setup: we have several LUTs. Suppose we have an EA that can find a spiking network
fitting each single LUT (one net per table — and fitting a SINGLE table is much
cheaper/easier than the whole). Now we want to fit the SUM of the tables (how the
tables behave together under summation).

Idea: instead of evolving the summed target from scratch, take the already-fit
per-table spiking networks, MERGE them somehow (mechanism TBD — e.g. pack disjoint
sub-nets + add aggregator neurons via the growth API, or weight-space blend), and
use that merged network as the INITIAL STATE / seed population for the evolution
that fits the sum.

Why it matters: if this works, evolution only needs to discover the (small)
aggregation glue rather than rediscover every table from noise → potentially
enormous speedup. The path 'single table -> several tables' becomes cheap. This
mirrors our analytic construction (per-table black boxes + a listening aggregator),
so the merged seed plausibly starts near a real solution.

Open question: the exact MERGE operator. Candidates: (a) disjoint packing of the
per-table sub-nets into one SpikingNet (now clean with per-edge explicit weights,
no meta-cap pressure) plus grown aggregator neurons; (b) some weight-space
composition. Worth testing whether warm-started evolution converges dramatically
faster than cold-start on the summed target.
