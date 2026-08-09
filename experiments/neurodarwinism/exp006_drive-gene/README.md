# exp006 — evolvable output-drive gene

**Hypothesis.** exp005 showed ties cannot be *priced* out. So give the network a knob instead
of a fine: a per-net scalar multiplying the weights of every synapse targeting an output
neuron. Turn it down and the output neurons integrate more slowly, so their first spikes land
later — and, the hope went, further apart. Make the scalar heritable and mutable and let
selection find the value that maximises separation.

**Setup.** exp002's architecture at K=32, delays 1–20, 300 rounds, `--drive-gene` with
`lo 0.3, hi 1.5` (initial range), `sigma 0.2`, `p 0.5` (mutation rate), `clip [0.1, 3.0]`.

**Implementation note.** The drive is carried as an array **parallel to `ewma`/`age`**, not as
a genome key. `dedupe` and `mutate_structural` index every genome key as a per-synapse array,
so a scalar stored there corrupts both.

## Result — the gene random-walked

| | |
|---|---|
| rounds | 300 / 300, `rc=0`, 0 restarts |
| peak corrected tau-b | **+0.4012** at round 175 |
| final EWMA best / mean | +0.3530 / +0.3514 |
| **held-out corrected tau-b** | **+0.3166** (member 2, 80,869 synapses) |
| pool collapsed at round | 55 |
| **tie rate, round 0 → 300** | **0.1310 → 0.2463** (max 0.2939) |
| **live lineages, round 0 → 300** | **32 → 1** |
| wall | 1824 s |

Held-out +0.3166, below exp002's +0.3742 at 3.7× less compute — and the tie rate rose by the
same factor as in exp005 despite the mechanism existing specifically to lower it.

## The gene's trajectory (`drive_trajectory.json`, from the 25-round snapshots)

| round | 0 | 25 | 50 | 100 | 150 | **175** | 200 | 225 | 250 | 299 |
|---|---|---|---|---|---|---|---|---|---|---|
| pool mean | 0.795 | 0.542 | 0.953 | 1.437 | 1.696 | **2.877** | 2.866 | 1.719 | 1.102 | **0.862** |
| pool max | 1.493 | 0.910 | 1.910 | 2.257 | 2.312 | **3.000** | 3.000 | 3.000 | 1.433 | 1.005 |

It drifts down, then up, pins members against the **3.0 upper clip** through rounds ~175–225,
then falls back and finishes at 0.862 with the whole pool inside [0.73, 1.01]. That is a random
walk, not a selection gradient. If drive mattered, the gene would have found a value and stayed.

The fitness peak (+0.4012) does land at round 175, when drive was pinned at the ceiling. **Do
not read that as evidence.** It is a single coincidence in a single run, the peak decayed while
drive stayed high through round 225, and the final score is unremarkable. n = 1.

## Why the knob was too weak — `check_drive_gene.py`

The unit check in this directory holds one net and one batch fixed and sweeps the drive scale,
reporting first-spike statistics per setting rather than asserting a direction, so a null
result stays visible (`check_drive_gene.out` is its output):

| drive | mean first-spike tick | **std** | distinct ticks / state | **tie rate** |
|---:|---:|---:|---:|---:|
| 1.50 | 24.95 | 4.12 | 4.66 | 0.110 |
| 1.00 | 26.57 | 3.87 | 4.86 | 0.092 |
| **0.60** | 28.53 | 2.74 | **4.89** | **0.084** |
| 0.30 | 31.75 | 2.05 | 3.53 | 0.232 |
| 0.15 | 34.57 | 1.89 | 3.22 | 0.254 |

Lowering the drive does exactly half of what the hypothesis needed: spikes move **later**
(24.95 → 34.57) but get **tighter** (σ 4.12 → 1.89). The knob *translates* the first-spike
distribution rather than spreading it.

There is a shallow optimum around **0.6–1.0** — ties fall from 0.110 to 0.084 — and then a
cliff: below 0.6 the compression dominates and ties nearly triple, to 0.254 at drive 0.15.

**That optimum is the damning part, not a consolation.** The pool finished at a mean drive of
**0.862**, i.e. sitting inside the best region the mechanism has to offer — and the run's tie
rate still went 0.131 → 0.246. The whole dynamic range of this gene is worth about **0.026** of
tie rate; the run's ties grew by **0.115** regardless. The knob is roughly an order of magnitude
too weak to matter, so a random walk over it and a perfectly-tuned value would have produced
much the same run.

```sh
python check_drive_gene.py     # needs a GPU; ~1 min
```

## What the checkpoint says

74.7 % of excitatory weights dead, 24.0 % at the ceiling, median exactly 0.0, exc/inh 4.05.
Maximum excitatory weight **58.2**, above the 45.0 clip — that is the drive gene multiplying
output-targeting weights past the STDP ceiling, which is expected and confirms the mechanism
was actually active. Delays: 25.3 % effective, |w|-weighted mean 5.41 — unchanged from
everything else.

## Reading

Two mechanisms tried, two failures with different causes: exp005 could not *price* ties,
exp006 could not *create separation* — its entire authority over the tie rate is ±0.026, and
it spent the run wandering across a range where most of that was unavailable anyway. Together
they say the problem is upstream of both: the readout produces 3–5 distinguishable ticks and
no per-net scalar changes that. The next thing to try should change the *decode* — window
length, graded rather than first-spike, or an evolvable per-output threshold — not the drive.
