# neurodarwinism

Research chapter: **can a spiking (LIF) network be *evolved* — not gradient-trained — to
reproduce a trained LUT policy's output ordering?**

The teacher is the exp19 Walker2d actor, a `FastMultiHeadLut` with a logsumexp readout
(return 5373.9). The student is a spnet reservoir of 800 excitatory + 200 inhibitory LIF
neurons, latency-coded in and time-to-first-spike out, whose **entire synapse list is the
genome**. A steady-state ALife loop — K nets alive at once, EWMA fitness, cull the worst,
replace with mutated clones of fitness-weighted survivors — is the only optimiser. No
backpropagation touches the spiking network at any point.

- **Tracking issue:** [#93](https://github.com/anatoli-starostin/spiky/issues/93)
- **Methodology:** [`claude/experiment-methodology.md`](../../claude/experiment-methodology.md)
- **Engine dependency:** this chapter is the reason
  [PR #92](https://github.com/anatoli-starostin/spiky/pull/92) exists — four CUDA bugs in
  `synapse_growth` / `connections_manager` surfaced here as unkillable hangs. **exp001 was
  ended by one of them.** Everything from exp002 on ran against the fixed engine, which is
  now in `main`.

## Layout

```
src/                 the reusable code. harness.py is the substrate; steady_state.py is the loop.
data/                the 100K-pair distillation dataset + the teacher checkpoint that made it
exp001_… exp006_…    one directory per run, chronological
```

Each `exp0NN_*/` holds `README.md` (hypothesis, config, result), `config.json`
(**reconstructed** — see the caveat below), `summary.json` (the headline numbers),
`history.json` (the per-round record) and `run.log` (the run's own stdout, the only place
some numbers survive).

### src/

| file | what |
|---|---|
| `harness.py` | **the substrate.** Geometry constants, group-aligned weight builder, delay-meta table, `LatencyEncoder`, genome construction/mutation, `run_episode`, `kendall_tau_b`, `fitness`, `own_null`. Was `es_harness.py`; the `es_` prefix was a misnomer — it is not an ES driver. |
| `data.py` | `load()` / `sample_batch()` — the only way anything touches the teacher data. Was inside `es_smoke.py`. |
| `steady_state.py` | **the loop.** Pool, EWMA, cull/clone, structural + weight mutation, the drive gene, checkpoint/resume. This is the file to read first. |
| `supervise_run.py` | relaunches a wedged or crashed run from its checkpoint. Written because of the engine hangs; kept because runs are hours long. |
| `eval_heldout.py` | scores a saved checkpoint's best member on the held-out set in a fresh process, `--repeats N` times. Written to recover exp004's lost number; keep using it, because the in-run evaluation builds the wrong meta bank and a single build is not a measurement. |
| `lut_ttfs.py`, `student.py`, `real_snn.py` | the gradient-trained comparison line (learnable race front-end, TTFS student, direct LIF student). Not part of the evolution loop. |
| `analytics/` | post-hoc, read-only. `run_all.py` regenerates the CPU-safe figures into `analytics/out/` (gitignored). |
| `tests/` | invariant checks against the spnet engine — chunk validation, sort order, multi-meta builds, and the weight-placement checks that caught the group-alignment bug. |

**Running an experiment.** Point the loop at its own directory so it writes beside its
README rather than into `src/`:

```sh
cd experiments/neurodarwinism/exp007_your-idea
ND_OUT=$PWD python -u ../src/steady_state.py --pool 32 --rounds 300 --tag _yourtag \
    > run.log 2>&1 &
```

`ND_OUT` selects the output directory, `ND_NPZ` overrides the dataset path, and
`ND_CKPT_DIR` tells the analytics where the (gitignored) checkpoints are. **Never run the
`tests/` under pytest** — the spnet suite reports spurious fixture errors there.

> **config.json is reconstructed, not recorded.** The runs pre-date this layout and their
> exact command lines were never saved. Each `config.json` marks which fields were *observed*
> in the run's own log header and which are CLI defaults or inference. Future runs should
> write their own `config.json` at launch; that is the first thing to fix.

## The six experiments

| # | run | rounds | peak τ_b | held-out (in-run) | held-out (refit) | wall | what it tested |
|---|---|---:|---:|---:|---:|---:|---|
| [001](exp001_k32-baseline) | K=32 baseline | 136/∞ | +0.3498 | — | — | 534 s | does the loop climb at all? |
| [002](exp002_k128-pool) | K=128 pool | 300 | **+0.4308** | +0.3742 | **+0.3277** ±0.0101 | 6772 s | does 4× the pool buy 4× the search? |
| [003](exp003_small-sparse-80exc-20inh) | 80 exc / 20 inh, ~1k syn | 300 | +0.3026 | +0.2490 | **+0.0000** ±0.0000 | 215 s | how much of the score is capacity? |
| [004](exp004_wide-exc-delays-1-48) | delays 1–48, 96 metas | 300 | +0.3995 | — (bug) | **+0.2761** ±0.0321 | 2153 s | is the delay range the bottleneck? |
| [005](exp005_tie-penalty-0.1) | fitness − 0.1·tie_rate | 208/300 | +0.3720 | — | **+0.3122** ±0.0093 | 1225 s | can selection be pushed off ties? |
| [006](exp006_drive-gene) | evolvable output drive | 300 | +0.4012 | +0.3166 | **+0.3125** ±0.0135 | 1824 s | can the net evolve its own spike timing? |

τ_b is Kendall tau-b against the teacher's action ordering, corrected by each model's **own**
label-shuffle null. exp001 never reached a held-out evaluation (the engine hung) and no
checkpoint survives, so it has neither column.

**The two held-out columns are different regimes. Do not compare across them** — read the
section immediately below before using any number in this table.

## A warning about every held-out number here

Recovering exp004's missing held-out score turned up two problems that apply to the whole
chapter. Both were found by re-running evaluations, not by reading code.

**1. Rebuilding the same genome does not give the same network.** Ten independent builds of one
saved genome, scored on the same held-out set, spread **±0.01 to ±0.03** in corrected tau —
exp004's ten draws run from +0.2225 to +0.3170. Every single-shot held-out number in this
chapter, including all the in-run ones, is one draw from that distribution reported without its
uncertainty. Differences smaller than ~0.05 between two runs are not resolvable by one
evaluation each. `src/eval_heldout.py --repeats N` exists because of this; use it.

**2. The engine fix changed the answers, so pre-#92 numbers do not reproduce.** The in-run
held-out column was produced by the engine *before*
[PR #92](https://github.com/anatoli-starostin/spiky/pull/92) — in particular before the
backward-groups hash-key overlap that corrupted delay readings for odd meta indices was fixed.
Re-scoring the identical checkpoints on the fixed engine gives materially different results,
and `analytics/tie_rate.py` reproduces the shift on its own metrics too (exp002 teacher
agreement 0.574 → 0.613, exp004 0.632 → 0.583). Genomes that evolved against buggy dynamics do
not carry over to correct ones.

**exp003 is the extreme case and deserves flagging on its own: on the fixed engine it is
completely dead.** 83 % of its outputs never fire inside the 96-tick window, five of six
dimensions are silent, only 2.0 distinct ticks per state remain, and corrected tau is exactly
**0.0000** — chance. Its recorded +0.2490 was measured on a net whose behaviour the current
engine does not produce. The "83× fewer synapses for 70 % of the score" result in
[exp003's README](exp003_small-sparse-80exc-20inh) should be treated as **unverified** until it
is re-run from scratch on the fixed engine; a 1k-synapse net is exactly where corrupted delay
handling would do the most relative damage.

The refit column is internally consistent (one engine, one method, 10 builds each) and is the
one to use for comparisons. It also flattens the chapter's headline: exp002, exp005 and exp006
land within 0.016 of each other, which is inside the build noise. On the fixed engine, **the
only clearly separated results are exp004 below the rest and exp003 at chance.**

## What we learned

> Points 1 and 5–6 below are stated in *training* τ_b and the pre-#92 analytics, because that
> is the regime the experiments were run and compared in. Where the refit column changes the
> conclusion, it says so.

**1. Pool size is the lever — on training score. It does not survive the refit.** K=32 → K=128
took peak +0.3498 → +0.4308 and in-run held-out to +0.3742, at 12× the wall clock, and nothing
else tried here beat the baseline by as much. But re-scored on the fixed engine, exp002
(+0.3277 ±0.0101), exp005 (+0.3122 ±0.0093) and exp006 (+0.3125 ±0.0135) sit within 0.016 of
each other — roughly one build-noise σ apart. **The K=128 advantage is visible in training and
not resolvable on held-out data at this measurement precision.** Establishing it properly needs
a K=32-vs-K=128 pair re-run on the fixed engine with repeated evaluations, which nothing here
has.

**2. The pool collapses to a single lineage, long before it stops improving.** exp005 and
exp006 are the only runs that logged `n_lineages` — the number of round-0 ancestors still
represented — and both go **32 → 1**. Every member of the final pool descends from one
founder. The softer proxy agrees across all six: best-minus-mean falls below 0.01 and stays
there at round 24 (exp003), 55 (exp006), 63 (exp004), 82 (exp005), 92 (exp001), 126 (exp002),
and the run then keeps going for hundreds of rounds. At the end of exp002 the *entire*
128-member pool sits in a band of σ = 0.0022 (min +0.3628, max +0.3765). Pool size only
postpones it — even K=128 collapses by round 126 of 300. What is running is a hill-climber
with K restarts, not a population search.

**3. Weights saturate to a two-valued distribution.** In every run the median excitatory
weight is exactly **0.0** and the histogram is bimodal — dead, or pinned at the ceiling:

| run | dead (\|w\| < 0.6) | at the w_max ceiling | exc/inh ratio |
|---|---:|---:|---:|
| exp002 K=128 | 76.8 % | 20.4 % | 4.13 |
| exp003 small | 69.2 % | 29.7 % | 5.05 |
| exp004 delays | 88.0 % | 10.9 % | 4.06 |
| exp005 tie-pen | 75.7 % | 18.1 % | 4.11 |
| exp006 drive | 74.7 % | 24.0 % | 4.05 |

Inhibitory weights are pinned at −5 by Dale's law in all of them. So mutation is not
exploring a continuum; it is flipping synapses on and off, and ~3/4 of the genome is inert.

**4. The delay dimension is barely used, and widening it made that worse.** At d_max = 20,
23–31 % of excitatory synapses carry non-negligible weight. exp004 quadrupled the range to
d_max = 48 and the figure fell to **12.0 %** — the |w|-weighted mean delay stayed at ~5.2
ticks in *both*. The extra delays were allocated and then left empty.

**5. The readout is degenerate: the network agrees with the teacher barely above chance.**
Pairwise ordering agreement with the teacher is **0.5738** (exp002), **0.6050** (exp003),
**0.6324** (exp004) against a 0.5 coin-flip floor — and 11–22 % of output pairs are exact
*ties*, because only 3.4–4.7 distinct first-spike ticks are produced per state across 6
outputs. The teacher never ties. Discount the ties and agreement jumps to 0.73–0.79, which
locates the problem precisely: **the ranking the net does produce is decent; it just cannot
produce enough distinct ranks.**

**6. Both attempts to fix the ties failed, in different ways.** exp005 penalised the tie rate
directly in the fitness; ties **more than doubled anyway** (0.115 → 0.256) and the peak came
in below baseline — a −0.1·tie_rate term is worth at most −0.029 against a τ_b range of ~0.4,
so selection simply paid the fine. exp006 gave each net an evolvable scalar on its output
drive so it could spread its own spikes; the gene random-walked (pinned at its 3.0 upper clip
through rounds 175–225, back to 0.862 by the end) and held-out came in at +0.3166. Its unit
check shows why that hardly mattered: lowering the drive *does* delay first spikes
(24.95 → 34.57 ticks) but **compresses** them (σ 4.12 → 1.89). There is a shallow optimum near
drive 0.6–1.0 worth 0.110 → 0.084 of tie rate, and a cliff below it (0.254 at drive 0.15) —
but the pool finished at 0.862, *inside* that optimum, while its tie rate still rose
0.131 → 0.246. The knob's entire authority is ~0.026; the problem is ~0.115. It is an order of
magnitude too weak to be the fix.

## Where this points

The bottleneck is not search and not capacity — it is **temporal resolution at the readout**.
Six outputs are being ranked from 3–4 distinct spike ticks. Anything that raises the number
of distinguishable output ticks (more ticks in the readout window, a graded readout rather
than pure first-spike, a per-output threshold that evolves) attacks the measured problem;
more rounds, more pool, or more synapses do not.

Also worth doing before anything else: **log the per-member fitness vector each round**. The
history only stores best and mean, so per-round pool variance — the thing every diversity
claim above needs — is recoverable only at the final round, from the checkpoint. That one
change makes the collapse measurable rather than inferred.
