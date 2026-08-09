# exp007 — windowed first-spike readout

> ## STATUS: NOT RUN — second design also fails its pre-flight, for a deeper reason
>
> Everything is implemented, tested and committed: the opt-in `--readout-window`, the
> `--coverage-penalty` training scaffold, and `--seed-genome` warm starting. The 300-round run
> was **not launched**. The pre-flight showed the pool reaching a **flat absorbing state by
> round 5**, and a follow-up measurement found the reason: **late-window network state is not
> reproducible between builds of the same genome — spikes after tick 64 vary 219× — so the
> quantity this experiment selects on is almost entirely build noise.** Evidence below.

## The design that was agreed

1. **Readout window [48,96)** — rank each output on its first spike inside the final 48 ticks,
   re-based to 0..47; pre-window spikes discarded; silent reads 48.
2. **Coverage penalty**, training-only: `fitness = corrected_tau − 1.0 × (mean number of the 6
   outputs with no spike in the window)`. A fully silent member sits at −6, dwarfing tau's ±1,
   so coverage dominates until satisfied. Never applied to the held-out score.
3. **Warm start** from exp002's best evolved genome (member 54, EWMA +0.3765, 84,626 synapses)
   cloned into all 32 slots, because a random genome has almost no output activity for a narrow
   window to see.
4. K=32, 300 rounds, geometry/d_max/stdp_lr/batch cloned from exp001, per-member logging on.

All four are implemented and work. `config.json` holds the exact configuration and launch
command; it is ready to fire unchanged.

## Pre-flight result: the pool dies by round 5

K=8, 8 rounds, warm-started, window [48,96), λ = 1.0:

| round | coverage miss (mean) | coverage miss (best) | fitness min | fitness max | **fitness σ** |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.809 | 5.641 | −5.971 | −5.670 | 0.0960 |
| 1 | 5.963 | 5.906 | −6.000 | −5.900 | 0.0330 |
| 2 | 5.934 | 5.516 | −6.000 | −5.500 | 0.1639 |
| 3 | 5.996 | 5.984 | −6.000 | −5.993 | 0.0032 |
| 4 | 5.998 | 5.984 | −6.000 | −5.979 | 0.0068 |
| 5 | **6.000** | **6.000** | −6.000 | −6.000 | **0.00000** |
| 6 | 6.000 | 6.000 | −6.000 | −6.000 | 0.00000 |
| 7 | 6.000 | 6.000 | −6.000 | −6.000 | 0.00000 |

Round 0 is technically non-flat, but **all eight members are identical clones** — the spread is
build noise, not genetic variation, so selection begins by ranking noise. Coverage then
**falls** instead of rising and pins at 6.000 (no output fires in the window, ever) by round 5.
At that point σ = 0 exactly: every member scores −6.000, the cull is arbitrary, and the state
is absorbing — mutation has no partial credit to climb. Held-out pure tau: **+0.0022**, chance.

## Why: the late window is not reproducible

The obvious next move would be to move the window earlier. The window curve on the warm-start
genome says that does not help either — and then explains why nothing will.

| window | pure tau | coverage miss | tie rate | distinct ticks/state |
|---|---:|---:|---:|---:|
| full 96 (default) | **+0.3299** | 0.000 | 0.208 | 3.78 |
| [8,96) | +0.2752 | 0.000 | 0.230 | 3.64 |
| [16,96) | +0.1054 | 0.000 | 0.317 | 2.95 |
| [24,96) | +0.0082 | 0.000 | 0.953 | 1.13 |
| [32,96) | −0.0477 | 0.000 | 0.812 | 1.47 |
| [40,96) | −0.0593 | 0.010 | 0.327 | 3.25 |
| [48,96) | −0.0432 | 0.116 | 0.108 | 4.68 |
| [56,96) | +0.0180 | 0.860 | 0.058 | 5.28 |
| [64,96) | −0.0009 | 1.464 | 0.093 | 5.01 |

Monotone decline from the default, through zero at [24,96), and negative from there on. Two
distinct failure modes bracket the range: open the window early and the outputs are still
firing continuously so the first in-window spike is the window start for everyone (tie rate
0.95 at [24,96)); open it late and there is nothing in it.

**But the decisive number is this.** Six independent builds of the *same* genome, same states:

| quantity | min | max | ratio |
|---|---:|---:|---:|
| total output spikes | 57,809 | 65,077 | **1.1×** |
| spikes at t ≥ 32 | 17,448 | 24,609 | **1.4×** |
| spikes at t ≥ 48 | 90 | 3,195 | **35×** |
| spikes at t ≥ 64 | 7 | 1,535 | **219×** |
| coverage miss under [48,96) | 1.886 | 5.860 | spans 65 % of the 0–6 range |
| **tau, full window** | **+0.3111** | **+0.3458** | **1.1×** |
| tau, [48,96) | −0.0633 | −0.0052 | negative in every build |

**Variability compounds with simulation time.** Early behaviour is nearly reproducible — which
is why the default TTFS readout is stable to ±0.02 and why every result in this chapter that
depends on early first spikes has held up. Late behaviour is not reproducible at all.

That makes the windowed readout unmeasurable rather than merely bad: the coverage signal the
penalty selects on swings across most of its range between builds of one genome, so selection
would be chasing build noise, and the target readout is worse than chance in all six builds.

(This also resolves an inconsistency in my own earlier numbers for this genome — 42 spikes at
t ≥ 64 in one measurement and ~3,600 in another. Both were correct; they were different builds.)

## What this points at

**The build nondeterminism is now the blocking problem, and it looks like a bug.** `build_pool`
passes a fixed `seed=1` to `_grow_explicit` and compiles with
`shuffle_synapses_random_seed=None`, so the same genome should produce the same network. It does
not, and the divergence grows with simulation time. This is the same family as the four bugs in
[PR #92](https://github.com/anatoli-starostin/spiky/pull/92) — that fixed a concurrent-sort race,
but something is still varying. Chasing it would benefit **every** result in the chapter, not
just this experiment; it is the reason single-shot held-out numbers need `--repeats`.

Until then, any readout that depends on network state after ~tick 32 cannot be measured on this
substrate, so this direction should wait rather than be re-tuned. Tuning λ, moving the window
again, or running longer would all be measuring the same noise.

## What was built (committed, opt-in, defaults unchanged)

| change | where |
|---|---|
| `readout_null(W)`; `run_episode(..., readout_window=W)` slices the raster to its final W ticks, argmax there, 0..W−1, silent → W. `None` = the previous path exactly. | `harness.py` |
| `coverage_miss_per_member()`; `score(..., coverage_penalty=λ)` subtracts λ × missing-output count. Docstring states it is a scaffold that must never reach a reported number. | `steady_state.py` |
| `--readout-window`, `--coverage-penalty`, `--seed-genome`, `--seed-member` | `steady_state.py` |
| Warm-start branch: clone one checkpoint member into all K slots, mutation diversifies from round 1 | `steady_state.py` |
| Per-round `coverage_miss_vec` / `_mean` / `_best` / `_min` logging | `steady_state.py` |
| **Held-out is now pure tau** — both penalties explicitly zeroed in `main()`'s held-out block and in `eval_heldout.py` | `steady_state.py`, `eval_heldout.py` |
| `--readout-window` mirrored so a checkpoint is always scored under the readout it evolved in | `eval_heldout.py` |

**A behaviour fix rode along here:** the old held-out call forwarded `a.tie_penalty`, so an
exp005-style run would have reported `tau − 0.1 × tie_rate` as its held-out score rather than
tau. It never mattered — exp005 was killed before evaluating — but it does now, and held-out is
pure tau from this commit on.

## Still missing, whatever happens next

No matched TTFS control on the post-#92 engine. exp001 ran 136 rounds pre-fix, so comparing any
exp007 number against its +0.3498 would repeat the cross-engine error documented in
[the chapter README](../README.md#a-warning-about-every-held-out-number-here).
