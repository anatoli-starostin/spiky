# Handcrafting the walker2d LUT teacher, stage 3: the OUTPUT stage

Investigation + design only. Nothing built, nothing run, no shared code touched.

## (a) Which artifacts are canonical

**Two `walker2d` directories exist and they are NOT duplicates — they are different chapters.**

| dir | what it is | size | newest file |
|---|---|---|---|
| `experiments/walker2d-lut/` (hyphen) | **the canonical one for this work.** Holds `exp19_lut-lse-expmlpcrit-t32`, the teacher our chapter distils | 484 MB | 2026-08-09 |
| `experiments/walker2d_lut/` (underscore) | a *different, earlier* line — `exp_c01_sac_baseline` … `exp_c36b_seeds3to5`, SAC/MJX/JAX scaffolding | 1.1 GB | 2026-08-06 |

Neither is stale relative to the other; they are separate experiment series that happen to
collide in name. **`walker2d-lut` (hyphen) is the one that matters here.**

Caveat worth knowing: `experiments/walker2d-lut/src/` now contains **only `__pycache__`** — the
`models.py` source is gone from the working tree. The docstring quoted below was recovered from
the compiled `models.cpython-312.pyc`. The `data/README.md` reference to
`src/spiky/lutorch/fast_multi_head_lut.py::_exp_outputs_fwd` lines 165–188 **no longer resolves**:
that function is not in the current file, which has since been rewritten around
`_FastMHLutSoft` / `_FastMHLutHybridSmooth`. So the formula below is *not* taken from the
line reference in the README — it is confirmed three independent ways (see below).

### The dataset

`experiments/neurodarwinism/data/distill_exp19_100k.npz`, 21.6 MB, committed to git.

| array | shape | dtype | meaning |
|---|---|---|---|
| `x` | (100000, 17) | float32 | raw observation |
| `x_norm` | (100000, 17) | float32 | **the normalised obs the LUT indexes with** |
| `y_action_mean` | (100000, 6) | float32 | **the target: 6 real outputs** |
| `y_prelog` | (100000, 6) | float32 | the sum-of-exponentials *before* the log |
| `weights` | (32, 64, 6) | float32 | the tables |
| `anchor_a`, `anchor_b` | (32, 6) | int64 | the comparison pairs |
| `tau` | () | float64 | 0.09036568 |
| `tables_per_head` | () | int64 | 32 |
| `exp_clamp` | () | float64 | 60.0 |

100,000 on-policy pairs from a real walking session (250 envs × 400 steps). Split: the **last
4,000** are held out and never enter training.

## (b) The exact output formula

**17 inputs → 32 tables × 64 cells → 6 outputs.** One head, so "sum over tables within a head"
and "sum over all 32 tables" are the same reduction.

```
d[b,t,j]      = x_norm[b, anchor_a[t,j]] - x_norm[b, anchor_b[t,j]]     # [B,32,6]
index[b,t]    = sum_j (d[b,t,j] > 0) * 2^(5-j)                          # [B,32], a 6-bit code 0..63
w_sel[b,t,o]  = weights[t, index[b,t], o]                               # [B,32,6]
z             = clamp(w_sel / tau, -60, +60)
out[b,o]      = tph * tau * ( log( sum_t exp(z[b,t,o]) ) - log(tph) )
              = tph * tau * log( (1/tph) * sum_t exp( w_sel[b,t,o] / tau ) )
```
with `tph = 32`, `tau = 0.09036568`.

**Answering the question precisely: the logsumexp is over the 32 TABLES, not over cells within a
table.** Each table contributes exactly ONE value — the cell its 6-bit code selects. There are no
per-cell weights beyond the stored value itself; `tau` is the temperature and `tph` the scale.
**Each of the 6 outputs has its own independent logsumexp** over the same 32 selected cells
(the cell *choice* is shared across outputs, the *values* are per-output).

The `1/tph` inside the log makes it a smooth **mean**, not a smooth max: as `tau → ∞` it tends to
`tph * mean_t(w_t)`, and as `tau → 0` toward `tph * max_t(w_t)`.

Confirmed three independent ways:
1. `data/README.md`;
2. the surviving docstring in `models.cpython-312.pyc`: `out = T * tau * log( (1/T) * sum_t exp( w_t / tau ) )`;
3. **I recomputed `y_action_mean` from the `.npz` alone in numpy — max |diff| 1.3e-06** against
   the stored labels, pure fp32 rounding.

The clamp never binds on this data: largest observed `|w/tau|` is **5.941** against a limit of 60.

## (c) The proposed spiking output stage

### The mechanism: a decaying LIF membrane computes logsumexp in spike time

This is exact, not an analogy. Take one output neuron per output dimension, receiving an impulse
of amplitude `A_t` from each of the 32 cell-neurons at time `t_t`. With a LIF membrane the
contribution of each impulse decays as `exp(-(T - t_t)/tau_m)`, so at time `T`

```
V(T) = sum_t A_t * exp( -(T - t_t) / tau_m )
```
and the threshold condition `V(T) = theta` rearranges to

```
T = tau_m * log( sum_t A_t * exp( t_t / tau_m ) ) - tau_m * log(theta)
```

**The output spike time IS a logsumexp of the input spike times.** That is exactly the shape of
the LUT readout, so the mapping is structural rather than approximate.

### Two encodings, and why one of them is not buildable here

**(B) value in the WEIGHT — matches the "fire simultaneously" assumption, and FAILS.**
Fire all 32 cell-neurons at a common `t_0` and set `A_t = exp(w_t / tau)`. Then
`T = tau_m * LSE_t(w_t/tau) + const`, correct. **But `w/tau` spans ±5.941, so the weights must
span `exp(11.88) ≈ 1.4e5` of dynamic range.** On the quantised `{0, 0.1, …, 1.0}` grid — or any
fixed-point weight — that is impossible. This kills the simultaneous-firing design.

**(A) value in the SPIKE TIME — buildable, and the one I propose.**
Keep every synapse at a common amplitude `A` and encode the value in *when* the cell fires:

```
t_t = (tau_m / tau) * w_sel[b,t,o] + c            # LATER spike = LARGER value
T   = tau_m * LSE_t( w_sel/tau ) + const
out = (tph * tau / tau_m) * ( T - const ) - tph * tau * log(tph)
```

**`out` is an exact AFFINE function of the output neuron's first-spike time** — precisely the form
the existing `diagls` readout (a·spike_time + b, least-squares) already fits. So this stage needs
no new readout at all.

Note this **contradicts the stated "neurons fire almost simultaneously" assumption**: the value
*must* live in the spread of firing times, because time is linear in `w` and can span the range,
whereas weight would have to be exponential in `w` and cannot. Simultaneous firing would mean all
32 values are equal. Worth settling before stage 1 and 2 are designed around the other assumption.

Two conveniences fall out: all synapses are a single positive amplitude, so **Dale's law is
satisfied trivially** (the output stage is purely excitatory even though `w_sel` is signed — the
sign lives in the timing offset `c`); and `T >= max_t(t_t)` automatically, so the output neuron
always fires after the last cell-neuron with no sequencing logic needed.

### Expected approximation error

Two sources, both governed by `tau_m`, and they trade against each other.

| | |
|---|---|
| **required tick spread** | `(tau_m/tau) * range(w) = tau_m * 11.88` ticks |
| **relative error per tick** of quantisation | `exp(1/tau_m) - 1` |
| **error in output units per tick** | `tph * tau / tau_m = 2.892 / tau_m` |

Against `var(y_action_mean) = 1.279` (sd 1.1309), with ±0.5-tick uniform rounding error:

| `tau_m` | spread needed | error per tick | MSE from quantisation | as % of output variance |
|---|---|---|---|---|
| 20 | ~238 ticks | 5.1% | 0.0017 | **0.14%** |
| 10 | ~119 ticks | 10.5% | 0.0070 | 0.55% |
| 5 | ~59 ticks | 22% | 0.0279 | 2.2% |

**The output stage is essentially lossless if we can afford a wide enough tick window**, and the
cost is entirely the window, not the mechanism. `tau_m = 10` (≈119 ticks, 0.55%) looks like the
sensible operating point; the 128-tick input window already validated in the LIF fixes work is
the same order.

Caveats not yet measured: the engine takes 2 Euler half-steps per tick, so the true decay is
`(1 - 0.5/tau_m)^2` per tick rather than `exp(-1/tau_m)` — a fixed, correctable reparameterisation
of `tau_m`, but it should be calibrated rather than assumed. And this analysis assumes the output
neuron does not spike before all 32 inputs arrive, which holds for `A` small enough that no single
impulse crosses threshold — a condition to check numerically when building.

### What this implies for stages 1 and 2

Stage 3 needs each cell-neuron to fire at a time **affine in its stored value**, at a common
amplitude. So stage 2 (lookup) must deliver "the selected cell of table `t` fires at
`(tau_m/tau)*w + c`" — a *timed* spike, not merely a selected one. That is a stronger contract
than "one neuron per cell fires when selected", and it is the interface stage 2 must be designed
against.
