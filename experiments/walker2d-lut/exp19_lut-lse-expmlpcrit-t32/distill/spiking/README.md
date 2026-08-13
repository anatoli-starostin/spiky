# exp19's LUT readout as a fixed latency-coded spiking layer — design note + v1 harness

**Headline: the sum-scaled log-sum-exp readout is not *approximated* by a first-spike
layer — under an exponential-kernel TTFS neuron it IS one, exactly.** The action mean comes
out as an exact affine function of the output spike time, and the two codings asked for
(weights vs delays) are *provably the same layer*.

| file | what |
|---|---|
| `lut_ttfs.py` | encoder, race/cell front-end, frozen TTFS readout (variants D and W), decode; run directly for the exactness self-test |
| `train_stub.py` | loads `../distill_exp19_100k.npz`, latency-matching loss, 5-step gradient smoke test |

---

## Part 1 — the exact readout math

Class: **`FastMultiHeadLut`**, `src/spiky/lutorch/fast_multi_head_lut.py` (not `models.py` —
`models.py:475` only *constructs* it). exp19's actor is `models.py:422`
`FastLUTLSESumExpMLPCriticActorCritic`, whose `actor_lut` is

```python
FastMultiHeadLut(input_dim=17, n_heads=1, n_outputs=6, n_anchor_pairs=6,
                 tables_per_head=32, forward_mode="hard", use_bf16=False,
                 exp_outputs=True, exp_outputs_scale="sum",
                 exp_outputs_init="additive", exp_outputs_tau_init=0.05)
```

The whole readout is `_exp_outputs_fwd`, **lines 165–188**. With `B` the batch:

```
 d[b,t,i]    = x[b, anchor_a[t,i]] - x[b, anchor_b[t,i]]          [B, 32, 6]
 bits[b,t,i] = 1[ d > 0 ]                                          [B, 32, 6]
 addr[b,t]   = sum_i bits[b,t,i] * 2^(5-i)                         [B, 32]   MSB-first
 w_sel[b,t,o]= weights[t, addr[b,t], o]                            [B, 1, 32, 6]
 z           = clamp(w_sel / tau, -60, +60)                        line 177
 lse[b,o]    = logsumexp(z, dim=tables)                            line 178
 out[b,o]    = 32 * tau * ( lse - log(32) )                        line 187, scale=="sum"
```

Equivalently, with `S = exp(lse)` — the `y_prelog` of the distillation set:

```
 S[b,o]   = sum_{t=0..31} exp( w[t, addr[b,t], o] / tau )          [B, 6],  strictly > 0
 out[b,o] = 32 * tau * log( S[b,o] / 32 )
```

`tau -> inf` recovers the plain sum over tables (exp10's readout); `tau -> 0` gives
`32 * max_t w`. The learned value is **`tau = 0.09036568`**.

### What the learnable parameters actually are

Introspected from the trained checkpoint (`rerun_ckpt/actor_s1.pt`):

| | shape | learnable? | what one element is |
|---|---|---|---|
| `weights` | **(32, 64, 6)** = 12,288 | **yes** | one scalar: table `t`, address `k`, output dim `o` |
| `exp_outputs_tau_raw` | () | **yes** | the readout temperature, `tau = softplus(raw)` floored at 1e-3 |
| `soft_anchor_a_long`, `soft_anchor_b_long` | (32, 6) int64 | **no — buffers** | which two of the 17 obs dims a race compares |
| `soft_powers` | (6,) int64 | no | `[32,16,8,4,2,1]`, MSB-first bit packing |
| `soft_bit_matrix`, `_table_offset`, `log_soft_score_temp`, `log_select_temp` | — | no | unused on this path (`forward_mode="hard"`, `learnable_temps=False`) |

So the actor has **12,289 learnable scalars**, and *the anchors are fixed* — chosen at
construction by `get_balanced_anchor_pairs`, never trained. In the trained model: 192
(a, b) slots, 136 of them distinct, no self-pairs, and coverage across the 17 obs dims is
balanced (each dim used 20–26 times).

Structure: `n_heads = 1`, `tables_per_head = 32` → **32 tables**, each with
`table_dim = 2^6 = 64` rows → **2,048 rows total**, each row holding a **6-vector** (one
scalar per action dim). Exactly **one row per table is active** per input, so 32 of 2,048
rows fire, contributing 32 × 6 = 192 scalars to the six sums.

Trained weight range: `w in [-0.5369, +0.3084]`, std 0.0923 → `w/tau in [-5.94, +3.41]`,
so the ±60 clamp never binds and `exp(w/tau) in [0.0026, 30.34]`.

### Which parameters map to weights vs delays

`weights` is the **only** per-cell learnable quantity and it enters the readout **solely as
`exp(w/tau)`** — a strictly positive multiplicative factor inside a sum. That is precisely
the form that a first-spike layer can realise *either* way:

* as a **synaptic weight** `W = exp(w/tau) > 0` (variant W), or
* as an **axonal delay** `d = D0 - w >= 0` (variant D), since `exp(-d/tau_m) ∝ exp(w/tau)`
  when `tau_m = tau`.

`tau` is not a per-cell parameter — it is the single global **membrane time constant** of
the output neurons. The anchors are not weights at all; they are the *wiring* of the race
layer. Nothing else is learnable, so there is no third candidate.

### The cell → neuron mapping — resolved, not ambiguous

You asked me to pause if this was ambiguous. There are two readings, but they are the same
computation, so this is a labelling choice rather than a fork in the design:

* **(i) one neuron per LUT row** `(t, k)` → **2,048 neurons**, each with **6 outgoing
  synapses** carrying `w[t,k,:]`. ← **what I built**
* (ii) one neuron per scalar `(t, k, o)` → 12,288 neurons, one synapse each.

Both have 12,288 synaptic parameters and produce identical spike times; (ii) just splits
each row-neuron into six copies of the same coincidence detector. (i) is the literal
reading of "one neuron per table cell" — a cell *is* a row, addressed by 6 bits — and it is
6× cheaper. **One neuron per cell, one synapse per LUT weight scalar.**

---

## Part 2 — the design

### The neuron that makes it exact

Take an output neuron whose PSPs **grow** exponentially: an input spike at time `s` with
weight `W` contributes `W · exp((t - s)/tau_m)` for `t >= s`; fire at threshold `theta`.
Then

```
 theta = sum_i W_i exp((t_f - s_i)/tau_m) = exp(t_f/tau_m) * sum_i W_i exp(-s_i/tau_m)
 =>  t_f = tau_m * [ log(theta) - log( sum_i W_i exp(-s_i/tau_m) ) ]
```

**The firing time is minus a log-sum-exp.** This is the standard exact-TTFS idealisation
(the exponential-kernel limit behind Mostafa-2017 / Göltz-2021 closed-form spike times).
Be aware this is where the honesty lies: on a *hardware* LIF with a finite synaptic time
constant the closed form picks up a Lambert-W correction and the map stops being exact. v1
is written against the idealised neuron on purpose, so any distillation error is
attributable to learning and not to the neuron model.

Set **`tau_m = tau`** and fire all 32 gated cells synchronously at `t_cell`:

| | synapse | delay | resulting sum |
|---|---|---|---|
| **variant W** | `W_c = exp(w_c/tau)` | 0 | `S` |
| **variant D** | `W_c = 1` | `d_c = D0 - w_c`, `D0 = max w` | `exp(-D0/tau) · S` |

The `D0` offset (needed only to keep delays non-negative) multiplies the sum by a constant,
i.e. shifts every `t_f` by exactly `+D0`. **The two variants are the same layer**, related
by a rigid time shift — `D0 = 0.3084`, delays land in `[0, 0.8453]`.

### Decode — exactly affine, no approximation

`log S = log theta - (t_f - t_cell)/tau`, so

```
 a_o = 32*tau*(log S - log 32) = -32 * (t_f,o - t_cell) + 32*tau*log(theta/32)
```

**Action = −tph × spike time + constant.** Earlier spike ⇒ larger action. Slope is exactly
−32. Then `clip(a, -1, 1)`, which the environment does anyway. Inverting it gives the
training target directly: `t* = (bias - a_teacher)/32`.

### How "log-sum-exp over cells" reads temporally

```
 t_f - t_cell = tau*log(theta) + softmin_tau({d_c}),   softmin_tau(d) := -tau*log sum_c exp(-d_c/tau)
```

The readout is a **soft-min over the 32 arriving synaptic events** — one per table.
`tau -> 0` is literally first-to-spike (winner-take-all); `tau -> inf` is the plain sum.

**Measured on 20k real observations, it sits nowhere near the winner-take-all limit:** the
participation ratio over the 32 tables is **16.97 effective tables** on average (range
3.72–27.04), and the earliest synapse carries only **14.0 %** of the sum on average (at
most 50.6 %). So this must be built as a genuine graded soft-min; a first-to-spike
approximation would discard most of the signal. Worth knowing before committing to
hardware that only implements WTA.

### Network — 2,455 neurons

| layer | count | function |
|---|---|---|
| input | 17 | linear latency code `t_j = c - m·x_j` into `[0, T]` |
| race | 384 | `r_{t,i} = 1[t_a < t_b]` and complement — "which input spikes first" |
| cell | 2,048 | one per LUT row; coincidence detector, threshold 6, fires iff its 6-bit pattern matches |
| output | 6 | exponential-kernel TTFS neuron, one per action dim |

The elegant part: **an anchor-pair bit is already a spike race.** `x[a] > x[b]` under a
monotone-decreasing latency code is exactly `t_a < t_b`. The LUT's addressing needs no
arithmetic in the spiking domain at all — just arrival order.

### The learnable front-end, minimally — and a structural catch

The encoder must apply **one shared map to all 17 dimensions**. The address bit compares
two *different* observation dims, so a per-dimension scale or offset changes which cell is
addressed and destroys the teacher. That leaves `t = c - m·x`, **two scalars**.

**And those two scalars cannot change the output at all.** `m > 0` preserves every
comparison; `c` cancels in every difference `t_b - t_a`. Confirmed numerically in the
smoke test: `d(loss)/dc ≈ 1e-26` (exactly zero up to float noise) and `d(loss)/dm ≈ 1e-12`
at an already-exact fixed point. **v1 has zero learnable content by construction** — it is
a *verification harness*, not a training run, and its value is that it proves the spiking
pipeline reproduces the teacher before any learning is attempted.

So the real minimal learnable front-end is one step up (`LearnableRaceFrontEnd`, in the
file): generalise each race neuron from a fixed index pair to a **learnable linear form on
the 17 latencies**, `d_{t,i} = sum_j A[t,i,j]·t_j` — **3,264 parameters**, initialised to
the exact ±1 anchor pattern so training *starts* at the exact teacher. The 2,048 cell
neurons and all 12,288 output synapses stay frozen. (Once `A` moves, its rows stop summing
to zero and `c` acquires a real gradient — it stops being a gauge freedom. The smoke test
shows that happening at step 1.)

---

## Part 3 — harness status

`python lut_ttfs.py --n 20000` (exactness self-test, both variants):

```
variant D   theta=1601  tau_m=0.090366  D0=0.3084  decode: a = -32.0*t + 21.181708
  spike times t_out   min 0.558268  max 0.777086
  |a_spiking - a_LUT| max 1.364e-06  mean 3.003e-07   (action scale std 1.149)
  delays d = D0 - w   min 0.000000  max 0.845266   (all >= 0: True)

variant W   theta=1601  tau_m=0.090366  D0=0.3084  decode: a = -32.0*t + 11.313779
  spike times t_out   min 0.249895  max 0.468713
  |a_spiking - a_LUT| max 1.364e-06  mean 3.003e-07
  synaptic W = exp(w/tau)  min 0.002628  max 30.3410
```

`1.364e-06` is the dataset's own fp32-storage residual — the spiking layer adds nothing.
Both variants agree to the last digit and their spike times differ by exactly `D0`.

`python train_stub.py --variant D --steps 5` — 5 steps, batch 512, latency MSE `1.4e-16`,
gradients flow, nothing trained. `--learnable-races` switches on the 3,264-parameter v2
front-end and shows a real gradient signal (`race.A` norm ~1e-3).

**No training run was started.**

## Open choices for you

1. **v2 front-end scope** — learnable races (3,264 params, above), or also unfreeze `tau`
   / the cell weights? Unfreezing the weights makes it ordinary distillation rather than
   "fixed readout, learn the front-end".
2. **Neuron realism** — stay on the exact exponential-kernel neuron, or move to a hardware
   LIF and accept an approximate decode? That choice decides whether error means "learning
   failed" or "the neuron model bites".
3. **Which teacher target** — `y_prelog` (the sum, natural in weight-space) or the action
   mean via the affine latency decode (used here). They are equivalent up to the exact
   affine map, so it is purely a question of where you want the loss weighting to sit.
