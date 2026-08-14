# A handcrafted spiking network that matches a quantised Walker2d policy

## 1. Where we landed

A **three-stage spiking neural network — 2,889 neurons, 25,953 synapses, ~155 ticks per
inference** — reproduces a quantised Walker2d PPO policy closely enough to be
**statistically indistinguishable from it in closed-loop control**: 6312.2 ± 26.6 against the
software teacher's 6253.4 ± 36.3 over ~520 episodes each (**+58.8 ± 45.0, +1.31σ**), with
98.58% exact agreement on discretised actions. Nothing is trained by gradient descent inside
the network: every weight and delay is derived analytically from the teacher's own lookup
table, and the only fitted quantity is a set of 12,288 readout weights held on an **8-bit
log-domain grid**. The network takes **no external clock** — it is triggered solely by its own
input spikes — and its readout computes a logsumexp *physically*, as the time an exponentially
growing membrane takes to reach threshold.

---

## 2. Training the LUT teacher

### 2.1 The architecture

The policy is an **anchor-pair lookup table**, not a matmul network. For a 17-dimensional
observation `x`:

```
bit_i   = 1[ x[a_i] > x[b_i] ]          6 fixed anchor pairs per table, MSB-first packed
addr_t  = the 6 bits of table t          ->  one of 64 rows
row_t   = weights[t, addr_t]             weights: (32 tables, 64 rows, 6 outputs)
mu[o]   = 32 * tau * log( (1/32) * sum_t exp(row_t[o] / tau) )
action  = clip(mu, -1, 1)
```

**12,288 trainable table entries**, one learned temperature `tau`, and *no arithmetic on the
observation at all* — the policy only ever compares coordinates against each other. That
property is what makes a spiking implementation possible: comparisons are cheap in spike
timing, and the addressing is exact rather than approximate.

The **sum-scaled log-sum-exp** readout (`fast_multi_head_lut.py::_exp_outputs_fwd`) is a
smooth generalisation of a plain sum over tables: `tau → ∞` recovers `sum_t w_t` exactly,
`tau → 0` gives `32·max(w)`. The learned value sits between the two (`tau = 0.09377`).

### 2.2 PPO

Trained with a from-scratch GPU-resident PPO (`experiments/walker2d-lut/src/ppo.py`, torch +
NVIDIA Warp, not JAX) on the batched Warp Walker2d env:

| | |
|---|---|
| arch | `fastlut_lse_sum_expmlpcrit`, 32 tables, 6 anchor pairs |
| envs × rollout × updates | 8192 × 32 × 768 → **201M env-steps** |
| epochs / minibatches | 4 / 8 |
| clip ε / GAE λ / γ | 0.2 / 0.95 / 0.99 |
| lr | 3e-4, cosine → 3e-5 |
| entropy / vf / max-grad | 0.0 / 0.5 / 0.5 |
| target-KL | 0.02, early-stop at 1.5× |
| log-std floor | −1.897, re-clamped every update |
| params | 82,953 |

Critic: a `[17, 256, 256, 1]` Tanh MLP whose final linear readout is replaced by the *same*
sum-scaled logsumexp pooling over its 256 penultimate units.

### 2.3 Quantisation-aware fine-tuning

The trained policy is continuous at both interfaces; a spiking network is not. A **384-update
QAT fine-tune** (full cosine 3e-4 → 3e-5, observation normaliser **frozen** so the fixed
bucket edges stay calibrated) put both quantisers in the training loop:

**Input — 128 Gaussian-companded buckets, σ = 1, ONE map shared across all 17 coordinates.**
`tick = round(127·Φ(x/σ))`. Shared *by necessity*: the LUT addresses by comparisons **between**
coordinates, so a per-coordinate map would change the meaning of every address bit. σ = 1 was
chosen by measurement — it is an interior minimum of the address-bit flip rate (0.486%, against
1.39% at σ=0.5 and 1.16% at σ=3.0), and coincides with the principled value for unit-variance
observations.

**Output — clip to [−1, 1], then snap to 22 uniform levels** (step 2/21 ≈ 0.0952), with a
straight-through estimator.

**The L2 out-of-band penalty — the part that mattered most.** A first QAT run raised return by
+411 but left the *raw, pre-clip* readout exactly as sprawled as before (~51% of components
outside [−1,1]) and made the table weights **wider**, so the spiking delay span **grew**. That
is structural, not accidental: the clip's gradient is exactly zero outside the band, and an
out-of-band action is free in both physics (MuJoCo clamps `ctrl` to `ctrlrange` — verified: +3.3
and +1.0 produce bit-identical trajectories) *and* reward (the training env clamps before
computing control cost). Nothing pulled it in. So:

```
loss += w · mean_batch( sum_o relu(|mu_raw[o]| − 1)^2 )      w = 0.3
```

| | parent | QAT, no penalty | **QAT + L2 (w=0.3)** |
|---|---:|---:|---:|
| raw output outside [−1,1] | 51.6% | 53.9% | **~13%** |
| mean Stage-3 delay span | 74.7 | 79.0 | **63.8** ticks |
| `dmax` | 84 | 96 | **81** ticks |

The weight `w` was chosen by a 20-update sweep, not guessed.

⚠️ **The table weights themselves were never quantised.** They are `float32`, 12,286 distinct
values out of 12,288, on no grid whatsoever (gap ratio 10⁷). "Quantisation-aware" here means
*at the interfaces*.

---

## 3. The spiking network

Constants: `DE=3, DI=2, W_EXC=1.5, W_INH=−10, TAU_M_RAIL=20, TAU_MEM=1200,
W_MEM=W_GATE=0.6, W_AND=0.18, W_DET=0.06, D_GATE=6, D_OUT=1, TAU_M_OUT=31.257`.

Census: **18** input+detector, **272** dual-rail comparators, **272** inhibitory interneurons,
**272** memory cells, **2048** LUT cells, **6** outputs, **1** completion detector = **2,889**.

### 3.1 Stage 1 — order detection

Each observation coordinate is emitted as a **first-spike latency**:

```
tick = 127 − searchsorted(in_quant_edges, x_norm)      larger value → EARLIER spike
```

Only the *edges* cross into the network — it consumes tick **ordering**, never the dequantised
value, so `in_quant_dequant` stays a software-side artefact. Ordering is preserved exactly,
including ties, which is why Stage-1 parity is **bit-exact (100.0000%, 0 bad of 98,304)**
rather than approximate.

136 **dual-rail winner-take-all comparators** resolve the 192 address bits. The rails are
**anti-leaky** (`cf_1 = +1/20`), so `V = 0` is an *unstable* equilibrium: the winner runs away
up and fires, the loser runs away down and stays silent for the whole episode. That
self-reinforcement holds the decision across the ~85-tick spread of arrival times with no
maintenance current.

**Ties and the GT-skew.** Quantisation *creates* ties — there are **zero** exact float ties in
the data, but 0.95% of address slots and **77% of samples** have two coordinates in the same
bucket. Membrane traces on the real kernel showed why they were a problem: **the
cross-inhibition lands one tick after the excitation it must cancel**, so on an exact tie both
rails see pure `+1.5187` and both fire, while 1 tick apart the loser sees `+1.5 − 10 = −8.5`
in a single tick and never rises. The original design handled this with 136 dedicated
tie-detector neurons. Adding **one tick of delay to the GT rail's excitation** (`DE_GT = 4`)
aligns the tie case with the veto and leaves the 1-apart case untouched, reproducing the
software's strict `d > 0` convention exactly — and **removes 136 neurons and 408 synapses for
free**. It is structural, not tuned: the gap between "fires" (+1.519) and "silent" (−8.606) is
10.1 against a threshold of 1.0.

**Self-timing.** A single **leak-free completion detector** (17 inputs at weight 0.06,
threshold 1.0) fires exactly when the *last* input arrives — each input fires exactly once, so
`16w < 1 ≤ 17w` makes this exact. It replaces an external fixed-tick stimulus, and the network
now takes **no external drive at all**. It must then wait out the comparator pipeline
(`D_GATE = 6`), because the last rail resolves at `t_last+3` and its veto at `t_last+5` —
gating earlier was measured at 99.9196% parity with 78 multi-selects.

### 3.2 Stage 2 — lookup

2,048 cell neurons, one per LUT cell. Six coincident memory spikes select exactly one cell per
table: `6 × 0.18 = 1.08 > 1.0`, five give `0.90 < 1.0`. The cells are **leak-free**
(`cf_1 = 0`), so a wrong cell — which receives exactly `6 − k` spikes for Hamming distance
`k` — is bounded by 0.90 **for all time**, monotonically, with no transient to overshoot.
Measured: **0 none, 0 multi of 16,384 table-selections.**

### 3.3 Stage 3 — the amplitude-encoded readout

The original design encoded each cell's stored value as a **delay**. The current one encodes it
as a **synaptic weight**:

```
every selected-cell → output synapse:   delay = 1,  weight = beta_o · exp(w_t / tau)
```

All 32 selected cells land on **one tick**, so the membrane integrates `V0 = Σ_t exp(w_t/tau)`
directly. Because the output neuron is exactly linear (`cf_2 = 0`), the two forms are
algebraically identical — with arrivals `a_t = A − (tau_eff/tau)·w_t`:

```
alpha · e^{t/tau_eff} · Σ_t e^{−a_t/tau_eff}
  = alpha·e^{−A/tau_eff} · e^{t/tau_eff} · Σ_t e^{w_t/tau}
```

so `beta_o = alpha_o · e^{−A/tau_eff}` and the readout is unchanged.

**The anti-leak membrane is the logarithm.** With `V(t) = V0·e^{t/tau_eff}` (exact at integer
ticks — `_calib` inverts the engine's two Euler half-steps precisely), threshold crossing gives

```
T = tau_eff·(log θ − log Σ_t e^{w_t/tau}) = const − (tau_eff / 32·tau) · out
```

The synapses supply the `exp`, the dendrite supplies the sum, and **time-to-threshold under
exponential growth supplies the `log`**. That is the whole logsumexp, computed physically.

What this bought:

| | delay-encoded | amplitude-encoded |
|---|---|---|
| Stage-3 delay span | 91 ticks | **1** |
| `dmax` (engine cap 255) | 91 | **3–6** |
| episode | 309 ticks | **~155** |
| `act()` on CPU | 12.9 ms | **5.4 ms** |
| synapse count | unchanged | unchanged (pure re-weighting) |

It is also **strictly more faithful** — the delay path rounds every delay with `rint`, so it
evaluated a *quantised* logsumexp (worth up to 0.25 tick) — and it removes the dependence on
cross-tick synaptic integration, which this kernel does not have (`I` is zeroed every tick).

**Decode**, in the actor:

```
mu = slope · (T_crossing − t_last) + offset,        slope = −32·tau/tau_eff
silence → −1;  then snap to the 22-level grid
```

Because self-timing makes the origin move per observation, the decode **must** be referenced
to the completion event; with an absolute reference, within-one-level collapsed to 50–70%.

### 3.4 The lattice, and why it constrains everything

`TAU_M_OUT = 31.257` was chosen so 22 ticks span 22 output levels. That makes the **decode
lattice pitch equal the output grid step** — 0.09523729 against 0.09523810. Consequences:

- Every decoded value sits at the **same phase** relative to the output grid, so an offset
  shift moves *no* samples across a boundary or *all* of them. Verified the hard way: an
  on-policy offset recalibration left five of six dims bit-identical and flipped the sixth
  wholesale, and scored **−175** in the walker.
- The crossing tick is `ceil` of a continuous time, and the decode slope is negative, so
  **rounding the tick up rounds the action down** — a systematic, strictly one-sided bias.
  Measured: 100% of errors exactly −1 level, ~21% of actions, mean −0.207 levels.
- The closed-form model must be **phase-pinned** to the network: solving against measured
  ticks gives **phase 0.750, base +13 — the same values on all six dims independently**,
  which is the evidence the model is structurally right.

---

## 4. Closing the gap: the 8-bit weight fit

The −0.207-level bias cost **−164.8 ± 50.3** of closed-loop return. It is not removable by
calibration (see the lattice argument), and not by adding levels (22 are already
representable). It *is* removable by adjusting individual weights, because a per-cell change is
not degenerate with a global offset.

**Method — coordinate descent, no gradients.** Gradient-based attempts failed twice: a
straight-through optimiser diverged (loss rose 29%), and the closed-form model was unfaithful
until phase-pinned. Coordinate descent avoids both. The structure makes it cheap: weight
`(t,k,o)` affects only output dim `o` and only states whose table `t` selected cell `k`, and it
enters through a **sum**, so a candidate move is `S' = S − exp(w_old) + exp(w_new)` over the
affected states only.

- **Objective:** discretised-action level-mismatch against the software teacher's 22-level
  action, scored by the phase-pinned closed form with the tick `ceil` inside the loss.
- **Constraint:** weights on an **8-bit log-domain grid**, active *during* the fit, not applied
  afterwards. (8-bit was shown to cost nothing versus full precision, and is comfortably inside
  the sensitivity bound: a relative error δ shifts the crossing by `tau_eff·δ`, so half an
  output level needs δ < 1.6%.)
- **Data:** 153,600 teacher input→output pairs from 512 independent rollouts, split by seed —
  115,200 train (seeds 0–2) / 38,400 held-out (seed 3).
- **Gates, both mandatory before any move:** the scorer must reproduce the true-SNN baseline
  (exact > 70% on every dim, uniformly negative bias), and a zero-step run must reproduce the
  baseline exactly.

Loss fell 151,004 → 17,100 in three sweeps (6 seconds). **5,390 of 12,288 weights moved**, max
log-domain drift 0.261.

**Validated on the true SNN** (GPU, measured spike ticks, held-out states):

| | overall exact | mean signed (levels) |
|---|---:|---:|
| baseline | 80.83% | −0.158 … −0.260, all negative |
| **optimised** | **98.58%** | **−0.004 … +0.002** |

The one-sided −1 mass collapses from 15–26% to 1–2% on **every** dim, with matching +1 mass —
symmetric, ~15× smaller, no aggregate cancellation.

### ⚠️ The offset must travel with the weights

The fit **re-derives the decode offset** (the weights change → the crossing ticks change), by
**−0.23 to −0.27 level units**. An earlier walker eval used the optimised weights with the
*shipped* offset — a quarter-level mis-decode, the same magnitude as the bias just removed —
and measured **−197.0**, which produced a confident but wrong conclusion that return was
decoupled from action agreement. With the matched offset the same weights score **+58.8**.
The two files are a **pair** (`stage3_weights_bigdata.npy` + `stage3_offset_bigdata.npy`) and
must never be used apart; the eval harness now refuses to run weights without an offset.

---

## 5. Results

**Parity gate**, 512 held-out samples:

| stage | result |
|---|---|
| Stage 1 address bits | **100.0000%** — 0 bad of 98,304, ties included |
| Stage 2 one-hot | **0 none, 0 multi** of 16,384 |
| Stage 3 within one level | **100.000%** on every dim |
| Stage 3 exact (true SNN) | **98.58%** |

**Closed-loop walker**, 256 envs × 1200 steps × 2 seeds, matched physics (solver 100 / ls 50),
deterministic:

| build | n | mean | sd | se | median | ep length | gap vs software |
|---|---:|---:|---:|---:|---:|---:|---:|
| software teacher | 521 | 6253.4 | 828.1 | 36.3 | 6407.9 | 979 | — |
| GT-skew, unoptimised | 523 | 6119.4 | 935.4 | 40.9 | 6330.5 | 969 | −164.8 ± 50.3 |
| **8-bit weight-fitted** | 515 | **6312.2** | **604.7** | 26.6 | **6415.9** | **986** | **+58.8 ± 45.0 (+1.31σ)** |

**Indistinguishable from the teacher**, and +192.8 ± 48.8 (+3.95σ) over the unoptimised build.
Two secondary signals agree: the spiking build's **per-episode variance is now lower than the
teacher's** (605 vs 828) and its **episodes last longer** (986 vs 979) — the tail failures that
characterised every earlier spiking build are gone.

**Cost:** 2,889 neurons, 25,953 synapses, episode 138 / **154.5** / 167 ticks (data-dependent),
5.4 ms per `act()` on CPU.

⚠️ **+58.8 is not evidence the network is *better* than the teacher.** At 1.31σ the honest
claim is "indistinguishable", and the number should not be quoted as a gain.

---

## 6. Notes for reproduction

**Files** (all forked; the delay-based `tiny_lut_full_pipeline.py` and the deployed
`spiking_lut.py` are untouched):

Paths below are repo-relative; the scripts themselves hardcode them under
`/home/astarostin/projects/spiky/`, so a clone elsewhere needs the roots adjusted.

| file | role |
|---|---|
| `experiments/walker2d-lut/walker2d-spiking/tiny_lut_quantised_pipeline.py` | builds and verifies the network |
| `experiments/walker2d-lut/walker2d-spiking/tiny_lut_quantised_export.py` | exports the actor artefact |
| `landing/walker2d-viz/server/actors/spiking_lut_quantised.py` | the served actor |
| `experiments/walker2d-lut/walker2d-spiking/collect_teacher_io.py` | the 153K teacher dataset |
| `experiments/walker2d-lut/walker2d-spiking/stage3_cd_bigdata.py` | the 8-bit coordinate-descent fit |
| `experiments/walker2d-lut/walker2d-spiking/eval_gtskew_large.py` | the paired walker eval |

The served actor is **not** pure numpy. Numpy does the input companding and the output decode,
and there is no scipy anywhere — but the network itself is built and run on the spnet engine:
`__init__` imports torch and `spiky.spnet`, grows the 2,889 neurons through
`SynapseGrowthEngine`, and `act()` calls `process_ticks`. What runs in the demo stand is the
real simulator, not a numpy replay of it.

**The observation set is carried.** `data/distill_exp19_100k.npz` (21.6 MB, 100,000 real
Walker2d states) is committed here, and every script in this directory resolves it relative to
itself — so the pipeline and its twelve siblings run with no `--data` argument. That file is what
every verification number in this document is measured on: the pipeline takes the held-out tail
of `--n` states from it.

**Inputs this branch does not carry.** Two stage outputs, both regenerable, live on
`research/walker2d-lut`:

| path | what produces it |
|---|---|
| `analysis/software_teacher_io_dataset_100k.npz` | `collect_teacher_io.py --out` |
| `deploy_quantised/spiking_lut_quantised_actor.npz` | `tiny_lut_quantised_export.py`; a byte-identical copy of the shipped build is on this branch at `landing/walker2d-viz/server/models/` |

`stage3_cd_bigdata.py` creates `deploy_quantised/` itself and writes the fitted
`stage3_weights_bigdata.npy` + `stage3_offset_bigdata.npy` pair there.

**Traps worth knowing**, each of which cost a wrong result before being caught:

1. **Censored evaluation.** Any rollout shorter than the 1000-step episode limit measures only
   the episodes that *failed*. A 600-step screen returned n=1 and a meaningless "mean".
2. **Exact-match is the wrong calibration objective.** It selects the biased-low decode; the
   objective that matters is a zero mean *signed* residual.
3. **Underpowered evals.** At n=65 the se is ~110; real effects of ~165 sit inside noise.
   ~520 episodes gives se ~30–40.
4. **Weights and offset are a pair.** See §4.
5. **Open-loop data.** All fitting uses teacher-visited states. It transferred here at 98.58%
   agreement; it did not need to.

**Shipped:** the fitted weights **are** baked into `spiking_lut_quantised_actor.npz`, together
with the offset they were fitted against — §4's rule that the two travel as a pair is honoured in
the artefact itself. The file carries `stage3_fit = "coord-descent 8-bit log-domain, 153k teacher
pairs"`; its `weights` differ from the teacher table by up to 2.56e-02 and match
`stage3_weights_bigdata.npy` to 1.5e-08, and its per-dim `affine` offsets are
`stage3_offset_bigdata.npy` exactly. The pre-fit GT-skew build — the one this paragraph used to
describe — is kept on `research/walker2d-lut` as
`spiking_lut_quantised_actor_GTSKEW_verified.npz.bak`, whose weights still match the teacher
table exactly.

The actor is deployed: it ships in `landing/walker2d-viz/server/models/` and the demo stand
serves it as **"Spiking LUT quantised (handcrafted SNN)"**.
