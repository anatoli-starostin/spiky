# A real, fully-simulated SNN approximating the exp19 LUT teacher

**Best held-out accuracy: 0.381 normalised** (mean |a_SNN − a_LUT| / action std 1.1309;
max 5.04) on a 4,000-sample validation split, from a genuinely simulated network — no
functional shortcuts anywhere. For reference the analytic construction reaches **1.1e-05**.
So the real circuit is roughly **4 × 10⁴ times less accurate**, and this document is mostly
about where that goes.

12 configurations, ~20 min total. Nothing committed.
Files: `real_snn.py`, `train_real_snn.py`, `probe_race.py`, `probe_out.py`,
`results/real_snn.json`.

---

## The architecture, as actually simulated

Discrete-time current-based LIF throughout (the standard Zenke form), spikes as events,
one spike per neuron per sample (TTFS):

```
I[n+1] = alpha_s * I[n] + sum_j W_j * S_j[n]
V[n+1] = alpha_m * V[n] + I[n]
S[n]   = H(V[n] - theta)          latched off after the first spike
```

| layer | count | what it is |
|---|---|---|
| input | 17 | one spike each at `t_j = c − m·x_j`, c = 0.5404, m = 0.06211, window T = 1 |
| **race** | **384** | 2 per (table, anchor-pair): `r+` fires if a arrives first, `r−` if b does. **Two real synapses each** — one excitatory from its winner input, one inhibitory veto from the other — with real weights and real, learnable, continuously-interpolated delays |
| **cell** | **2048** | one per LUT row; 6 real synapses from the race neurons matching its address, threshold 5.45 so **all six** are required; long `tau_m` to latch, plus the validated square-wave inhibitory gate opening at 1.05 T |
| output | 6 | TTFS LIF, `tau_s = tau_m/2`, one synapse from every cell (12,288 total), weight-coded `exp(w/tau)`, `tau_m = k·tau` |

12,288 output synapses + 2,304 race synapses. Initialised from the analytic construction:
output weights `exp(w/tau)`, anchor wiring, `tau_m` tied to the teacher's `tau = 0.09037`,
decode fitted at init. Trained end-to-end by BPTT with a fast-sigmoid surrogate
(`dS/dV = 1/(1+25|V−θ|)²`), target = action mean, through the output neuron.

### Two things had to be fixed before it worked at all

**1. The veto must be persistent, not a PSP.** With a single fast synaptic current the
inhibition is transient: if the winner spikes much earlier than the loser, its veto has
decayed by the time the loser's excitation arrives, and the *wrong-order* neuron fires
anyway. That failure alone capped order accuracy at **0.83** with errors at gaps up to
0.19 T. Giving the inhibitory synapse its own long time constant (`tau_s_veto = 10 T`)
removed it: errors then sit at gaps below 0.018 T. This is a real design constraint, not a
tuning detail — a spiking order detector needs *latching* feedforward inhibition.

**2. The output threshold must be set below the weakest sample's peak.** Calibrating on the
batch maximum leaves ~31 % of output neurons silent, and a silent TTFS neuron is pinned to
the end of the window — a large one-sided error. Threshold is now
`0.95 × quantile_{0.005}(per-sample peak V)`; firing rate is then 0.99–1.00.

## Where the error comes from

Measured by decomposition (`probe_out.py`), before any training, at dt = 1/128:

| | normalised error |
|---|---|
| **ORACLE** — feed the output layer the *correct* 32 cell spikes, all simultaneous | **0.10 – 0.13** |
| **FULL** — the real simulated network | **0.30 – 0.51** |

So roughly a quarter of the error is the output neuron + decode + time quantisation, and
three quarters is the price of simulating the race and cell layers as real neurons.

### The race layer's temporal dead zone is the dominant limit

A spiking order detector cannot resolve two spikes closer together than the time its
membrane needs to reach threshold. Measured (`probe_race.py`, 512 samples, 192 pairs each):

| dt | order accuracy `r+` | dead zone (p95 of \|Δt\| among errors) | cells with all 6 races right |
|---|---|---|---|
| 1/64 | 0.845 | 0.016 | 0.22 |
| 1/128 | 0.898 | 0.005 | 0.42 |
| 1/256 | 0.909 | 0.003 | 0.54 |

**The dead zone is ≈ 1 dt** — i.e. the detector is already at the time-quantisation limit
and the design cannot be improved further at fixed dt. The problem is the *input* side: the
encoder packs 17 dimensions into [0, 1] with slope m = 0.0621, so a typical anchor pair is
separated by only ~0.09 T and a substantial tail falls below any achievable dead zone. A
cell needs all six of its races right, so a per-race error rate of ε costs `(1−ε)⁶` at the
cell — 10 % per-race error leaves only ~54 % of cells correct.

That propagates straight through: **cells firing per sample** (of 32) is 10.7 at dt = 1/64,
23.4 at 1/128 and 24.0 at 1/256. Every missing cell is a dropped synapse on the output
neuron.

## The full sweep

Held-out val (4,000 samples), 20,000 training pairs, 400 BPTT steps, batch 64, Adam 3e-3.
"base" = at initialisation from the analytic construction; "best" = best-val checkpoint.

| config | trainable | sim steps | base | **best** | max | cells/32 | pairs to best | wall |
|---|---|---|---|---|---|---|---|---|
| BASE dt 1/128, mlp decode | 1,848 | 330 | 0.4928 | **0.4744** | 3.17 | 23.4 | 3,200 | 80 s |
| frozen front end (decode only) | 312 | 330 | 0.4928 | 0.4917 | 5.25 | 19.1 | 12,800 | 33 s |
| decode affine | 1,848 | 330 | 0.7065 | 0.6954 | 3.65 | 30.7 | 3,200 | 80 s |
| decode corrected (analytic) | 1,848 | 330 | 0.5654 | 0.5654 | 4.69 | 19.1 | 0 | 80 s |
| dt 1/64 (165 sim steps) | 1,848 | 165 | 0.6698 | 0.5947 | 4.24 | 10.7 | 12,800 | 40 s |
| **dt 1/256 (860 sim steps)** | 1,848 | 860 | 0.3807 | **0.3807** | 5.04 | 24.0 | 0 | 276 s |
| k_out 8 | 1,848 | 430 | 0.4786 | 0.4474 | 3.35 | 24.5 | 6,400 | 106 s |
| veto 20 | 1,848 | 330 | 0.5077 | 0.4697 | 4.28 | 23.6 | 3,200 | 81 s |
| cell tau_m 50 | 1,848 | 330 | 0.4928 | 0.4928 | 5.08 | 19.1 | 0 | 81 s |
| + train output weights | 14,136 | 330 | 0.4928 | 0.4749 | 3.50 | 23.7 | 3,200 | 83 s |
| + train out & cell | 14,137 | 330 | 0.4928 | 0.4928 | 5.08 | 19.1 | 0 | 85 s |
| dt 1/256 + train out | 14,136 | 860 | 0.3807 | 0.3807 | 5.04 | 24.0 | 0 | 284 s |

### What matters, in order

1. **dt / simulation length.** 0.595 → 0.474 → 0.381 for 165 → 330 → 860 steps. This is the
   only knob that moves the result substantially, and it is a pure compute trade: **error
   falls roughly as a power ~0.35 of the step count**, so halving the error costs ~7× the
   compute. At this rate reaching even 0.05 would need ~10⁵ simulation steps per sample.
2. **The decode.** affine 0.695 → analytic-corrected 0.565 → small learnable MLP on the
   spike time 0.474 (294 params, sees nothing but `t_out`). Interesting inversion: the
   analytic corrected decode, which was *exact* in continuous time, is beaten here by a
   fitted one — the discrete LIF with a spread of non-simultaneous, partly-missing inputs
   is simply not the continuous neuron the formula inverts.
3. **`k_out`** (output membrane time constant, i.e. the timing dynamic range) gives a small
   real gain: 0.474 → 0.447.
4. **Veto strength** 6 → 20: 0.474 → 0.470. Marginal.
5. **Cell `tau_m` 15 → 50:** no change. The gate already opens at 1.05 T and 15 T of
   membrane is plenty of latch.

### Training barely helps, and that is the headline negative result

The best configuration's best checkpoint is its **initialisation** — 0 pairs seen. Across
the sweep, surrogate-gradient BPTT improves the analytic init by at most **11 %**
(0.6698 → 0.5947) and usually by 0–4 %, and unfreezing the 12,288 output weights adds
nothing (0.4744 vs 0.4749). Train and val agree to three decimals (0.3797 vs 0.3807), so
this is neither overfitting nor a capacity limit — it is a **systematic, structural** error
that gradients cannot reach.

This is consistent with what the earlier analytic sweep found (`RESULTS.md`): the surrogate
gradient does not vanish at good solutions and cannot find its way back into their basin.
Here the same effect shows up as "the analytic initialisation is already the best point
training will find".

Sample efficiency is therefore not really measurable: the best-val curve for the strongest
config is flat at 0.3807 from 1,000 pairs to 20,000. For the best *trainable* config
(k_out 8) it is 0.4559 at 1k → 0.4474 at 5k → flat thereafter. **The 100K dataset is not
the constraint; the circuit is.**

## Accuracy / compute trade-off

| sim steps/sample | neurons | synapses | val error | wall for 400 BPTT steps |
|---|---|---|---|---|
| 165 | 2,455 | 14,592 | 0.595 | 40 s |
| 330 | 2,455 | 14,592 | 0.474 | 80 s |
| 860 | 2,455 | 14,592 | 0.381 | 276 s |

Neuron and synapse counts are fixed by the LUT's structure; only time resolution buys
accuracy, and it buys it slowly.

## Flags for validation

1. **The 0.381 is honest but the architecture is not fully optimised.** I fixed two real
   bugs (transient veto, threshold calibration) mid-run and did not re-tune the race time
   constants afterwards. A dedicated race-layer search could plausibly gain something; I do
   not think it changes the order of magnitude, because the dead zone is at the dt limit.
2. **The MLP decode (294 params, per-output, sees only `t_out`) is a judgement call.** It is
   still "decode the action from output spike timing", but it is more than an affine read.
   The affine number (0.695) is in the table if you want the stricter reading.
3. **`theta_race = 1` with weights normalised so one input spike peaks at 2θ** is arbitrary;
   only the ratio to the veto matters, and I swept only two veto values.
4. **Encoder constants are frozen at the analytic values** (c = 0.5404, m = 0.06211). Since
   the dead zone is absolute and the anchor gaps scale with m, **stretching the input window
   would directly reduce the relative dead zone** — at proportionally more simulation steps.
   I did not sweep it because it is the same trade as dt; worth confirming you agree.
5. **No refractory/reset dynamics** — every neuron fires at most once and is then latched
   off. That is the TTFS assumption and it is what makes "spike time" well defined, but it
   is not a general-purpose spiking model.
6. **Single seed per configuration.** Given training moves the result so little, seed
   variance is probably small, but I have not measured it.

## Honest summary

A fully-simulated spiking network built from the LUT's own structure reaches **0.381
normalised error** — about a third of an action standard deviation. That is a working
approximation, not a faithful copy: the analytic construction is 4 × 10⁴ times better, and
the entire difference is the cost of making the race and cell layers real. The binding
constraint is that **a spiking order detector cannot resolve spike pairs closer than one
simulation timestep**, and the anchor-pair gaps of this encoder are frequently that close.
Surrogate-gradient training does not close the gap; the analytic initialisation is
essentially the best point available.

## Reproduce

```sh
cd experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/distill/spiking
python probe_race.py                       # race-layer dead zone
python probe_out.py                        # oracle vs full decomposition
python train_real_snn.py --sweep --steps 400 --eval-every 50 \
       --n-train 20000 --n-val 4000 --tag real_snn
```
