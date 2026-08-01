# Walker2d int4-LUT → recurrent E/I spiking net (surrogate-gradient distillation)

Prototype that **distills** the Walker2d int4 LUT actor into a from-scratch, biologically-plausible
recurrent spiking net (Dale's law, learnable delays, latency I/O), trained by surrogate-gradient BPTT.
The LUT is a **ground-truth oracle only** — no structural warm-start. This is the *trainable* alternative
to (a) the structural pure-spnet conversion (`walker_pure_spnet.py`, ~0.5% error) and (b) evolution,
which gives no per-weight / per-tick credit assignment.

## Files
| file | role |
|---|---|
| `walker_rsnn_distill.py` | model + training (128 exc / 32 inh). Std-sweep, calibration, prune, eval. |
| `walker_rsnn_ordering.py` | output-ordering eval vs the LUT (exact / pairwise-Kendall / top-1). Caches a checkpoint. |
| `walker2d_lut_actor_int4.npz` | **teacher**: the int4 Walker2d LUT actor (per-table int4 codes + scales). |
| `walker_dataset_stats.json` | obs mean/std for normalization `(obs-mean)/(std+1e-6)`. |
| `figures/walker_rsnn_curve.png` | loss curve + init-std sweep. |
| `figures/walker_rsnn_ordering.png` | ordering-agreement figure. |
| `results/*.txt` | raw metric dumps. |

Paths are resolved relative to the script (`EVO = dirname(__file__)`), so it runs after a fresh clone.
Needs only `numpy` + `torch` (CPU is fine; the prototype was trained on a NUC). No mujoco/env.

## Run
```bash
python walker_rsnn_distill.py      # full: std sweep -> train std=1.0, 300 steps -> prune -> eval
python walker_rsnn_ordering.py     # reproduces the identical net (deterministic), caches ckpt, orders
```

## Architecture
- **I** = 17 inputs, latency-coded: neuron *i* forced to fire at `t_i = c_in − α_in·x_i` (one spike).
- **H_ex** = 128 excitatory, **H_inh** = 32 inhibitory — all-to-all recurrent among the 160 hidden
  (no self-loops); both project to **O** = 6 outputs; O has no outgoing.
- Excitatory outgoing (from I and H_ex): `+softplus(θ)` weights **and learnable per-synapse integer
  delays [1,16]** trained DCLS-style (continuous Gaussian over the delay axis, width annealed, snapped
  to integer at eval).
- Inhibitory outgoing (from H_inh): fixed delay 1, `−softplus(θ)` weights.
- Neurons: single-τ LIF (τ=8), soft reset (subtract θ), arctan surrogate. **T = 32 ticks**, BPTT.
- Output decode: first-spike time → `action = (c_out − t_first)/α_out`; non-firing → `t_first = T`.
- Loss: distill-to-LUT MSE + L1 + delay penalty + firing-rate penalty.

## Key methodology findings (the non-obvious part)
1. **Train on a differentiable expected first-spike time** `E[t] = Σ_t t·g_t·Π_{t'<t}(1−g_{t'}) + T·(never)`
   with `g_t = sigmoid(β(v−θ))`. The **hard** first-spike tick gives **zero** task gradient (it's an
   integer-constant), so the net only obeys the regularizers and self-prunes to a constant. Evaluate on
   the true hard first-spike.
2. **Init weight std must be ≈ 1.0, not 1e-3.** Below that the soft spike-time **collapses to an
   input-independent constant** (the membrane hovers at threshold, `g≈0.5` every tick) — nothing to
   distill. Regime scan: contrast ~0 for std ≤ 0.1, emerges at 0.3, faithful (soft ≈ hard) at 1–3.
3. **Regularizers must sit ~100× below the task gradient** (softplus + recurrent contraction make the
   task gradient ~1e-7/param); an L1 at 2e-4 swamped it and self-pruned the net before it learned.

## Preliminary results (128/32, init std 1.0, 300 Adam steps, held-out 512 obs)
- task MSE **3.55 → 1.41**; distillation error **~24% of action range** (median 23.0%, max 6.56).
- **6.5%** of dims within one output tick (T=32 → 0.33/tick; strict), **1/3072** outputs non-firing.
- firing rates H_ex ~0.25 / H_inh ~0.19 / O ~0.29 per neuron per tick.
- **Ordering vs LUT**: exact full 6-dim argsort **1.2%** (chance 0.14%); pairwise concordant **61.2%**
  ((τ_a+1)/2 = 0.676); Kendall τ_b **+0.377**, Spearman ρ **+0.451**; top-1 argmax **41.8%** (chance
  16.7%), top-1 argmin **27.5%**. → learns the coarse action *ordering* well above chance but not the
  fine permutation.

For contrast, the structural pure-spnet conversion reaches **~0.5%** error — a from-scratch learned net
is a much harder ask.

## Next levers (intended to continue on an H100 / nebius)
- **Longer T** (finer tick resolution) — likely the biggest single win.
- More training steps + larger hidden.
- **Anneal the surrogate toward straight-through** to close the soft-train / hard-eval + snap-delay gap.
- A **dedicated sparsity phase AFTER** fidelity is reached, rather than fighting L1 during training.
