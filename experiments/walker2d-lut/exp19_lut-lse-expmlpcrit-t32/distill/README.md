# exp19 distillation dataset — 100,000 on-distribution (obs → pre-log) pairs

A teacher dataset for training a spiky (LIF) student on exp19's LUT actor. The primary
target is the **summed exponentials before the readout's final log**, so a student can
learn the positive, additive quantity and have the log applied afterwards.

| file | what |
|---|---|
| `distill_exp19_100k.npz` | the dataset, 21.6 MB, 15 arrays (below) |
| `meta.json` | machine-readable provenance + every stat printed below |
| `collect_distill.py` | the generator (rollout + capture) |
| `verify_dataset.py` | numpy-only re-derivation of the targets from the `.npz` alone |

## The target — exactly what `y_prelog` is

exp19's actor is `FastMultiHeadLut(exp_outputs=True, exp_outputs_scale="sum")`. The
readout is `src/spiky/lutorch/fast_multi_head_lut.py::_exp_outputs_fwd`, lines **165–188**:

```python
d        = x[:, anchor_a] - x[:, anchor_b]                       # [B, 32, 6]
index    = ((d > 0) * powers).sum(-1)                            # [B, 32]  MSB-first
w_sel    = weights.view(32*64, 6)[index + table_offset]          # [B, 1, 32, 6]
z        = clamp(w_sel / tau, -60, +60)                          # line 177
lse      = logsumexp(z, dim=2)                                   # line 178
out      = tph * tau * (lse - log(tph))                          # line 187, scale=="sum"
```

`y_prelog` is the quantity inside that outer log:

```
S[b, o] = Σ_{t=0..31} exp( clamp( w_sel[b, 0, t, o] / tau, -60, +60 ) )   == exp(lse)
```

— the per-output sum of exponentials over the **32 tables**, before the log and before the
`tph·tau` scaling. Strictly positive by construction. The action mean is recovered exactly:

```
y_action_mean = tph * tau * ( log(y_prelog) - log(tph) )
              = tph * tau * log( y_prelog / tph ),      tph = 32, tau = 0.09036568
```

Two notes on faithfulness:

* **The clamp is inside the definition.** The dataset records what the network actually
  computes. It never binds on this data — the largest `|w/tau|` seen was **5.941** against
  a limit of 60 — so `S = Σ exp(w_t/tau)` unclamped is the same thing here.
* **`n_heads == 1`** for the actor, so "sum over the tables within a head" and "sum over
  all 32 tables" are the same reduction. There is no head ambiguity to resolve.

## Arrays in the `.npz`

| key | shape | dtype | what |
|---|---|---|---|
| `x` | (100000, 17) | float32 | **RAW observation**, as the env emits it — *before* normalisation |
| `x_norm` | (100000, 17) | float32 | the **normalised** observation the LUT actually indexes with |
| `y_prelog` | (100000, 6) | float32 | **PRIMARY target** `S` — the pre-final-log sum of exponentials |
| `y_action_mean` | (100000, 6) | float32 | secondary target — the full readout output |
| `y_prelog_f64`, `y_action_mean_f64` | (100000, 6) | float64 | the same two, full precision |
| `tau` | () | float64 | `0.09036568` — the actor's learned `exp_outputs_tau` |
| `tables_per_head` | () | int64 | `32` |
| `exp_clamp` | () | float64 | `60.0` |
| `obs_mean`, `obs_var` | (17,) | float32 | the training-time running normalisation stats |
| `obs_count` | () | float64 | `2.013e8` samples behind those stats |
| `anchor_a`, `anchor_b` | (32, 6) | int64 | the fixed anchor pairs |
| `weights` | (32, 64, 6) | float32 | the teacher's LUT tables |

`x_norm = (x - obs_mean) / sqrt(obs_var + 1e-8)` — that is `RunningNorm.norm` verbatim
(`src/ppo.py:36`). Both are stored so either convention can be used without guessing;
**the teacher's input is `x_norm`**, and `x` is what a deployment would see first.

The anchors and weights are included so the teacher can be re-evaluated on new inputs
without loading torch — `verify_dataset.py` does exactly that, and it is how the
consistency checks below were produced.

## How the inputs were collected

On-policy rollouts of the trained actor in **the same GPU-batched MuJoCo-Warp env used for
training** (`src/warp_env.py::WarpWalker2dVecEnv`), not random noise.

| | |
|---|---|
| policy | `rerun_ckpt/actor_s1.pt` — arch `fastlut_lse_sum_expmlpcrit`, tph 32, seed 1 |
| its training return | 5373.9 |
| env | 250 parallel envs × 400 steps = **100,000 pairs**, 1.9 s wall |
| env settings | `solver_iters=10`, `ls_iters=8`, `obs_clip_vel=None`, `reset_noise=5e-3`, `max_steps=1000`, seed 0 |
| normalisation | the checkpoint's **frozen** final `obs_mean`/`obs_var` (not re-estimated) |
| drive signal | deterministic action mean **+ Gaussian dither, std 0.0619** |
| targets | always the **deterministic** readout at the visited state |

**Why dither, and why this much.** Reset noise is only `5e-3`, so a purely deterministic
policy walks all 250 parallel envs down a near-identical trajectory and "100,000 samples"
would collapse to a few hundred distinct states. The dither is **0.3× the policy's own
trained exploration std** (0.2062), so the visited states stay well inside the distribution
training actually explored — training itself drove the sim with the *full* 0.2062. The
dither changes *which* states get visited; it never touches the label, which is always the
deterministic readout evaluated at the state that was actually reached.

**Which checkpoint, and why not the deploy `.npz`.** `rerun_ckpt/` is exp19's config run
verbatim with `--save-model` added (`rerun_for_checkpoints.sh`); the original exp19 run
saved no weights. It is therefore the faithful exp19 actor. The deploy artifact
`deploy/walker2d_fastlut_lse_exp19.npz` is deliberately **not** used: it is the separate
`--obs-clip-vel 10.0` retrain, whose observation convention differs from exp19's.

### The policy was walking, not falling

| | |
|---|---|
| reward per recorded step | **5.142** (exp19's training arm averages ~5.4/step; a fallen env earns ~0) |
| steps recorded in a healthy state | **99.99 %** |
| episodes that finished during collection | 7, mean return 1280.4, mean length 271.3 |
| envs still running at the end | 250, mean return-so-far 2020.8 over 392.4 steps |
| **distinct observation vectors** | **100,000 / 100,000** |
| distinct output vectors | 64,571 / 100,000 — expected: the LUT is piecewise constant, so nearby states share an address pattern |

## Sanity checks

No NaN, no inf, in any array.

| array | min | max | mean | std |
|---|---:|---:|---:|---:|
| `x` (raw obs) | −74.9574 | 72.4183 | 0.2949 | 6.0800 |
| `x_norm` | −7.3277 | 8.6494 | −0.0322 | 1.1255 |
| `y_prelog` | **8.9476** | 100.7718 | 43.6313 | 15.0492 |
| `y_action_mean` | −3.6850 | 3.3171 | 0.6957 | 1.1309 |

**`y_prelog` is strictly positive** — minimum 8.9476, three orders of magnitude clear of 0,
so `log` is safe everywhere and a student can be trained in either `S` or `log S` space.

Per-output-dimension:

| dim | `y_prelog` min / max / mean / std | `y_action_mean` min / max / mean / std |
|---|---|---|
| 0 | 11.68 / 100.77 / 46.55 / 12.56 | −2.91 / 3.32 / 0.97 / 0.83 |
| 1 | 10.08 / 88.24 / 43.63 / 15.14 | −3.34 / 2.93 / 0.69 / 1.15 |
| 2 | 8.95 / 63.67 / 30.79 / 10.85 | −3.69 / 1.99 / −0.28 / 0.97 |
| 3 | 11.55 / 97.77 / 53.49 / 14.31 | −2.95 / 3.23 / 1.36 / 0.88 |
| 4 | 12.50 / 98.98 / 47.54 / 12.06 | −2.72 / 3.27 / 1.04 / 0.82 |
| 5 | 9.00 / 80.43 / 39.79 / 14.26 | −3.67 / 2.67 / 0.39 / 1.26 |

### Pre-log reproduction error — does `y_prelog` really sit before the log?

Applying the outer readout to `y_prelog` and comparing against the module's own
`logsumexp` output (`y_action_mean`):

```
| tph*tau*(log(y_prelog) - log(tph))  -  y_action_mean |     max 1.521e-06     mean 2.983e-07
```

against an action-mean scale of ~1.13 (std) — a **relative error of 1.3e-6**. The residual
is pure fp32-vs-fp64 rounding: `y_prelog` is an `exp().sum()` in float64 over float32
weights, `y_action_mean` is torch's fused float32 `logsumexp`, and they are not required to
round identically. There is no systematic component.

`verify_dataset.py` re-derives everything from the `.npz` alone, in numpy, with no torch
and no access to the training code:

```
1. x_norm re-derived from x + obs stats     max|diff| 5.743e-07
2. y_prelog re-derived from weights          max|diff| 3.544e-06   max rel 5.541e-08
3. y_action_mean = tph*tau*log(y_prelog/tph) max|diff| 1.523e-06
4. ... from the float32 y_prelog             max|diff| 1.552e-06
PASS — the .npz is self-contained and internally consistent.
```

Line 4 is the number that matters for a student trained on the float32 `y_prelog`: storing
the target in float32 costs **1.6e-6** of action-mean accuracy, i.e. nothing.

## One thing to know before training a student

**`y_action_mean` is the *unclipped* readout, and 60.6 % of its components lie outside
[−1, 1].** The env applies `action.clamp(-1, 1)` (`warp_env.py:105`) *after* the policy, so
the actuator only ever sees the clipped value. The dataset stores the raw readout on
purpose — it is the actual function the LUT computes, and clipping is a property of the
environment, not of the teacher. A student is free to be graded on `clip(ŷ, −1, 1)`, which
is a strictly easier target, but it should be *trained* against the raw value or it will be
fitting a function with large flat regions.

## Reproduce

```sh
cd experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/distill
python collect_distill.py --envs 250 --steps 400      # ~2 s on the RTX 5090
python verify_dataset.py
```

Both seeds are fixed (torch `--seed 0`, env seed 0), but CUDA reductions in the physics
step are not bitwise deterministic, so a re-run reproduces the statistics rather than the
exact bytes.
