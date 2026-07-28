# Walker2d-v5 baselines — results (issue #75, roadmap step 1)

Both baselines are **solved** by the issue's definition (mean return ≥ 3000 over 100
deterministic episodes). Everything below was measured on gpustar (RTX 5090).

## Headline

| baseline | env-steps | wall-clock | 100-ep deterministic eval | solved |
|---|---:|---:|---|:--:|
| **SAC** (SB3, CPU-env, the pinned reference) | 1M | 1.76 h | **5273.4 ± 33.9** | ✅ |
| **MJX/PPO** (GPU-parallel, solver 10/8) | 200M | **0.46 h** | **5555.5 ± 34.4** *(in the CPU reference env)* | ✅ |

Read this as a **wall-clock and throughput** result, not a sample-efficiency one: PPO
used 200× the environment steps. What it buys is that a full Walker2d run costs **27
minutes** instead of 1.8 hours — which is the property the CMA-ES LUT arm (roadmap
step 3) actually needs, since that arm's cost is dominated by rollouts.

## SAC baseline (`exp_c01_sac_baseline`)

Issue spec, verbatim: MLP [256,256] ReLU, lr 3e-4, buffer 1e6, batch 256, γ 0.99,
τ 0.005, train_freq 1, gradient_steps 1, learning_starts 10k, auto-entropy
(target −6), 1M steps, seed 0.

* **Final: 5273.4 ± 33.9** over 100 deterministic episodes (`run_seed0/summary.json`).
* Sample-efficiency milestones (the issue's checkpoints), 10-episode eval:

  | steps | return |
  |---:|---:|
  | 100k | 520.4 |
  | 300k | 3898.8 |
  | 1M | 5274.6 |

* Wall-clock 1.76 h at 157.6 fps. Full curve in `sac_curve.csv`.
* Seeds 1 and 2 were started and then **stopped at 125k steps** once the GPU-parallel
  track became the main path; seed 0 alone is the pinned reference. Their partial logs
  remain in the run dirs.

**Throughput note worth keeping.** SAC's wall-clock here is *not* simulator-bound.
Profiled per env-step: `env.step` 0.067 ms (1.2%), actor inference 0.262 ms (4.8%),
**SAC gradient step 5.105 ms (93.9%)**. And the update is launch-latency-bound, not
compute-bound — batch 256 → 4096 costs only 3% more time (16× the arithmetic for
nothing). Two consequences: vectorising the environment would buy ~3%, and the default
CPU run collapses to 20 fps from torch thread-pool thrash unless threads are pinned
(pinned: 184 fps CPU, 233 fps CUDA — a 9× swing from one env var).

## MJX/PPO baseline (`exp_c02_mjx_scaffold`)

MuJoCo MJX (JAX) stepping thousands of environments in one batched GPU kernel, with a
compact purejaxrl-style PPO. 4096 envs × rollout 32 = 131,072 env-steps/iteration,
1526 iterations = **200,015,872 env-steps in 27.4 min** (~121,686 env-steps/s
end-to-end, ~130k steady-state).

* **CPU-reference eval: 5555.5 ± 34.4** over 100 deterministic episodes — evaluated in
  gymnasium's CPU `Walker2d-v5`, *not* in MJX, so it is directly comparable to SAC.
* Curve in `ppo_curve.csv` (downsampled 1/25).

### Batched-simulation throughput

Env-steps/s (**not** physics steps — Walker2d-v5 has `frame_skip=4`, so one env-step is
four `mjx.step` calls) against the measured **14,835 env-steps/s** single-env CPU
baseline:

| solver | 4,096 envs | 16,384 envs | 32,768 envs |
|---|---:|---:|---:|
| 100/50 (stock XML) | 79,569 (5.4×) | 183,733 (12.4×) | 175,497 (11.8×) |
| **10/8 (adopted)** | 158,802 (10.7×) | **340,190 (22.9×)** | 317,702 (21.4×) |
| 4/4 | 178,350 (12.0×) | 388,124 (26.2×) | 390,822 (26.3×) |

GPU memory stays trivial — 4.9 GB of 32 GB at 32,768 envs. Fusing the four frame-skip
physics steps into a single jitted `lax.scan`, rather than four dispatches, was worth
**3×** on its own.

### The 10/8 solver decision, and its cross-check

`walker2d_v5.xml` ships `solver iterations=100 / ls_iterations=50`. Reducing them is
standard MJX practice for GPU throughput but **changes the dynamics**, so it was
measured rather than assumed (`cross_check.py`).

**Open-loop** — identical initial state and action sequence, 200 env-steps, mean |Δqpos|:

| comparison | mean | @50 | @200 |
|---|---:|---:|---:|
| MJX @100/50 vs CPU | 0.3399 | 0.0469 | 1.4921 |
| MJX @10/8 vs CPU | 0.3409 | 0.0468 | 1.5888 |
| MJX @10/8 vs MJX @100/50 | 0.0319 | 0.0004 | 0.1261 |

The reduced solver is **not** the source of the disagreement. MJX@100/50 already
diverges from CPU MuJoCo by 0.3399 — that is the MJX-vs-MuJoCo-C engine gap, present
at any solver setting. Going to 10/8 moves it to 0.3409, i.e. **+0.3%**, while 10/8 vs
100/50 *within* MJX differs an order of magnitude less than either differs from CPU.
The comparability cost was paid by choosing MJX at all; 10/8 adds essentially nothing.

**Closed-loop** — the same policy evaluated in three engines:

| engine | return |
|---|---|
| MJX @10/8 (trained in) | 1160.7 ± 55.4 |
| MJX @100/50 (stock) | 1157.9 ± 50.5 |
| **CPU Walker2d-v5 (reference)** | **1178.3 ± 51.5** |

**Transfer gap 10/8 → reference: +1.5%**, inside the ±51 spread and in the *favourable*
direction. Confirmed again at full scale: the 200M-step policy scored ~5272 by the
in-training proxy under MJX@10/8 and **5555.5 in the CPU reference**.

## Two measurement traps recorded

1. **The in-training proxy overestimates.** `mean reward/step × 1000` assumes every
   episode survives 1000 steps. For a weak policy it is badly optimistic (proxy 2678 vs
   deterministic 1161); once the policy stops falling the two converge (5272 vs 5555).
   Quote the deterministic eval, not the proxy.
2. **Physics steps ≠ env-steps.** `frame_skip=4`, so a raw `mjx.step` count overstates
   throughput 4× against the CPU baseline. All numbers here are env-steps.

## Artifacts not in git

Excluded by `.gitignore` (see it for the list) and regenerable:

| artifact | where | regenerate |
|---|---|---|
| SAC checkpoints (21 × 3.2 MB) | `exp_c01_sac_baseline/run_seed0/ckpt/` | `launch.sh 0 1000000` (1.8 h) |
| PPO policy weights (570 KB) | `exp_c02_mjx_scaffold/ppo_policy_full.msgpack` | `ppo_mjx.py --iters 1526 …` (27 min) |
| Gait videos (1.6 MB each) | `walker2d_sac_step650000.mp4`, `walker2d_mjx_ppo_200M_trackcam.mp4` | `eval_and_render.py` / `render_cpu.py` |
| Full per-iteration PPO log | `ppo_mjx_full.json` (482 KB) | downsampled copy **is** committed as `ppo_curve.csv` |

Both gait videos were posted to the Slack thread; happy to attach them to issue #75 on
request rather than committing them.

**Rendering gotcha:** render with the model's own camera (`camera="track"` — the
`<camera name="track" mode="trackcom">` defined in `walker2d_v5.xml`), *not* a camera
reconstructed from `DEFAULT_CAMERA_CONFIG`. Gymnasium uses the model camera
(`cam.type == mjCAMERA_FIXED`, `fixedcamid == 0`), which makes those config keys inert;
using the model camera reproduces gymnasium's frames **pixel-identically** (verified,
mean |diff| = 0.0), whereas a reconstructed one does not.

---

# Phase 1–2: LUT distillation and the representability curve (exp_c03)

**The keystone result: a LUT policy can represent this controller, and at a fraction of
the teacher's size.** Every number below is the deterministic 100-episode return in the
**CPU reference env** — never a training proxy.

Teacher: the 200M-step MJX/PPO policy (**5555.5 ± 34.4**). Dataset: 4,001,792
(obs → clipped teacher action) pairs, collected in batched MJX at 126k pairs/s (32 s),
half the envs driven with exploration noise for DAgger-style state coverage.

## The curve

| config | params | rows | held-out action MSE | return (CPU ref, 100 ep) | teacher retention | solved |
|---|---:|---:|---:|---|---:|:--:|
| hyperplane nap4 tph8 | 1,346 | 16 | 0.0255 | 441 ± 31 | 7.9% | — |
| **hyperplane nap4 tph32** | **5,378** | 16 | 0.0086 | **5512 ± 431** | **99.2%** | **✅** |
| hyperplane nap6 tph16 | 7,874 | 64 | 0.0080 | 4999 ± 1301 | 90.0% | ✅ |
| hyperplane nap8 tph16 | 26,882 | 256 | 0.0053 | 5484 ± 589 | 98.7% | ✅ |
| hyperplane nap6 tph64 | 31,490 | 64 | 0.0030 | 5583 ± 29 | 100.5% | ✅ |
| *fast* nap8 tph64 | 98,306 | 256 | 0.0101 | 4084 ± 1764 | 73.5% | ✅ |
| hyperplane nap8 tph64 | 107,522 | 256 | 0.0020 | 5584 ± 38 | 100.5% | ✅ |
| hyperplane nap10 tph64 | 404,738 | 1024 | 0.0015 | 5577 ± 38 | 100.4% | ✅ |
| hyperplane nap8 tph256 | 430,082 | 256 | 0.0012 | 5579 ± 33 | 100.4% | ✅ |
| *fast* nap10 tph256 | 1,572,866 | 1024 | 0.0037 | 5359 ± 828 | 96.5% | ✅ |
| hyperplane nap12 tph64 | 1,586,690 | 4096 | 0.0012 | 5569 ± 39 | 100.2% | ✅ |
| hyperplane nap10 tph256 | 1,618,946 | 1024 | 0.0013 | 5575 ± 32 | 100.4% | ✅ |

**Smallest LUT clearing 3000: 5,378 parameters → 5512 ± 431 (99.2% of teacher).**
For scale, the SAC actor is 73,484 parameters — so the smallest solving LUT is **~14×
smaller than a comparable MLP policy**, and it also beats the SAC baseline (5273.4).

Three things the curve says:

1. **The transition is a cliff, not a slope.** 1,346 params gives 441 (a policy that
   falls immediately); 5,378 gives 5512. Between those two points the policy goes from
   useless to essentially teacher-equivalent. Below the cliff the table cannot address
   finely enough to separate the states that matter.
2. **It saturates immediately above the cliff**, at ~5580 (100.4–100.5% of teacher)
   from 31k params onward — 300× more parameters buys nothing. Several configs slightly
   *exceed* the teacher, plausibly because behaviour cloning smooths the PPO mean.
3. **Learned addressing matters far more than table size.** At comparable size,
   FastMHL's fixed anchor pairs give 4084 ± 1764 where HyperplaneMHL gives 5584 ± 38 —
   and even at 1.57M params FastMHL only reaches 5359 ± 828. Note the *variance*: the
   fixed-anchor policies are erratic (±1764) where the learned ones are metronomic
   (±38). Spending parameters on *where to look* beats spending them on *what to store*.

Variance is itself a readout of competence here: every config at ≥100% retention has
σ ≈ 30–40 (never falls), while marginal ones sit at σ = 400–1800 (falls sometimes).

![representability curve](exp_c03_distillation/representability_curve.png)

## Two bugs that had to be fixed before any of this worked

Both produced a *silently* untrainable setup rather than an error, so they are worth
recording:

1. **The labels were outside the action space.** The PPO head is an unbounded Gaussian
   mean, but the environment applies `clip(a, -1, 1)` — so the teacher's *behaviour* is
   the clipped action. Regressing the raw mean put 63% of targets outside the reachable
   set (`mean(y²) = 7.53` vs 0.86 clipped) and weighted the loss towards magnitudes the
   env discards.
2. **A `tanh` on the student head.** The clipped teacher sits *exactly* at ±1 about 63%
   of the time; `tanh` reaches ±1 only asymptotically, so MSE demanded ever-growing
   pre-activations and the gradient died. The LUT stores arbitrary reals — emit raw and
   let the env clip, exactly as the teacher is clipped.

Before: action MSE 3.23 (*worse than predicting zeros*, whose MSE is 0.86) and a return
of 405. After: MSE 0.006 and a solved policy.

## Scaling limit found

`hyperplane nap12 tph256` (6.3M params) **OOMs at batch 4096**: the always-soft backward
materialises a full-K surrogate of shape [batch, n_tables, K] = 4096 × 256 × 4096, which
is ~17 G elements. This is a property of the soft-backward *training* path, not of the
LUT itself — the hard forward at that size is cheap. Rerunning at a smaller batch; the
curve has saturated far below this point regardless.

---

# The JAX LUT port (exp_c04)

The hard forward of `HyperplaneMultiHeadLUT` ported to JAX in ~10 jittable lines, so a
LUT policy can be evaluated inside the MJX rollout loop (gradient-free search needs only
the forward).

**Verified bit-for-bit against torch**, fp32, autocast off:

| config | max abs diff (JAX − torch) | verdict |
|---|---:|---|
| nap4 tph8 h1 | 0.000e+00 | **exact** |
| nap6 tph16 h1 | 0.000e+00 | **exact** |
| nap8 tph64 h1 | 0.000e+00 | **exact** |
| nap10 tph256 h1 | 1.073e-06 | fp32 sum-order (bound 8.0e-06) |
| nap12 tph64 h2 | 0.000e+00 | **exact** |

**The first attempt failed at max |Δ| = 1.4e-1** — table-entry-sized, i.e. *wrong rows*,
not arithmetic drift. Cause: **JAX defaults to TF32 for fp32 matmuls on GPU and torch
does not.** TF32's ~10-bit mantissa perturbs the pre-activation `aᵢ`, and near a decision
boundary that flips `1[aᵢ > 0]`, selecting a different row. The fix —
`precision=jax.lax.Precision.HIGHEST` — is a **correctness requirement for a rank-coded
primitive, not a performance knob**, and is commented as such in `jax_lut.py`. This is
the same discrete-flip failure the torch module's own docstring warns about for bf16,
arriving from the other framework.

The verification harness deliberately distinguishes "wrong row" from "sum-order noise"
and fails loudly rather than passing on a loose tolerance; the torch↔JAX handoff is an
.npz because the two frameworks live in separate venvs.

---

# Phase 3: the gradient-free ceiling (exp_c05)

Scalable ES over batched MJX rollouts. **Vanilla CMA-ES is inapplicable here** — it
maintains a d×d covariance, which at d ≈ 5–8k is 25–64 M entries and O(d³) per update —
so this uses OpenAI-ES (antithetic mirrored sampling + rank normalisation) and
sep-CMA-ES (diagonal covariance), both O(d).

Each run: 150 generations × 128 population × 2 episodes × horizon 400 = **15.36 M
env-steps**, ~20 min. MJX fitness is a horizon-400 proxy; the CPU-reference column is
the comparable number.

| run | params | MJX fitness | CPU reference (30 ep) | solved |
|---|---:|---:|---|:--:|
| MLP · sep-CMA-ES | 1,830 | 1391.6 | 2996.7 ± 913.8 | — (just below) |
| MLP · OpenAI-ES | 1,830 | 1353.4 | 2051.1 ± 157.7 | — |
| LUT · OpenAI-ES | 7,872 | 976.3 | 904.0 ± 222.6 | — |

**A clean negative, and a useful one.** At this budget gradient-free search does *not*
solve Walker2d — sep-CMA-ES reaches the edge of the bar with a σ of ±914, meaning it
still falls often, against PPO's 5555 and SAC's 5273. The harness demonstrably
optimises (mean fitness climbs from ~5 to ~1400 over 150 generations), which is what
Phase 3 was for; it simply needs far more budget to compete.

**The LUT is harder to evolve than the MLP** (904 vs 2051 under identical settings).
This does *not* contradict Phase 1, where a 5,378-parameter LUT represented the policy
at 99.2% retention. The gap is **searchability, not capacity**: 4.3× the search
dimension, and discrete addressing means a perturbation either changes nothing or jumps
to a different row — a rugged, partly-flat landscape that isotropic Gaussian ES handles
badly. **A LUT is easy to fill and hard to evolve.**

Single seed, one budget — indicative rather than settled.

---

# Phase 4: a LUT trained FROM SCRATCH by backprop (exp_c06)

**The headline: yes — 4406.9 ± 426.8, solved, from random init with no teacher.**

## The ported backward, and its verification

The differentiable backward is a **hybrid**, and reproducing that faithfully is the
whole job:

* **table weights** → the honest *hard* gradient: a 1-row scatter of `grad_out` at the
  row the forward actually selected (not a softmax-weighted average);
* **x, hyperplanes, temperatures** → the *soft* full-K surrogate
  `y = Σ_k softmax(ts_k/T_sel)·W[t,k,:]`, with the sign pattern pinned to the row the
  forward chose and the table weights held constant.

Rather than transcribe torch's softmax backward by hand (easy to get subtly wrong,
hard to notice), the soft path is written once as a forward function and differentiated
by JAX itself inside a `jax.custom_vjp`; `stop_gradient` on the weights reproduces
torch's `d_sel_soft` path exactly. Temperatures are parametrised as log T, as in torch,
so their gradients match without a chain-rule fixup.

**Verified against torch's custom autograd** (fp32, hard forward, autocast off):

| tensor | max abs Δ | max rel Δ |
|---|---:|---:|
| forward `y` | 0.000e+00 | **exact** |
| `grad_x` | 7.00e-07 | 8.5e-07 |
| `grad_w` (hyperplanes) | 2.03e-06 | 7.9e-07 |
| `grad_b` | 1.19e-06 | 6.0e-07 |
| `grad_weights` (table) | 2.86e-06 | 1.8e-07 |
| `grad_log_T_soft` | 7.15e-07 | 2.5e-06 |
| `grad_log_T_sel` | 7.15e-07 | 4.6e-07 |

**Worst relative disagreement 2.5e-06 — fp32 noise. PASS.**

**TF32 struck again, and more quietly this time.** The forward needed
`Precision.HIGHEST` because a TF32 sign flip picks a whole wrong row — an obvious,
loud failure. The *backward* needs it for a subtler reason: its einsums and the vjp's
GEMMs also default to TF32, which showed up as ~5e-3 relative gradient error. That is
small enough to pass a lax tolerance and be dismissed as "close enough", while
silently degrading every gradient. Pinning precision took the worst error from
**5.3e-03 to 2.5e-06 — a 2000× improvement.**

**A finite-difference caveat worth recording:** the *hard* forward is piecewise constant,
so its true derivative is 0 almost everywhere and finite-differencing it is
meaningless. The FD check therefore targets the soft surrogate — the thing the backward
actually claims to differentiate. At fp32 that probe initially reported 8e-1 relative
error on a gradient that is provably correct; it is dominated by cancellation noise
(~1e-7/ε), not by the derivative. A false alarm, not a bug.

## From-scratch training

PPO on the batched MJX loop with a random-init LUT as the policy (a small MLP critic is
scaffolding for the update, not the representation under test), gradients through the
verified surrogate.

* LUT: nap 8, tph 16 → **24,576 table + 2,304 addressing = 26,880 params**
* 600 iterations × 1024 envs × rollout 32 = **19.66 M env-steps in 5.1 min**
  (~64,600 env-steps/s)
* proxy return climbs 141 → 4245 monotonically

**CPU reference, deterministic, 100 episodes: 4406.9 ± 426.8 → SOLVED.**

## What the three routes together say

| route | LUT params | CPU reference | solved |
|---|---:|---|:--:|
| distillation from a PPO teacher | 5,378 | 5512 ± 431 | ✅ |
| **backprop from scratch (no teacher)** | **26,880** | **4406.9 ± 426.8** | **✅** |
| evolution (OpenAI-ES) | 7,872 | 904 ± 223 | ✗ |

A LUT can be **filled** (distillation), and it can be **trained** (backprop) — but at
this budget it cannot be **evolved**. The differentiable surrogate is doing real work:
it is the difference between 4407 and 904. For the #74 spiking track, that argues for
obtaining LUT tables by gradient training and *then* compiling them, rather than hoping
to search for them directly.
