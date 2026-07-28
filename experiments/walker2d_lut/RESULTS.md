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
