# walker2d_lut — solving Walker2d-v5 with a LUT policy

*Tracking issue: [#75](https://github.com/anatoli-starostin/spiky/issues/75) ·
branch `walker2d-lut` · implementer: gpustar (RTX 5090).
Methodology: [`claude/experiment-methodology.md`](../../claude/experiment-methodology.md).
Idea credit for the Walker2d-LUT direction: Андрей Белкин.*

The friendlier-tolerance control-task track for the LUT→spiking programme, and the
companion to [#74](https://github.com/anatoli-starostin/spiky/issues/74). A control
task tolerates approximation far better than a language transformer, so it is a
lower-stakes way to prove out the LUT-policy → spiking-compile pipeline end to end.

**Results: [`RESULTS.md`](RESULTS.md).** Both baselines are solved —
SAC **5273.4 ± 33.9**, MJX/PPO **5555.5 ± 34.4** (both 100-episode deterministic,
both measured in the CPU reference env).

## Roadmap status

| step | what | status |
|---|---|---|
| 1 | **SAC baseline** → save the actor as the distillation teacher | ✅ done, 5273.4 ± 33.9 |
| 1b | **GPU-parallel track** (MJX + PPO) — added for wall-clock and to enable step 3 | ✅ scaffolded + baselined |
| 2 | **LUT policy via distillation** from the SAC actor | not started |
| 3 | **LUT policy via CMA-ES** from scratch | not started |
| 4 | Bridge to #74 (hand the LUT policy to the spiking-construction track) | not started |

## Two environments, deliberately

| venv | holds | used for |
|---|---|---|
| `~/projects/spiky/.venv` | torch 2.9.1+cu130, SB3, gymnasium, mujoco, imageio | SAC baseline, CPU reference eval, video encoding, **and the future distillation arm** |
| `~/projects/walker2d_mjx/.venv` | jax 0.11 (cuda12), mujoco-mjx, brax, flax, optax — **no torch** | MJX batched simulation + PPO |

They are separate on purpose: JAX ships its own CUDA 12.9 stack and installing it
alongside torch's cu130 risked breaking a *running* baseline. Keeping them apart also
makes the framework split below unambiguous.

**Framework split (approved):** *distillation stays on PyTorch*, with autograd through
the real `HyperplaneMultiHeadLUT` / `FastMultiHeadLut` modules; *JAX/MJX is used only
for fast rollout generation*. The handoff between them is plain arrays.
For the CMA-ES arm this costs nothing — it is gradient-free, and a LUT *forward* is
just sign-tests + gather + sum, a few lines of JAX.

Always run JAX with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or it grabs ~75% of VRAM
and starves anything else on the card. Render with `MUJOCO_GL=glfw` (EGL throws
`EGLError` on this box and osmesa is not installed).

## `exp_c01_sac_baseline` — the pinned reference

SAC via stable-baselines3 at the issue's hyperparameters verbatim (see `config.json`).

```bash
cd exp_c01_sac_baseline
./launch.sh 0 1000000                     # detached-friendly; writes train_seed0.log
MUJOCO_GL=glfw python eval_and_render.py --episodes 100    # eval + gait MP4
python slack_progress.py --task <BODY_TASK_ID>             # optional Slack progress bar
```

`launch.sh` pins `OMP_NUM_THREADS=1`: without it torch's thread pool thrashes on the
[256,256] MLP and throughput collapses from ~233 fps to ~20 fps.

## `exp_c02_mjx_scaffold` — the GPU-parallel track

```bash
cd exp_c02_mjx_scaffold
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/projects/walker2d_mjx/.venv/bin/python

$PY bench_mjx.py                 # batched-sim throughput sweep (physics steps)
$PY bench_mjx_solver.py          # env-steps + the solver-setting tradeoff
$PY ppo_mjx.py --iters 1526 --num-envs 4096 --rollout 32 \
      --save-params ppo_policy_full.msgpack --out ppo_mjx_full.json
$PY cross_check.py --params ppo_policy_full.msgpack        # comparability check
MUJOCO_GL=glfw $PY render_cpu.py --params ppo_policy_full.msgpack --episodes 100
~/projects/spiky/.venv/bin/python encode_frames.py <frames.npz> <out.mp4>
```

| file | what |
|---|---|
| `mjx_walker2d.py` | Walker2d-v5 on MJX — same `walker2d_v5.xml`, same obs/reward/termination/reset-noise. Solver **10/8** (see RESULTS.md) |
| `ppo_mjx.py` | compact purejaxrl-style PPO; everything jitted, GPU-resident, no host round-trip per iteration |
| `bench_mjx.py`, `bench_mjx_solver.py` | throughput sweeps |
| `cross_check.py` | open-loop dynamics divergence + closed-loop return transfer across MJX@10/8, MJX@100/50, CPU |
| `render_cpu.py` | eval + render an MJX-trained policy **in the CPU reference env** |
| `encode_frames.py` | frames → MP4, run under the torch venv (the JAX venv has no imageio) |
| `ppo_smoke.py` | ⚠️ kept as a record: the brax-PPO route, which **does not work** (see below) |

### Why MJX and not Brax

MJX runs the *same MuJoCo model and dynamics* as `Walker2d-v5` — it loads the identical
`walker2d_v5.xml` — which keeps the comparison against the SAC reference honest. Brax's
`walker2d` is a reimplementation on Brax's own physics with different reward/termination
details.

Two findings settled it beyond preference:

* brax now warns on import that it **"is not actively being maintained"**, pointing at
  MJX / mujoco_playground;
* brax's PPO **crashes on jax 0.11** — it calls the removed `jax.device_put_replicated`.
  Hence the hand-written PPO in `ppo_mjx.py` rather than reusing brax's.

### Caveats to carry forward

* MJX's solver is not bit-for-bit MuJoCo-C on contacts, and Walker2d is contact-rich.
  **Any MJX number must be re-evaluated in the CPU `Walker2d-v5` env** — that is the
  reference, and `render_cpu.py` exists for exactly this.
* PPO is on-policy and needs far more environment steps than SAC for the same return.
  Compare **wall-clock to a given return**, and report sample counts separately.
* Scale reminder for step 3: CMA-ES cost is population × episodes × 1000 steps, so a
  256-population generation is ~2M env-steps. At the CPU baseline's 14.8k steps/s that
  is minutes per *generation*; on the batched GPU path it is seconds. That is the whole
  reason this track exists.
