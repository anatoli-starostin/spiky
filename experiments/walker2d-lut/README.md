# walker2d-lut — pure-PyTorch GPU-batched Walker2D RL framework

A fully GPU-resident, **pure-PyTorch (no JAX)** reinforcement-learning framework for
`Walker2d-v5`, built as the fast-iteration substrate for the LUT → spiking policy
programme (companion to the `origin/walker2d-lut` JAX/MJX work). Physics runs batched on
the GPU via **MuJoCo-Warp**; the policy/value nets are eager PyTorch. Swappable
actor/critic architectures (MLP today; hyperplane-LUT / LIF-detector next) drop in behind
one interface.

## What's here

**Framework**
- `warp_env.py` — `WarpWalker2dVecEnv`: thousands of Walker2d envs stepped on the GPU
  (MuJoCo-Warp), state as torch tensors via zero-copy `wp.to_torch` — no host↔device
  transfer per step. Obs/reward/reset faithful to gymnasium `Walker2d-v5`. Optional
  **physics CUDA-graph capture** (`build_physics_graph`) collapses the many-kernel
  `mjw.step` dispatch into one graph replay → saturates the GPU.
- `models.py` — the **swappable interface**: `BaseActorCritic` (a new architecture
  implements only `forward(obs) -> (mean, value)`; base supplies the Gaussian head +
  PPO act/evaluate), a `REGISTRY` + `@register` decorator, `MLPActorCritic` reference,
  and `QCritic` (SAC's arch-independent `Q(obs,act)`).
- `ppo.py` — GPU-resident PPO (preallocated on-device rollout buffers, GAE on GPU,
  `--graph` physics capture, `--compile`).
- `sac.py` — GPU-resident SAC (on-device replay buffer, twin critics + targets, squashed
  Gaussian actor, auto entropy temperature). Reuses the same registry for the actor.
- `buffers.py` — `GPUReplayBuffer` (circular, all-GPU).
- `train.py` — single entry point: `--algo {ppo,sac}` dispatches to the two backends.

**Probes / bench tooling** — `probe_throughput.py` (physics saturation), `rollout_bench.py`
(rollout eager-vs-graph), `capture_test.py` (unified torch+warp capture attempt — kept as
the record of why it fails: mujoco_warp uses conditional graph nodes torch can't capture),
`summarize_bench.py` (tables + charts), `run_bench{,2,3,4}.sh`, `progress_monitor*.py`.

**Benchmark results** — `bench2/` (fair equal-data PPO vs SAC, 3 seeds), `bench3/`
(3 PPO seeds ×384, parallel), `bench4/` (3 PPO seeds ×768), each with per-run `.json`
learning-curve histories + `.gpu` utilization samples; PNG charts
(`gpu_framework_summary`, `graph_capture_summary`, `ppo_vs_sac`, `sac_summary`,
`bench4_curves`).

**CPU SAC baseline** (`cpu_sac_baseline/`) — `train_sac.py` (SB3 SAC on single CPU
gymnasium env, the exp_c01 config), its learning curve, and the `run_seed0/` summary/monitor
(the trained checkpoint `.zip`s are **excluded**, see below).

## How to run

```bash
# throughput probe (seconds)
python probe_throughput.py
# PPO (recommended default), physics graph on
python train.py --algo ppo --arch mlp --envs 8192 --graph --updates 384
# SAC (off-policy), same registry actor
python train.py --algo sac --arch mlp --envs 8192 --graph --updates 10000 --utd 4
```
Env stack (in `~/projects/spiky/.venv`): torch 2.13+cu130, `warp-lang`, `mujoco-warp`,
gymnasium, mujoco. Set `WARP_CACHE_PATH=/tmp/warp_cache` (the default cache dir is
read-only in the sbox cage).

## Key results

| setting | result |
|---|---|
| **Physics throughput** (N=32768, graph) | **1.1M env-steps/s @ 98% GPU** (8191× the 135/s single-env CPU baseline) |
| **PPO** 384 updates ×3 seeds (≈101M steps) | best **4909 ± 53**, stable |
| **PPO** 768 updates ×3 seeds (≈201M steps) | best **6281 ± 585** — but **late-training instability**: kept improving through ~576 updates, then 2/3 seeds partially collapsed by 768 (final variance ±1281). Take the best/eval checkpoint; add LR decay / entropy annealing to hold late gains. |
| **Batched SAC** (UTD 4, equal ~82M steps) | best **3006 ± 1434**, high-variance — *underperforms PPO in the massively-parallel regime*. SAC's single-env sample-efficiency edge does not carry over when 8192 decorrelated envs feed on-policy PPO; see the caveat block atop `sac.py`. |
| **CPU SAC baseline** (this framework, `cpu_sac_baseline/`) | **4433** (single CPU env, SB3 SAC, 1M steps) |
| origin `walker2d-lut` exp_c01 SAC reference | **5273** (single CPU env, UTD=1, 1M steps) |

**Parallel-vs-sequential seeds:** running 3 seeds concurrently on one H100 is a *wash*
(aggregate ~494k/s ≈ a single run's 492k) because one N=8192 run already saturates GPU
compute (82% util). Pack concurrently only when a single run's GPU util is well under ~50%
(i.e. small-N sweeps).

## PPO stabilization (bench4–7)

The 768-update runs revealed **late-training instability**: with constant LR (`bench4`)
2/3 seeds partially collapsed in the last quarter (final 5278±1281 vs best 6281±585). A
sweep of fixes (all in `ppo.py`, flags default to prior behavior):

| run | recipe | final | best | best−final | collapsed |
|---|---|---:|---:|---:|---:|
| bench4 | constant LR | 5278 ± 1281 | 6281 ± 585 | ~1003 | 2/3 |
| bench5 | cosine → 0 | 4638 ± 1393 | 5409 ± 537 | 771 | 1/3 |
| bench6 | cosine→1e-5 + log_std floor + ent 0.005 | 4634 ± 782 | 5149 ± 314 | 515 | 1/3 |
| **bench7** | **+ exact truncation bootstrap + return-norm + KL-stop, ent 0, lr_min 3e-5** | **5952 ± 416** | **5985 ± 424** | **33** | **0/3** |

**Winning recipe (bench7) — collapse eliminated AND peak reclaimed simultaneously:**
cosine LR `3e-4 → 3e-5` (`--lr-schedule cosine --lr-min 3e-5`) + **log_std floor** std≥0.15
(`--logstd-min -1.897`) + **return normalization** (`--norm-returns`, reward scaled by
running discounted-return std) + **exact truncation bootstrap** (`warp_env` exposes
`true_next_obs`; GAE bootstraps `V(true_next)` at the 1000-step time-limit and zeroes only
on true termination) + **KL early-stop guard** (`--target-kl 0.02`) + **entropy 0**.

The **two decisive fixes were return-normalization and the exact truncation bootstrap**
(they stabilize the value/advantage scale and correctly value 1000-step survivors);
dropping the entropy bonus reclaimed the peak that `bench6`'s exploration floors had capped.
The **KL early-stop never fired** (`avg_epochs_per_update = 4.0`) — a cheap safety net, not
the active ingredient here. Net: bench7's *final* is now a trustworthy proxy for its *best*
(gap ~33), so "train and take the final" works. Curves: `four_way_stability.png`.

## Excluded from git

- **SAC checkpoint `.zip`s** — `cpu_sac_baseline/run_seed0/*.zip` + `ckpt/*.zip`
  (21 files, **~65 MB**): trained SB3 policies, regenerable via `train_sac.py`. Omitted to
  keep the repo light. (Present locally in `~/projects/walker2d_sac_pt/run_seed0/`.)
- **`.log` training logs** — excluded by spiky's global `*.log` `.gitignore`. The
  per-update learning curves they contained are preserved in the committed `bench*/*.json`
  and `*_results.json`.
