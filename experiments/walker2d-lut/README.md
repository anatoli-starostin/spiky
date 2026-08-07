# walker2d-lut — pure-PyTorch GPU-batched Walker2D RL framework

A fully GPU-resident, **pure-PyTorch (no JAX)** reinforcement-learning framework for
`Walker2d-v5`, built as the fast-iteration substrate for the LUT → spiking policy
programme (companion to the `origin/walker2d-lut` JAX/MJX work). Physics runs batched on
the GPU via **MuJoCo-Warp**; the policy/value nets are eager PyTorch. Swappable
actor/critic architectures (MLP, hyperplane-LUT, anchor-pair LUT; LIF-detector next) drop
in behind one interface.

## Layout (spiky experiment convention)

Following `claude/experiment-methodology.md` — *"every experiment gets its own folder …
`experiments/<idea>/exp_<slug>/`, each holding at least `config.json`, `metrics.csv`, and
`summary.json` (plus any plots)"*. Here the idea is `walker2d-lut`; the framework source is
shared across every run so it lives once under `src/`.

```
walker2d-lut/
├─ src/                 all framework + tooling sources + kept probe outputs (see below)
├─ figures/            cross-experiment comparison charts
├─ cpu_sac_baseline/   single-CPU SB3-SAC reference baseline (4433)
│
├─ exp00_ppo-384-baseline              early PPO baseline (384 upd) — superseded
├─ exp01_ppo-sac-baseline             PPO-vs-SAC equal-data comparison
│  PPO stabilization series (768 upd):
├─ exp02_ppo-const-lr                 constant LR (late-training collapse appears)
├─ exp03_ppo-cosine-to-zero           cosine LR → 0
├─ exp04_ppo-cosine-logstd-floor      cosine → 1e-5 + log_std floor + entropy
├─ exp05_ppo-truncbootstrap-retnorm-kl  WINNER: +trunc-bootstrap +return-norm +KL-stop
│  Hyperplane-LUT policies (learned per-bit sign-tests, decoupled straight-through):
├─ exp06_lut-hyperplane-t32-mlpcrit   tph32 actor + MLP critic
├─ exp07_lut-hyperplane-t32-lutcrit   tph32 actor + tph32 LUT critic
├─ exp08_lut-hyperplane-t64-lutcrit   tph64 actor + tph64 LUT critic
├─ exp09_lut-hyperplane-t64-mlpcrit   tph64 actor + MLP critic
│  Anchor-pair LUT policies (FastMultiHeadLut, fixed addressing, tables train):
├─ exp10_lut-anchor-pair-t32          tph32 + MLP critic
├─ exp11_lut-anchor-pair-t64          tph64 + MLP critic
└─ exp12_lut-anchor-pair-t128         tph128 + MLP critic
```

Each `expNN_*/` holds the convention trio — `config.json` (full run config + description),
`metrics.csv` (per-seed per-update learning curve), `summary.json` (headline mean±std over
seeds) — plus a short `README.md`, the **raw per-seed run records** `ppo_s{0,1,2}.json`
(full history, preserved verbatim), the `*.gpu` / `agg.gpu` utilization traces, and that
experiment's plot(s). `exp01` additionally has `metrics_sac.csv` + `sac_s*.json` for the
SAC arm.

## Framework source (`src/`)

- `warp_env.py` — `WarpWalker2dVecEnv`: thousands of Walker2d envs stepped on the GPU
  (MuJoCo-Warp), state as torch tensors via zero-copy `wp.to_torch` — no host↔device
  transfer per step. Obs/reward/reset faithful to gymnasium `Walker2d-v5`. Optional
  **physics CUDA-graph capture** (`build_physics_graph`) collapses the many-kernel
  `mjw.step` dispatch into one graph replay → saturates the GPU. Exposes `true_next_obs`
  for exact truncation bootstrap.
- `models.py` — the **swappable interface**: `BaseActorCritic` (a new architecture
  implements only `forward(obs) -> (mean, value)`; base supplies the Gaussian head +
  PPO act/evaluate), a `REGISTRY` + `@register` decorator, `MLPActorCritic`, `QCritic`
  (SAC's arch-independent `Q(obs,act)`), and the LUT arches: `HyperLUTHead` /
  `hyperlut` / `hyperlut2` / `*_t64`, and `fastlut` (wraps `spiky.lutorch.FastMultiHeadLut`).
- `ppo.py` — GPU-resident PPO (preallocated on-device rollout buffers, GAE on GPU,
  `--graph` physics capture, `--compile`, cosine LR, log_std floor, exact truncation
  bootstrap, return normalization, KL early-stop).
- `sac.py` — GPU-resident SAC (on-device replay buffer, twin critics + targets, squashed
  Gaussian actor, auto entropy temperature). Reuses the same registry for the actor.
- `buffers.py` — `GPUReplayBuffer` (circular, all-GPU). `train.py` — single entry point
  (`--algo {ppo,sac}`). `probe_throughput.py`, `rollout_bench.py`, `capture_test.py`,
  `summarize_bench.py` — probes/tooling. `run_bench*.sh`, `progress_monitor*.py` — the
  per-experiment run orchestrators and live Slack progress bars. Kept probe **outputs** live
  here too: `probe_results.json` (throughput headline), `rollout_bench{,_graph}.json`
  (rollout eager-vs-graph), and `gpu_framework_summary.png` / `graph_capture_summary.png`.

## How to run

```bash
cd src
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
python train.py --algo ppo --arch mlp --envs 8192 --graph --updates 384          # PPO baseline
python train.py --algo ppo --arch fastlut --tables-per-head 64 --envs 8192 \
    --graph --updates 768 --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
    --ent-coef 0.0 --target-kl 0.02 --norm-returns                                # anchor-pair LUT
python train.py --algo sac --arch mlp --envs 8192 --graph --updates 10000 --utd 4 # SAC
```
Env stack (in `~/projects/spiky/.venv`): torch 2.13+cu130, `warp-lang`, `mujoco-warp`,
gymnasium, mujoco.

## Key results

| setting | result |
|---|---|
| **Physics throughput** (N=32768, graph) | **1.1M env-steps/s @ 98% GPU** (8191× the 135/s single-env CPU baseline) |
| **PPO stabilized** (exp05, 768 upd ×3 seeds) | **final 5952 ± 416, best 5985 ± 424, 0/3 collapse** |
| **Batched SAC** (exp01, UTD 4, equal data) | final 1759 ± 1175, high-variance — underperforms PPO in the 8192-env regime |
| **CPU SAC baseline** (`cpu_sac_baseline/`) | 4433 (single CPU env, SB3 SAC, 1M steps) |
| origin `walker2d-lut` exp_c01 SAC reference | 5273 (single CPU env, UTD=1, 1M steps) |
| **LUT — hyperplane** (best: exp09, t64 + MLP critic) | ~92% of MLP; LUT critic is the bigger deficit |
| **LUT — anchor-pair** (best: exp12, t128 + MLP critic) | **~102% of MLP**, 0/9 collapse, ~20-27% faster than hyperplane |

## PPO stabilization (exp02 → exp05)

The 768-update runs revealed **late-training instability**: with constant LR (`exp02`)
2/3 seeds partially collapsed in the last quarter. A sweep of fixes (all in `src/ppo.py`,
flags default to prior behavior):

| exp | recipe | final | best | collapsed |
|---|---|---:|---:|---:|
| exp02 | constant LR | 5278 ± 1281 | 6281 ± 585 | 2/3 |
| exp03 | cosine → 0 | 4638 ± 1393 | 5409 ± 537 | 1/3 |
| exp04 | cosine→1e-5 + log_std floor + ent 0.005 | 4634 ± 782 | 5149 ± 314 | 1/3 |
| **exp05** | **+ exact truncation bootstrap + return-norm + KL-stop, ent 0, lr_min 3e-5** | **5952 ± 416** | **5985 ± 424** | **0/3** |

The **two decisive fixes were return-normalization and the exact truncation bootstrap**
(they stabilize the value/advantage scale and correctly value 1000-step survivors);
dropping the entropy bonus reclaimed the peak that `exp04`'s exploration floors had capped.
The **KL early-stop never fired** (`avg_epochs_per_update = 4.0`) — a cheap safety net, not
the active ingredient. Cross-experiment charts in `figures/`.

## Excluded from git (gitignore discipline preserved)

- **SAC checkpoint `.zip`s** — `cpu_sac_baseline/run_seed0/*.zip` (trained SB3 policies,
  regenerable via `cpu_sac_baseline/train_sac.py`). Present locally only.
- **`.log` training logs** — excluded by spiky's global `*.log` `.gitignore`. The
  per-update learning curves they held are preserved in every folder's `*.json`/`metrics.csv`.
- **Checkpoints** (`*.pt`) — never tracked; reproduce from each `config.json`.
