# walker2d-lut — a lookup table that learned to walk, and became a spiking network

Four stages, end to end:

**train a LUT policy with PPO → construct a spiking network from it in closed form →
quantise → deploy it in the live demo stand.**

The narrative document is
[`walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md`](walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md) —
the design, the numbers, and the traps. This README is the map: what is here, how to run it,
and what deliberately isn't.

The headline: a handcrafted spiking network of **2,889 neurons and 25,953 synapses**, running
~155 integer ticks per action on the `spnet` engine, is statistically indistinguishable from the
software policy it reproduces — **6312.2 ± 26.6 against 6253.4 ± 36.3** over ~520 episodes each
(+58.8 ± 45.0, +1.31σ), with 98.72% exact agreement on discretised actions and 100.000% within
one output level.

## Scope of this branch

This directory is **curated for the story, not for completeness**. It carries only what
supports the arc above, as a reviewable PR candidate. The full research trees — every
superseded run, every probe, every negative result — stay on `research/walker2d-lut`.
[What stays there](#what-is-not-here) is listed at the bottom, so nothing here implies a file
exists when it doesn't.

**Curated, but self-contained:** every step the blog post describes reproduces from this branch
alone, from the PPO training runs through the spiking construction to the closed-loop eval. See
[`REPRODUCE.md`](REPRODUCE.md). What's excluded is *superseded and alternative work*, not
anything the story needs.

## Layout

```
walker2d-lut/
├─ src/                                 the RL framework: 7 files, shared by every run
├─ exp05_ppo-truncbootstrap-retnorm-kl/ the winning PPO stabilization recipe
├─ exp19_lut-lse-expmlpcrit-t32/        the run that produced the deployable actor
│   └─ deploy/                          the exported float actor, and quantised/ beside it
├─ exp23_qat_obs_quant/                 the quantisation-aware fine-tune design
└─ walker2d-spiking/                    the analytic SNN construction + its observation set
```

Experiment folders follow `claude/experiment-methodology.md` — `config.json` (full run config),
`summary.json` (headline mean±std over seeds), and a `README.md`. Per-seed run records,
`metrics.csv`, `*.gpu` utilization traces and plots live with the full runs on
`research/walker2d-lut`; the two folders kept here carry the config + summary + README trio.

> Those per-experiment `README.md` files are preserved **as written for the full run**. They
> describe the run truthfully, but they reference companion files — raw per-seed JSON, plots,
> collection and plotting scripts, checkpoints — that this branch does not carry. Treat any
> filename in them that isn't listed in this README as living on `research/walker2d-lut`.

### Stage A — train a lookup table to walk

`src/` is a fully GPU-resident, **pure-PyTorch (no JAX)** RL framework for `Walker2d-v5`.
Physics runs batched on the GPU via **MuJoCo-Warp**; policy and value nets are eager PyTorch.

- **`warp_env.py`** — `WarpWalker2dVecEnv`: thousands of Walker2d envs stepped on the GPU, state
  as torch tensors via zero-copy `wp.to_torch`, no host↔device transfer per step. Obs / reward /
  reset faithful to gymnasium `Walker2d-v5`. Optional **physics CUDA-graph capture**
  (`build_physics_graph`) collapses the many-kernel `mjw.step` dispatch into one graph replay.
  Exposes `true_next_obs` for exact truncation bootstrap.
- **`models.py`** — the swappable interface: `BaseActorCritic` (a new architecture implements
  only `forward(obs) -> (mean, value)`; the base supplies the Gaussian head and PPO
  act/evaluate), a `REGISTRY` + `@register` decorator, and 14 registered architectures — `mlp`,
  the hyperplane-LUT family, and the anchor-pair `fastlut*` family that wraps
  `spiky.lutorch.FastMultiHeadLut`.
- **`ppo.py`** — GPU-resident PPO: preallocated on-device rollout buffers, GAE on GPU,
  `--graph` physics capture, `--compile`, cosine LR, log_std floor, exact truncation bootstrap,
  return normalization, KL early-stop. Runnable directly.
- **`ppo_qat_obs.py`** — the same loop with the training-time observation and action quantisers
  attached, for the QAT fine-tune of Stage C.
- **`obs_quant.py` / `act_quant.py`** — the Gaussian-companding observation quantiser and the
  uniform action quantiser.
- **`buffers.py`** — `GPUReplayBuffer` (circular, all on GPU).

`ppo.py` is the training entry point; `run_exp19.sh` calls it directly. This branch is
**PPO-only** — there is no SAC trainer and no `--algo` dispatcher.

#### exp05 — the PPO stabilization recipe, established on a **plain MLP** policy

`"arch": "mlp"` (`exp05_ppo-truncbootstrap-retnorm-kl/config.json`). This experiment is **not a
LUT run** — it is the fourth and winning step of a PPO-stabilization sweep, conducted on an
ordinary MLP actor-critic so the fix was isolated from any architecture question. 142,605
params, 8192 envs × 32 rollout × 768 updates, 3 seeds.

The recipe: cosine LR 3e-4 → 3e-5, log_std floor −1.897, **exact truncation bootstrap**,
**return normalization**, KL early-stop at 0.02, entropy 0.
**final 5952.1 ± 415.9, best 5985.3 ± 423.9, 0/3 collapsed.**

Return-normalization and the truncation bootstrap are the two that mattered — they stabilise the
value/advantage scale and correctly value 1000-step survivors. The KL early-stop never fired; it
is a safety net, not the active ingredient. Everything downstream inherits these flags. (The
exp00–exp04 steps that led here are on `research/walker2d-lut`.)

#### exp19 — the LUT run that produced the deployable actor

`"arch": "fastlut_lse_sum_expmlpcrit"` (`exp19_lut-lse-expmlpcrit-t32/config.json`, registered in
`src/models.py`). Two halves:

- **Actor** — a 32×64×6 anchor-pair `FastMultiHeadLut`: each of 32 tables is addressed by 6 bits,
  bit *i* being `1[x[a_i] > x[b_i]]`, and the 32 table rows are pooled by a **sum-scaled
  log-sum-exp** with a learned temperature τ. Addressing is fixed; the table entries train.
- **Critic** — exp10's MLP critic (obs → [256,256] Tanh, orthogonal gain 1.0) with **only its
  final linear readout replaced** by the matching sum-scaled log-sum-exp over the 256 penultimate
  per-unit contributions. Backbone bit-identical to exp10's; one new trainable scalar τ_c.

82,953 params, 8192 × 32 × 768 = **201M env-steps**, ~14 min/seed.
**final 5553.1 ± 223.6 over 3 seeds, 0/3 collapsed.**

Its own verdict, from the README beside it: the exponential critic readout is *harmless but
inert* — 5553.1 ± 223.6 against the exp17 control's 5403.8 ± 34.4 (Δ +149, |t| 0.93), and τ_actor
still drifts up toward the plain sum rather than down. The experiment ships not because that
hypothesis worked but because **seed 2 (final 5966.3) is the best LUT actor the programme
produced**, and it is the checkpoint everything downstream is built from.

### Stage B — construct the spiking network

`walker2d-spiking/` turns the trained table into an `spnet` network **in closed form** — no
backprop inside the network. Three stages: order detection → lookup → an amplitude-encoded
first-spike readout.

Ten scripts, and every one is either on the path or produces something shipped:

| script | role |
|---|---|
| `tiny_lut_quantised_pipeline.py` | **entry point** — builds and verifies the network |
| `tiny_lut_order_full.py` | the 136 comparators; the entry point imports `pair_list` from it |
| `tiny_lut_order_detect.py` | one dual-rail comparator unit; imported by the two above |
| `tiny_lut_quantised_export.py` | exports the actor artefact from a calibration JSON |
| `collect_teacher_io.py` | produces the 153k teacher input→output dataset |
| `stage3_cd_bigdata.py` | the 8-bit log-domain coordinate-descent readout fit |
| `bake_and_verify_actor.py` | bakes the fitted weight+offset pair in, verifies end to end |
| `eval_gtskew_large.py` | the paired ~520-episode walker eval behind the headline number |
| `tiny_lut_full_pipeline.py` | the earlier **delay-based** build |
| `tiny_lut_export_actor.py` | exports it to `server/models/spiking_lut_actor.npz` |

The last two are here because the demo stand still serves what they produce, as a separate
actor. They look like superseded history; they aren't.

`walker2d-spiking/data/distill_exp19_100k.npz` (21.6 MB) holds 100,000 real Walker2d
observations. Every verification number in the write-up is measured on it, and every script
resolves it relative to itself — so nothing needs a path argument.

The library side of this stage is `src/spiky/lutorch/fast_multi_head_lut.py`, which gains an
**opt-in, default-off** log-sum-exp table readout (the `exp_outputs` path exp19 uses). The
spiking simulator itself — `spiky.spnet` and its native kernels — is already on `main` and
unchanged here.

### Stage C — quantise

The spiking readout can only resolve so many output levels, so the *policy* is retrained to fit
what the network can express, rather than the network stretched to fit the policy.

**exp23** is that quantisation-aware fine-tune. It resumes from the exp19 seed-2 checkpoint
(`deploy_matched/actor_s2.pt`, final 5966.3 — a `.pt`, so never tracked; regenerate from exp19's
`config.json`) and trains on with two quantisers **in the loop**, using `src/ppo_qat_obs.py`:

- **input** — the normalised observation is snapped to 128 Gaussian-companded buckets (σ=1)
  through **one shared monotone map across all 17 coordinates**. Shared, not per-coordinate,
  because the LUT addresses by comparisons *between* coordinates — a per-coordinate map would
  change the meaning of every address bit spanning two of them.
- **output** — the action mean is clipped to [−1, 1] and snapped to a **22-level** uniform grid,
  which is exactly what the spiking Stage-3 first-spike readout can resolve.

Plus an L2 out-of-band penalty at w=0.3. The arm that ships is `qat_n22_l2` seed 0, **final
5867.3** — 1.7% below its unquantised parent, for a policy the spiking network can reproduce
exactly. Source: `exp23_qat_obs_quant/export_quantised.py` and the two PLAN documents;
provenance recorded in `walker2d_fastlut_lse_exp19_quantised_meta.json`
(`source_experiment`, `parent_checkpoint`).

`exp19_lut-lse-expmlpcrit-t32/deploy/` holds both exports side by side: the float actor
(`walker2d_fastlut_lse_exp19.npz`, seed 2, τ 0.086469) and, under `quantised/`, the QAT policy
(`walker2d_fastlut_lse_exp19_quantised.npz`, τ 0.093768) with the numpy actor implementations and
the exporter. Both exporters run a **parity gate** that refuses to write unless the numpy
artefact and the torch module agree (float path: max abs diff 2.0e-06).

### Stage D — deploy

Four served actor classes and their models live under `landing/walker2d-viz/server/`:

| model | served as |
|---|---|
| `spiking_lut_quantised_actor.npz` | **"Spiking LUT quantised (handcrafted SNN)"** — the headline network |
| `spiking_lut_actor.npz` | **"Spiking LUT (handcrafted SNN)"** — the delay-based build |
| `walker2d_fastlut_lse_exp19_quantised.npz` | "fastlut_lse (exp19, quantised)" — the software teacher |
| `walker2d_fastlut_lse_exp19.npz` | "fastlut_lse (exp19)" — the float actor |

The rest of the stand — client, `server.py`, Docker, Caddy — lives on `main` / `live/walker2d-viz`.

Note the served `spiking_lut_quantised.py` is **not** pure numpy: numpy does the input
companding and the output decode, but the network is built through `spiky.spnet` at construction
and run with `process_ticks` in `act()`. What the stand runs is the real simulator.

## Reproducing the headline result

**[`REPRODUCE.md`](REPRODUCE.md) is the full runbook** — every step the blog post describes, with
the command that actually runs it, marked as either runnable here or living on
`research/walker2d-lut`, and the number each one should produce. The short version:

The construction pipeline needs no arguments — both its inputs are committed and resolved
relative to the script:

```bash
cd walker2d-spiking
python tiny_lut_quantised_pipeline.py --gt-skew --no-tie-break --tau-m-out 31.257
```

which prints, on the shipped configuration:

```
CHECK D  census          : 2889 neurons, 25953 synapses, dmax 6
EPISODE (data-dependent): min 142  mean 154.7  max 167
STAGE 1 bit parity : 100.0000%   (0 bad of 12288)
STAGE 2 one-hot    : 0 none, 0 multi of 2048
STAGE 3 exact match on the 22-level grid: within-1-level 100.000% on all six dims
```

**`--gt-skew` does not imply `--no-tie-break`** despite what its help text suggests — pass both,
or you silently build the 3,025-neuron variant with the tie detectors still in.

Training (`ppo.py` is the entry point; `run_exp19.sh` calls it with these flags for 3 seeds):

```bash
cd src
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
python ppo.py --arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 --envs 8192 --graph \
    --updates 768 --seed 0 --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
    --ent-coef 0.0 --target-kl 0.02 --norm-returns --out ppo_s0.json
```

Environment (`~/projects/spiky/.venv`): torch 2.9.1+cu130, `warp-lang`, `mujoco-warp`,
gymnasium, mujoco.

## What is *not* here

Deliberately excluded, kept on `research/walker2d-lut`:

- **`experiments/walker2d_lut/`** (1,488 files) — the separate JAX/MJX c36 reproduction track.
- **exp00–04, exp06–18, exp20–22** and their `figures/` — the PPO stabilization sweep, the
  hyperplane-LUT and alternative anchor-pair architectures, and the LIF-actor runs. The story
  uses the exp05 recipe and the exp19 actor; the rest is context, not path.
- **All SAC** — `cpu_sac_baseline/` (the single-CPU SB3-SAC reference runs), `src/sac.py` (the
  batched GPU SAC trainer) and `src/train.py` (the `--algo` dispatcher that existed to choose
  between the two). This branch is PPO-only. The demo stand still *serves* two SAC-derived
  actors — "Walker2d SAC baseline" and "Walker2d LUT-SAC c21" — but those are deployed
  checkpoints already on `main`, not something this branch trains.
- **`src/` benchmarking tooling** — `probe_throughput.py`, `rollout_bench.py`, `capture_test.py`,
  `summarize_bench.py`, the `run_bench*.sh` orchestrators, `progress_monitor*.py`, and their
  probe outputs.
- **`exp012_tiny-direct-genome/` analysis, probe and deploy sub-trees** — the construction's
  ~50 result artefacts. Only the write-up came across, now at
  `walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md`.
- **`exp19.../distill/spiking/`** (~180 files) — the earlier *trainable*-SNN distillation R&D.
  The shipped network is the analytic construction instead.
- **The construction tree's 16 stepwise, diagnostic and negative-result scripts** — the Stage-1
  latch and gate stages, the Stage-2 and output-stage exploration, the Izhikevich probes, the
  membrane traces that dated the cross-inhibition, the on-policy recalibration, the paired action
  diagnostic, and the earlier and failed Stage-3 fits. Nothing imports them, none produces a file
  carried here, and none is cited in the write-up as the source of a reported number — but their
  *results are* reported there, so the write-up records where the code lives.
- **`exp23` sweep/probe/`qat_*` trees**, raw `.npy` dumps and `*.gpu` traces — dev clutter.
- **`src/spiky/lutorch/lif_layer.py`** + its test, and the `liflut_mlpexpcrit` /
  `liflayer_mlpexpcrit` architectures in `models.py` — the *trainable*-LIF actor line, which only
  ever served the excluded exp20/exp21. Nothing on the story's path imports them: the
  construction pipeline and both deployed spiking actors use `spiky.spnet` +
  `spiky.util.synapse_growth` only, and exp19 uses `FastMultiHeadLut`.
  **`src/spiky/lutorch/lif_multi_head_lut.py` and its test are untouched from `main`** — this
  branch is a no-op on them. Retiring a library class is not this PR's job; the reworked version
  stays on `research/walker2d-lut`.

Also never tracked anywhere, by gitignore discipline: training `*.log` files (the per-update
curves they held are preserved in each run's JSON) and checkpoints `*.pt` (reproduce from
`config.json`).
