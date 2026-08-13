# Walker2d: LUT → spiking network — curated narrative branch

**Branch:** `feature/walker2d-lut-to-spiking` (first draft, provisional name — renameable).
**Purpose:** collect *only* the pieces that support the end-to-end success story
— **train a LUT policy with PPO → construct a spiking network from it → quantise →
deploy the live demo** — as a reviewable PR candidate into `main`. Curated for
clarity, not completeness; the full research trees stay on `research/walker2d-lut`.

The narrative doc is **`experiments/walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md`**
(and gpustar's public write-up, *"A lookup table that learned to walk — and then
became a spiking network"*, on `gh-pages` at `walker2d-spiking/`).

## The arc, and the minimal code for each stage

### Stage A — train a lookup table to walk (PPO)
- `experiments/walker2d-lut/README.md` — the pure-PyTorch + MuJoCo-Warp GPU RL framework.
- `experiments/walker2d-lut/src/{models,ppo,ppo_qat_obs,buffers,obs_quant,act_quant}.py` — the
  framework core (policy/critic, PPO loop, the QAT-obs variant, the companding obs/act quantisers).
- `exp05_ppo-truncbootstrap-retnorm-kl/` (README+config+summary) — the winning **PPO
  stabilization recipe**: truncation-bootstrap + return-norm + KL early-stop.
- `exp19_lut-lse-expmlpcrit-t32/` (README+config+summary+run+verify+export_for_viz) — the run
  that produced the **deployable actor**: 32×64×6 anchor-pair `FastMultiHeadLut` +
  log-sum-exp pooling (learned τ=0.09377), 201M env-steps.

### Stage B — construct the spiking network (closed-form, no backprop inside the net)
- `experiments/walker2d-spiking/` (26 files) — the analytic **3-stage construction**
  (order → lookup → readout) that turns the trained table into an spnet network;
  *(relocated from the historically-misnamed `experiments/neurodarwinism/src/`)*;
  entry point **`tiny_lut_quantised_pipeline.py`** ("builds and verifies the network",
  identical wiring to the deployed actor), with its `tiny_lut_order_*` / `stage2` /
  `output_stage` / `stage3_*` siblings.
- `src/spiky/lutorch/{lif_layer,lif_multi_head_lut,fast_multi_head_lut}.py` (+ `tests/test_lif_layer.py`)
  — the LIF layer and LUT extensions the programme adds on top of the core library.
- The spiking simulator itself (`spnet`, `src/spiky/spnet/` + `native/`) is **already on main**.

### Stage C — quantise the weights (8-bit, log-domain)
- `exp19_lut-lse-expmlpcrit-t32/deploy/quantised/` — `act_quant.py`, `obs_quant.py`,
  `export_quantised.py`, `fastlut_lse_quantised.py`, + exported `*_quantised.npz` / `_meta.json`.
- `exp19.../deploy/` — the float `fastlut_lse.py` actor + `*.npz` / `_meta.json`.
- `exp23_qat_obs_quant/` — the **quantisation-aware fine-tune** design: `PLAN_spiking_quantised.md`,
  `PLAN_2026-08-13_amplitude_stage3.md`, `export_quantised.py` (the four deliberate constraints:
  128 Gaussian buckets, LSE pooling, 22-level output, the L2 out-of-band penalty w=0.3).

### Stage D — the deployed demo (sourced from `live/walker2d-viz`, the running stand)
- `landing/walker2d-viz/server/actors/{fastlut_lse,fastlut_lse_quantised,spiking_lut,spiking_lut_quantised}.py`
  — the served actor classes (updated to the running versions, incl. spike-viz methods).
- `landing/walker2d-viz/server/models/` — the served artifacts: `walker2d_fastlut_lse_exp19[_quantised].npz`+meta,
  and **`spiking_lut_quantised_actor.npz`**+meta (the deployed spiking policy, "Spiking LUT quantised").
- The rest of the web stand (client, server.py, Docker/Caddy) already lives on main/`live/walker2d-viz`.

## Deliberately EXCLUDED (kept on `research/walker2d-lut`, not part of the story)
- `experiments/walker2d_lut/` (1,488 files) — the separate JAX/MJX **c36 reproduction** track.
- `experiments/neurodarwinism/exp012_tiny-direct-genome/` analysis/probe/deploy sub-trees (kept only the write-up, now at `experiments/walker2d-spiking/WALKER2D_SPIKING_WRITEUP.md`).
- `exp00–04, exp06–18, exp20–22` — superseded / alternative-architecture runs; the story uses **exp05 recipe + exp19 actor**.
- `exp19.../distill/spiking/` (~180 files) — the earlier *trainable*-SNN distillation R&D; the shipped net is the **analytic** construction.
- `exp23` sweep/probe/`qat_*`/raw `.npy` trees, all `progress_monitor*.py`, `run_bench*.sh`, `*.gpu` traces — dev clutter.

## Notes for review (with gpustar)
- The served `spiking_lut_quantised.py` **builds the net via `spiky.spnet` (torch) at
  construction** (lazy import in `__init__`); the public write-up's "pure NumPy at inference"
  describes a numpy replay — worth reconciling which actor the PR should ship as canonical.
- Provisional branch name. The construction code was relocated from its historically-misnamed
  home `experiments/neurodarwinism/src/` to **`experiments/walker2d-spiking/`**; the
  `experiments/neurodarwinism/` path no longer exists on this branch.
