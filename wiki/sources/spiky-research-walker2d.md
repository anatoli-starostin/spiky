# Sources checklist: `research/walker2d-lut`, `live/walker2d-viz`, `sync/walker2d-viz-from-live`

Generated 2026-11-04 by a script from `anatoli-starostin/spiky` (read-only). Refs: `research/walker2d-lut` @ `04055f0c`, `live/walker2d-viz` @ `4eb7e770`, `sync/walker2d-viz-from-live` @ `a7bc2ea1`.

Line format: `- [ ] [path](url) — what it is; goes to: <Primitives|Optimisation|Images|LM small|LM NanoChat|Spikes|undecided|left out>`. Nothing is ticked. Folder links use /tree/, files /blob/. A grouped line (`path (+N more)`) stands for N files of one module group, listed in brackets. Family lines (`prefix*`) stand for many sibling experiment folders (count and 3 examples given). Branch-only paths are linked on their branch, not on main. `MODIFIED vs main` = file exists on main and the branch changes it. 'goes to' is a guess from names and the ideas map, to be confirmed.

Items in this file: **25**.

---

## A. research/walker2d-lut: experiments/walker2d-lut (exp00-exp24, shared neurodarwinism files excluded)

- [ ] [experiments/walker2d-lut/README.md](https://github.com/anatoli-starostin/spiky/blob/research/walker2d-lut/experiments/walker2d-lut/README.md) — MODIFIED vs main: walker2d-lut experiment series: PPO baselines to quantised LUT actors; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp00-05_*, cpu_sac_baseline](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — PPO/SAC MLP baselines and PPO stabilisation steps (exp00-05, cpu_sac) [7 folders, 63 files; e.g. cpu_sac_baseline, exp02_ppo-const-lr, exp05_ppo-truncbootstrap-retnorm-kl]; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp06-09_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — hyperplane LUT actors, T=32/64, MLP vs LUT critic (exp06-09) [4 folders, 36 files; e.g. exp06_lut-hyperplane-t32-mlpcrit, exp08_lut-hyperplane-t64-lutcrit, exp09_lut-hyperplane-t64-mlpcrit]; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp10-15b_*, repro_exp10_gpustar](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — anchor-pair LUT actors T=32/64/128, MLP/LUT critic (exp10-15b), GPU repro [8 folders, 73 files; e.g. exp10_lut-anchor-pair-t32, exp14_lut-anchor-pair-lutcrit-t64, repro_exp10_gpustar]; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp16-19_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — exp-output and log-sum-exp LUT actors (exp16-19) [5 folders, 204 files; e.g. exp16_lut-anchor-pair-t32-expout, exp18_lut-lse-lutcrit-t32, exp19_lut-lse-expmlpcrit-t32]; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp20-22_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — LIF-layer actors with MLP critic (exp20-22) [3 folders, 9 files; e.g. exp20_lif-actor-mlpexpcrit, exp21_liflayer-32-32-mlpexpcrit, exp22_liflayer-32-64-mlpexpcrit]; goes to: Spikes
- [ ] [experiments/walker2d-lut/exp23-24_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut) — quantisation-aware obs/actions (exp23) and pooling ablation (exp24) [2 folders, 81 files; e.g. exp23_qat_obs_quant, exp24_pooling-ablation]; goes to: Spikes
- [ ] [experiments/walker2d-lut/src/capture_test.py (+22 more)](https://github.com/anatoli-starostin/spiky/blob/research/walker2d-lut/experiments/walker2d-lut/src/capture_test.py) — PPO/SAC training code, env, models, quantisers [23 files: capture_test.py, progress_monitor22.py, train.py]; goes to: Spikes
- [ ] [experiments/walker2d-lut/figures/](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d-lut/figures) — walker2d result figures (5 files); goes to: Spikes

## B. research/walker2d-lut: experiments/walker2d_lut (exp_c01-c21, JAX/MJX, SAC LUT)

- [ ] [experiments/walker2d_lut/BUCKET_LINE_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/walker2d-lut/experiments/walker2d_lut/BUCKET_LINE_SUMMARY.md) — The bucket-LIF line (exp_c32 → exp_c35) - summary; goes to: Spikes
- [ ] [experiments/walker2d_lut/README.md](https://github.com/anatoli-starostin/spiky/blob/research/walker2d-lut/experiments/walker2d_lut/README.md) — branch `walker2d-lut` · implementer: gpustar (RTX 5090); goes to: Spikes
- [ ] [experiments/walker2d_lut/RESULTS.md](https://github.com/anatoli-starostin/spiky/blob/research/walker2d-lut/experiments/walker2d_lut/RESULTS.md) — Walker2d-v5 baselines - results (issue 75, roadmap step 1); goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c01-c08_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — baselines and scaffolds: SAC, MJX, distillation, JAX LUT, ES, robustness (c01-c08) [8 folders, 97 files; e.g. exp_c01_sac_baseline, exp_c05_es, exp_c08_sac_distill]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c09-c21_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — LUT SAC actor studies: addressing, anchors, hyperplane, seeds, controls (c09-c21) [13 folders, 281 files; e.g. exp_c09_lut_sac, exp_c15_hyperplane_torchfaithful, exp_c21_seed4_long]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c22-c29_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — matched power, EGGROLL, CMA, phase SAC, action/input quantisation (c22-c29) [8 folders, 230 files; e.g. exp_c22_matched_power, exp_c26_action_quant, exp_c29_magnitude_anchors]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c30-c31_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — LIF detectors as actor index front-end, incl. pure TTFS LIF (c30-c31) [3 folders, 84 files; e.g. exp_c30_lif_detectors, exp_c30b_lif_pmatched, exp_c31_pure_lif]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c32-c37_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — bucket-addressed LIF variants and boundaries (c32-c37) [6 folders, 159 files; e.g. exp_c32_bucket_lif, exp_c35_frozen_boundaries, exp_c37_bucket_32x64]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c38-c47_*](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — LIFMultiHeadLUT shapes, delay and table init (c38-c47) [11 folders, 362 files; e.g. exp_c38_mhl_6det_2bkt, exp_c42b_confirm, exp_c47_128tab_1det_16bkt_pertable]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c48-c54_*, exp_profiling](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut) — repro, delay-clamp ablations, ordering, frozen-critic probe (c48-c54), profiling [9 folders, 212 files; e.g. exp_c48_c36_repro, exp_c51_delay_const_shift, exp_profiling]; goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c32b_bucket_fixed/](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut/exp_c32b_bucket_fixed) — exp_c32b - the CORRECTED bucket-addressed LIF actor (33 files); goes to: Spikes
- [ ] [experiments/walker2d_lut/exp_c36b_seeds3to5/](https://github.com/anatoli-starostin/spiky/tree/research/walker2d-lut/experiments/walker2d_lut/exp_c36b_seeds3to5) — exp_c36b - reproducing the anchor: c36's 3/3 was a lucky draw (25 files); goes to: Spikes

Shared with `research/neurodarwinism` (left out): `experiments/neurodarwinism/*` files also appear on this branch and are excluded here by blob identity.


## C. live/walker2d-viz (deployed visualiser state)

- [ ] [.dockerignore](https://github.com/anatoli-starostin/spiky/blob/live/walker2d-viz/.dockerignore) — docker ignore file for deploying the visualiser; goes to: left out
- [ ] [experiments/neurodarwinism/exp012_tiny-direct-genome/](https://github.com/anatoli-starostin/spiky/tree/live/walker2d-viz/experiments/neurodarwinism/exp012_tiny-direct-genome) — quantised spiking LUT actor deployment files (2 files); goes to: Spikes
- [ ] [experiments/neurodarwinism/src/tiny_lut_export_actor.py](https://github.com/anatoli-starostin/spiky/blob/live/walker2d-viz/experiments/neurodarwinism/src/tiny_lut_export_actor.py) — export of tiny LUT actor for the spiking demo; goes to: Spikes

On this branch the actor code and models of `landing/walker2d-viz/server` are removed relative to main (not listed).


## D. sync/walker2d-viz-from-live

- [ ] [landing/walker2d-viz/](https://github.com/anatoli-starostin/spiky/tree/sync/walker2d-viz-from-live/landing/walker2d-viz) — snapshot of walker2d-viz brought to the deployed state, already merged into main (branch tip is an ancestor of main); goes to: Spikes
