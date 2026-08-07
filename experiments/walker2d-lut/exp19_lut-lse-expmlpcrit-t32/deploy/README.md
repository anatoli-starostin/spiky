# Deploying the exp19 LUT policy to the walker2d-viz demo — for nucstar

Everything here is ready to drop into `landing/walker2d-viz/server/`. Two files ship; nothing
else is needed. **Verified end-to-end in gymnasium `Walker2d-v5` (the server's own env):
mean return 4845.9 ± 1372.2 over 30 episodes, 25/30 running the full 1000 steps.**

## The two files

| this folder | goes to | what it is |
|---|---|---|
| `fastlut_lse.py` | `server/actors/fastlut_lse.py` | the actor — pure numpy, no torch |
| `walker2d_fastlut_lse_exp19.npz` | `server/models/walker2d_fastlut_lse_exp19.npz` | 47 KB of weights + obs stats |

`walker2d_fastlut_lse_exp19_meta.json` is provenance only — **it does not need to be
deployed** (the actor reads nothing from it). Copy it along if you want it in the repo.

Actor `name` (the dropdown label, and the registry key): **`fastlut_lse (exp19)`**. It is
unique against the existing actors (`lut_teacher`, `sac_baseline`, `lut_sac_c21`, `random`,
`zero`).

## Deploy

Per `ADDING_MODELS.md` §3–4 — commit to the **`landing`** branch, then rebuild on the VM:

```sh
# from a landing-branch checkout
cp fastlut_lse.py                    landing/walker2d-viz/server/actors/
cp walker2d_fastlut_lse_exp19.npz    landing/walker2d-viz/server/models/
# commit + push on `landing` (as nucstarbot), then:

rsync -av server/actors/ nucstar@<host>:/home/nucstar/walker2d-viz/server/actors/
rsync -av server/models/ nucstar@<host>:/home/nucstar/walker2d-viz/server/models/
ssh nucstar@<host>
cd /home/nucstar/walker2d-viz
sudo docker compose up -d --build      # actors/ and models/ are COPYed at build time — rebuild, not restart
sudo docker compose logs server | tail  # the startup line must list  fastlut_lse (exp19)
```

No client change is needed — the dropdown is populated from the server's `actors` message.
**No new Python dependency**: the actor is numpy-only, so `requirements.txt` is untouched and
the image stays torch-free.

## What the policy is

The anchor-pair LUT actor from `exp19_lut-lse-expmlpcrit-t32`, tph=32, NAP=6 — 32 lookup
tables of 2⁶ rows × 6 outputs. Forward pass (all of it):

```
x      = (obs - obs_mean) / sqrt(obs_var + 1e-8)          # 17-dim, stats stored in the npz
bit_i  = 1[ x[a_i] - x[b_i] > 0 ]                          # 6 FIXED anchor pairs per table, MSB-first
row_t  = weights[t, addr_t]                                # one 6-dim row per table
means  = T·τ·log( (1/T)·Σ_t exp(row_t / τ) )               # sum-scaled log-sum-exp, τ = 0.0843 (learned)
action = clip(means, -1, 1)
```

Two deliberate differences from the existing `lut_teacher.py`, both of which matter if you
read the code side by side:

- the table reduction is a **temperature-τ log-sum-exp**, not a plain sum (τ→∞ would recover
  the plain sum; τ→0 gives T·max). τ was learned during training.
- the action is **clipped, not tanh-squashed**. This is a PPO Gaussian policy and the
  training env applied `action.clamp(-1, 1)`; `tanh` would be a different function and would
  visibly change the gait.

## Which checkpoint, and why that one

**Seed 0 of a checkpointed re-run of exp19.** Two things worth knowing:

1. **exp19's original run saved no weights.** `ppo.py` had no `torch.save` at all (repo
   policy: "checkpoints never tracked; reproduce from config.json"), and the obs-normalisation
   stats were not saved either — without them the weights are unusable. So exp19 was re-run
   with a new, default-off `--save-model` flag. **The re-run did not reproduce exactly**:
   finals came out 5146.8 / 5373.9 / 5272.7 against the original 5400.5 / 5869.2 / 5389.6
   (−2.2% to −8.4%). These runs are evidently not bit-reproducible at fixed seed —
   nondeterministic CUDA reductions are the likely cause. Statistically the two runs agree;
   individual numbers do not.

2. **Seed 0 was chosen on DEPLOYED performance, not training return** — and the two criteria
   disagree:

   | seed | training ep_ret | deployed mean (30 eps) | full-1000 |
   |---:|---:|---:|---:|
   | **0** | 5146.8 (*worst*) | **4845.9** (*best*) | **25/30** |
   | 1 | 5373.9 (best) | 3881.9 | 21/30 |
   | 2 | 5272.7 | 3951.0 | 9/30 |

   The training number is a *stochastic* policy (log_std floor −1.897, so σ ≥ 0.15) on
   MuJoCo-Warp physics with a reduced solver (`iterations=10, ls_iterations=8`); the demo
   runs the *deterministic* mean action on stock `Walker2d-v5`. Different quantities — so
   the best-training seed is not the best seed to ship. Reproduce with `../select_seed.py`.

## Assumptions to confirm

- **Obs convention.** The actor assumes the server's `obs` is standard 17-dim Walker2d-v5
  (`concat(qpos[1:], qvel)`, x excluded) — which is the gymnasium default and matches how the
  policy was trained. If the server ever switches to
  `exclude_current_positions_from_observation=False` (18-dim), this actor silently reads the
  wrong features. It defensively slices `[:17]`, so it would not crash — it would just walk badly.
- **Physics gap is real and expected.** Trained on MuJoCo-Warp with a reduced solver, deployed
  on stock MuJoCo. The 4845.9 above already measures the deployed side, so it is the honest
  number — but it is why the demo figure is below exp19's headline 5553.
- **Variance.** 5/30 episodes fall early (min 1097.5). The walker does not always survive the
  full 1000 steps; that is the policy, not the port.
- `mujoco` version differs slightly between here (3.10.0) and the server image (3.11.0). Not
  expected to matter for Walker2d, but the 4845.9 was measured on 3.10.0.

## Provenance / reproducing

| file (parent folder) | what |
|---|---|
| `../rerun_ckpt/actor_s{0,1,2}.pt` | the torch checkpoints (weights + obs stats + config) |
| `../rerun_for_checkpoints.sh` | the re-run that produced them |
| `../export_for_viz.py` | `.pt` → `.npz`, with a torch-vs-numpy parity gate (**refuses to export above 1e-4**; actual worst **1.35e-6**) |
| `../verify_deploy.py` | builds a throwaway copy of the server layout, imports this exact actor file, runs 30 gymnasium episodes |
| `../select_seed.py` | the seed table above |

The `.pt` checkpoints are ~344 KB each and are **not** intended for git (repo policy); the
47 KB `.npz` is the deployable artifact.
