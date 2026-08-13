# Deploying the exp19 LUT policy to the walker2d-viz demo — for nucstar

**REPLACES the first artifact, which fell over on the server.** Same two files, same paths,
same actor `name` — only the trained model changed, so this is a straight re-pull and
redeploy with nothing to reconfigure.

**Verified in gymnasium `Walker2d-v5` (the server's own env): mean return 6284.7 ± 319.5
over 30 episodes, worst episode 5421.3, 26/30 running the full 1000 steps.** The previous
artifact scored 4845.9 ± 1372.2 with a worst episode of 1097.5.

## The two files

| this folder | goes to | what it is |
|---|---|---|
| `fastlut_lse.py` | `server/actors/fastlut_lse.py` | the actor — pure numpy, no torch |
| `walker2d_fastlut_lse_exp19.npz` | `server/models/walker2d_fastlut_lse_exp19.npz` | 47 KB of weights + obs stats |

`walker2d_fastlut_lse_exp19_meta.json` is provenance only and does not need deploying.

**The actor `name` is unchanged: `fastlut_lse (exp19)`** — the server's default actor points
at this exact string, so it must not be renamed. (It was briefly renamed while the name was
under suspicion for the first artifact's failure; that suspicion was ruled out — see the
table below — and the original name is restored.) Both files keep their existing paths and
filenames, so the rebuild overwrites the old versions in place and leaves no stale duplicate.

## Deploy

Exactly as before (`ADDING_MODELS.md` §3–4) — commit on the **`landing`** branch, then:

```sh
cp fastlut_lse.py                    landing/walker2d-viz/server/actors/
cp walker2d_fastlut_lse_exp19.npz    landing/walker2d-viz/server/models/
# commit + push on `landing`, then on the VM:
rsync -av server/actors/ nucstar@<host>:/home/nucstar/walker2d-viz/server/actors/
rsync -av server/models/ nucstar@<host>:/home/nucstar/walker2d-viz/server/models/
ssh nucstar@<host> 'cd /home/nucstar/walker2d-viz && sudo docker compose up -d --build'
sudo docker compose logs server | tail     # must list  fastlut_lse (exp19)
```

No client change, no new Python dependency.

## What was wrong — and what is still unconfirmed

**Be aware: the original failure was never reproduced locally, so the fix below is a
gap-closing change, not a confirmed repair of an identified defect.** What was checked:

| suspect | verdict |
|---|---|
| obs dimension / ordering (17 vs 18, x excluded) | **ruled out** — server passes stock gymnasium obs; layout matches training exactly |
| server applying its own preprocessing | **ruled out** — `Sim.step` is `actor.act(self.obs)` then `env.step(action)`, nothing else |
| server harness (reset(seed=0), continuous stepping, auto-restart) | **ruled out** — reproduced exactly here: 3 episodes, mean 5270, lengths 878/1000/1000 |
| actor `name` mismatch → silent fallback to `random` | **ruled out** — two shipped actors already use spaces, and the client sets the option `value` as a DOM property, so the string round-trips. Renamed anyway, defensively |
| brittleness to solver settings | **ruled out** — score is *identical* across `iterations` 10→200 (the solver converges well inside 10 for these states) |
| observation distribution shift | **mild** — normalised deployment obs is mean +0.032, std 0.862, per-dim \|mean\| ≤ 0.36 |
| **velocity clipping** | **REAL, and fixed** — see below |
| `mujoco` 3.10.0 (here) vs 3.11.0 (server image) | **UNTESTED** — installing 3.11 needs network approval, which timed out. This is the one remaining difference between this box and the server |

### The real mismatch that was found and fixed

Gymnasium's Walker2d builds its observation as `concat(qpos[1:], clip(qvel, -10, 10))` — it
**clips the velocities**. Our training env (`warp_env.py`) did not. On a trained exp19
policy, **9.0% of velocity components exceed |10| (peak 73.9)**, so the deployed policy was
being fed a vector it had never seen in training, and its normalisation statistics had been
fitted to the unclipped distribution.

The model was retrained with `--obs-clip-vel 10.0`, which makes the training observation
byte-for-byte the one a gymnasium deployment produces. (`--solver-iters 100 --ls-iters 50`
was set at the same time for parity with stock MuJoCo, but measurement shows the solver
setting makes no difference here, so that part is cosmetic.)

The retrain is better on every axis that matters for a demo:

| | old artifact | **new artifact** |
|---|---:|---:|
| deployed mean (30 eps) | 4845.9 | **6284.7** |
| deployed std | 1372.2 | **319.5** |
| worst episode | 1097.5 | **5421.3** |
| full-1000 episodes | 25/30 | 26/30 |

The variance collapsed ~4× and the walker no longer has bad episodes — which is the
behaviour you want on a public demo. It is also a sign the train/deploy gap really did close:
under the old mismatched physics, "best training seed" and "best deployed seed" *disagreed*;
under matched physics they **agree**.

### If it still falls after redeploying

Then the cause is the one thing not testable from here. Useful things to send back:

1. `sudo docker compose logs server | tail -40` — confirm the startup line lists
   `fastlut_lse (exp19)`, and that there is no load traceback.
2. `python -c "import mujoco, gymnasium, numpy; print(mujoco.__version__, gymnasium.__version__, numpy.__version__)"` **inside the container**.
3. Whether the walker falls immediately *from a fresh reset*, or only right after switching
   actor mid-episode — `set_actor` does **not** reset the env, so selecting any actor while
   the previous one is lying on the ground shows a fallen walker until the next auto-restart.
4. The `md5sum` of the deployed `.npz` (should match `walker2d_fastlut_lse_exp19.npz` here).

## What the policy is

Anchor-pair LUT actor, tph=32, NAP=6 — 32 tables of 2⁶ rows × 6 outputs. Whole forward pass:

```
x      = (obs - obs_mean) / sqrt(obs_var + 1e-8)      # 17-dim, stats in the npz
bit_i  = 1[ x[a_i] - x[b_i] > 0 ]                      # 6 FIXED anchor pairs/table, MSB-first
row_t  = weights[t, addr_t]
means  = T·τ·log( (1/T)·Σ_t exp(row_t / τ) )           # τ = 0.0865, learned
action = clip(means, -1, 1)
```

Note it **clips** rather than tanh-squashing (unlike `lut_teacher.py`): this is a PPO
Gaussian policy and the training env applied `action.clamp(-1, 1)`.

## Provenance

| file (parent folder) | what |
|---|---|
| `../deploy_matched/` | the deployment-matched retrain (3 seeds, `--obs-clip-vel 10.0`) + its `.pt` checkpoints |
| `../deploy_matched/run_deploy_matched.sh` | the run |
| `../export_for_viz.py` | `.pt` → `.npz`, refuses to export if numpy and torch disagree > 1e-4 (actual **1.98e-6**) |
| `../verify_deploy.py` | imports this exact actor file through a mirror of the server layout, runs 30 gymnasium episodes |
| `../select_seed.py` | the seed table (`--dir /tmp/dmsel --train-dir deploy_matched`) |
| `../diagnose_deploy_gap.py`, `../robustness_probe.py` | the ruled-out-suspects checks above |
| `../rerun_ckpt/` | the earlier, mismatched-physics run that produced the failed artifact |
