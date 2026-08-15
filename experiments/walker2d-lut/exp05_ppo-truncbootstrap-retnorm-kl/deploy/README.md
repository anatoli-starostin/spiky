# Deploying the exp05 MLP PPO baseline to the walker2d-viz demo — for nucstar

The plain-MLP PPO baseline: the reference arm every LUT experiment is measured against, and
the first step of the LUT→spiking story. Intended as the demo's **baseline actor**, taking
the slot the removed SAC baseline used to hold.

**Verified in gymnasium `Walker2d-v5` (the server's own env): mean return 5331.2 ± 30.5 over
30 episodes, worst episode 5240.7, and 30/30 running the full 1000 steps.** It never falls.

## The two files

| this folder | goes to | what it is |
|---|---|---|
| `mlp_ppo.py` | `server/actors/mlp_ppo.py` | the actor — pure numpy, no torch |
| `walker2d_mlp_ppo_exp05.npz` | `server/models/walker2d_mlp_ppo_exp05.npz` | 564 KB of weights + obs stats |

`walker2d_mlp_ppo_exp05_meta.json` is provenance only and does not need deploying.

The actor `name` is **`mlp_ppo (exp05)`**. `Sim.set_actor` fails *silently* on an unknown
name (`if name in self.registry:`, no else) and a session's default is `random`, so if you
prefer a friendlier dropdown label, change it here **and** anywhere the server names it —
a mismatch presents exactly as "the walker falls over immediately".

## Deploy

Exactly as for `fastlut_lse` (`ADDING_MODELS.md` §3–4):

```sh
cp mlp_ppo.py                    landing/walker2d-viz/server/actors/
cp walker2d_mlp_ppo_exp05.npz    landing/walker2d-viz/server/models/
# commit + push on the deployment branch, then on the VM:
rsync -av server/actors/ nucstar@<host>:/home/nucstar/walker2d-viz/server/actors/
rsync -av server/models/ nucstar@<host>:/home/nucstar/walker2d-viz/server/models/
ssh nucstar@<host> 'cd /home/nucstar/walker2d-viz && sudo docker compose up -d --build'
sudo docker compose logs server | tail     # must list  mlp_ppo (exp05)
```

No client change, no new Python dependency.

## Trained deployment-matched — and why that is not optional

This artifact comes from a **retrain under deployment-matched physics**
(`--obs-clip-vel 10.0 --solver-iters 100 --ls-iters 50`), not from exp05's original run.
That is the same correction exp19 needed, for the same reason, and skipping it was measured
here rather than assumed:

> gymnasium's Walker2d builds its observation as `concat(qpos[1:], clip(qvel, -10, 10))` — it
> **clips the velocities**; `warp_env` does not.

Trained without the flag, the exp05 policy puts **9.96–14.87 % of velocity components outside
|10|** (peak 84.2), with at least one component clipped on more than half of all timesteps —
worth up to **6 σ** of shift in the normalised input, against statistics fitted on the
unclipped distribution. exp19 hit this and its first artifact scored 4845.9 ± 1372.2 on the
server with a worst episode of 1097.5.

**The same test was run here.** The un-matched exp05 checkpoint scored **6030.7 in the warp
training env** and **652.2 in gymnasium**, falling after ~193 steps — a 9× gap between what
training reports and what the server would show. With `--obs-clip-vel 10.0` the exposure is
exactly **0.000 %**, peak |qvel| lands on 10.00, and the velocity variance the normalisation
is fitted to drops ~4× (max `obs_var` 196–256 → 41.6–47.0).

| | un-matched checkpoint | **this artifact** |
|---|---:|---:|
| warp training-env eval | 6030.7 | 5335.8 |
| **gymnasium, 30 eps** | **652.2** | **5331.2** |
| std | — | **30.5** |
| worst episode | fell at ~193 steps | **5240.7** |
| full-1000 episodes | 0/5 | **30/30** |

## Seed selection

By **deployed** performance, per exp19's convention — not by training return. All three
deploy-matched seeds, 30 gymnasium episodes each:

| seed | mean | std | min | full-1000 | train-final |
|---|---:|---:|---:|---:|---:|
| 0 | 6013.0 | 1446.6 | 966.4 | 26/30 | 5721.6 |
| **1 (shipped)** | **5335.8** | **32.8** | **5276.9** | **30/30** | 5000.7 |
| 2 | 5803.3 | 198.3 | 4780.8 | 29/30 | 5194.8 |

**Seed 0 has the best mean and is not the right artifact.** Its std is 1446.6 and its worst
episode is 966.4 — it falls over roughly 4 times in 30. That is the profile exp19 explicitly
rejected: *"the variance collapsed ~4× and the walker no longer has bad episodes — which is
the behaviour you want on a public demo."* A visitor watching one episode is far more likely
to see a fall than to notice 700 points of mean.

Seed 1 never falls in 30 episodes and its worst episode (5276.9) is within 3 % of exp19's
shipped worst (5421.3). Seed 2 is the alternative if you want the higher mean and can accept
1 fall in 30 — say so and it is a one-line re-export.

## What the policy is

Plain MLP actor-critic; only the actor ships. Whole forward pass:

```
x      = (obs - obs_mean) / sqrt(obs_var + 1e-8)      # 17-dim, stats in the npz
h      = tanh(W0 x + b0)                              # [17 -> 256]
h      = tanh(W1 h + b1)                              # [256 -> 256]
means  = W2 h + b2                                    # [256 -> 6]
action = clip(means, -1, 1)
```

Note it **clips** rather than tanh-squashing: this is a PPO Gaussian policy and the training
env applied `action.clamp(-1, 1)`. 142,605 parameters, of which the actor trunk is what ships.

## Provenance

| file | what |
|---|---|
| `../config.json` | exp05's flags (bench7) — this retrain is those plus the three physics flags |
| `../run_deploy_matched.sh` | the run — 3 seeds parallel, 779 s wall, 262k env-steps/s/seed |
| `../select_seed.py` | the seed table above (`--dir <ckpt dir> --episodes 30`) |
| `../export_for_viz.py` | `.pt` → `.npz`, refuses to export if numpy and torch disagree > 1e-4 (actual **2.642e-06**) |
| `../verify_deploy.py` | imports this exact actor file through a mirror of the server layout, runs 30 gymnasium episodes |
| `../rerun_for_checkpoints.sh` | the earlier, **un-matched** run — the training-comparison arm, not deployable |

Checkpoints (`*.pt`) are not tracked, per the repo's policy; reproduce them from
`run_deploy_matched.sh`.
