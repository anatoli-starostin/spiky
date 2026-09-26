# The QUANTISED exp19 LUT policy — a spiking-hardware-faithful variant

**This is an ADDITIONAL artifact, not a replacement.** Nothing in the parent `deploy/` folder
changes, and the existing actor name `fastlut_lse (exp19)` is untouched. This one registers
as **`fastlut_lse (exp19, quantised)`** and can be deployed alongside, or not at all.

It exists because the walker2d demo also runs a **handcrafted spiking network** that
reproduces this LUT teacher, and that network cannot consume continuous values: it encodes
each observation as a first-spike *tick* and decodes each action from a first-spike tick.
This policy was therefore fine-tuned with those two quantisers **in the training loop**, so
it is a policy that is *good at being quantised*, rather than a continuous policy that
merely survives it.

## The two quantisers — and the warning that goes with them

| | what |
|---|---|
| **INPUT** | the normalised observation is snapped to **128 Gaussian-companded buckets** (σ = 1), through **ONE shared monotone map over all 17 coordinates** |
| **OUTPUT** | the action mean is **clipped to [-1, 1]** and snapped to a **22-level uniform grid** (step 2/21 ≈ 0.0952) |

🔴 **Both must be applied. This artifact is not a drop-in for the continuous one.** The
policy was trained with them in the loop; running it without either is running a different
model. The actor file applies both internally, so as long as `fastlut_lse_quantised.py` and
`walker2d_fastlut_lse_exp19_quantised.npz` are deployed **together**, this is handled — but
mixing this `.npz` with the parent actor, or vice versa, will silently misbehave.

**The emitted action is strictly inside [-1, 1] and always exactly one of the 22 grid
points**, both rails included. That is by construction, not by clamping downstream.

⚠️ **The input map is shared across all 17 coordinates and must stay that way.** The LUT
addresses by comparisons *between* coordinates (`bit = 1[x[a] > x[b]]`), so a per-coordinate
scale or offset would change that comparison for every pair spanning two maps, and the
address bit would stop meaning "coordinate a exceeds coordinate b".

Both maps are baked into the `.npz` as plain arrays (`in_quant_edges`, `in_quant_dequant`)
and applied with `np.searchsorted` — **the server has no scipy**, so `erf`/`erfinv` are not
available there. The baked edges were verified to reproduce the training-time tick
assignment **bit-identically**.

## The files

| this folder | goes to | what it is |
|---|---|---|
| `fastlut_lse_quantised.py` | `server/actors/fastlut_lse_quantised.py` | the actor — pure numpy, no torch, no scipy |
| `walker2d_fastlut_lse_exp19_quantised.npz` | `server/models/walker2d_fastlut_lse_exp19_quantised.npz` | 50 KB: weights, anchors, obs stats, both quantiser tables |

`walker2d_fastlut_lse_exp19_quantised_meta.json` is provenance only and does not need
deploying.

## What the fine-tune did

Parent: `../deploy_matched/actor_s2.pt` (the checkpoint this folder's parent ships,
`final_ep_ret` 5966.3). 384 updates, 8192 envs, full cosine 3e-4 → 3e-5, matched physics
(`--obs-clip-vel 10 --solver-iters 100 --ls-iters 50`), observation normaliser **frozen** at
the parent's statistics so the fixed bucket edges stay calibrated.

**The headline eval** — 1024 envs × 2000 steps, deterministic mean action, matched physics,
both quantisers active — is **≈ 6291** for this checkpoint, against **6037** for the parent
measured the same way. Single seed; treat the exact number with the caution that deserves.

### The L2 out-of-band penalty (w = 0.3) — the part that matters for the spiking build

An earlier fine-tune with the same quantisers but no penalty raised return by +411 yet left
the **raw, pre-clip** LUT output exactly as sprawled as the parent (≈ 51% of components
outside [-1, 1]) and made the table weights *wider*. That is not an accident: the clip's
gradient is exactly zero outside [-1, 1], and an out-of-band action is free in both physics
(MuJoCo clamps `ctrl` to `ctrlrange`) and reward (the training env clamps before computing
the control cost). Nothing was pulling the raw output in.

So this run adds an explicit term:

```
loss += w * mean_batch( sum_o relu(|mu_raw[o]| - 1)^2 )        w = 0.3
```

which supplies the gradient the clip cannot. Measured:

| | parent | no penalty | **this artifact (w = 0.3)** |
|---|---:|---:|---:|
| raw output outside [-1, 1] | 51.6% | 53.9% | **~13%** |
| mean Stage-3 delay span (ticks) | 74.7 | 79.0 | **63.8** |
| dmax (ticks) | 84 | 96 | **81** |
| spiking episode length (ticks) | 302 | 314 | **299** |

Roughly **87% of the raw readout is now in-band**, and the spiking network built from these
tables would be *shorter* than the one built from the parent, where the un-penalised
fine-tune made it longer.

⚠️ **`dmax` is a max over the six action dims and dim 0 dominates it.** Five of six dims
shrink hard (dim 3: 66 → 55 ticks) while dim 0 barely moves (82 → 78), so the mean span
improves 15% but `dmax` only 4%. **Dim 0 alone now caps the episode length** — a per-dim
penalty weighted toward it would likely buy more than raising the global weight.

## A numerical detail worth knowing

The training-time straight-through estimator is `x + (xq - x).detach()`, which is **not
value-exact in float32**: for a large |x| the round-trip returns `xq` plus a
relative-epsilon perturbation. Two coordinates landing in the *same* bucket therefore came
out slightly unequal during training, and the LUT's `d > 0` tie was broken by float noise.

**This actor dequantises exactly**, so equal ticks give equal values and a tie
deterministically yields bit 0 — which is also what the spiking encoder does by construction,
and what a reproducible artifact requires. Cost of that choice, measured on 100,000 real
observations: **0.0018%** of (sample, table) address rows differ from the training-time
forward, **0.0400%** of samples have any differing table, and all of it sits in the saturated
end buckets (1.32% of scalars). The exporter's parity gate compares against the exact form
and passed at **5.68e-08**.

## Deploy

Same procedure as the parent folder (`ADDING_MODELS.md` §3–4) — this is purely additive, so
no existing file is overwritten:

```sh
cp fastlut_lse_quantised.py                     landing/walker2d-viz/server/actors/
cp walker2d_fastlut_lse_exp19_quantised.npz     landing/walker2d-viz/server/models/
# commit + push on `landing`, then on the VM:
rsync -av server/actors/ nucstar@<host>:/home/nucstar/walker2d-viz/server/actors/
rsync -av server/models/ nucstar@<host>:/home/nucstar/walker2d-viz/server/models/
ssh nucstar@<host> 'cd /home/nucstar/walker2d-viz && sudo docker compose up -d --build'
sudo docker compose logs server | tail    # must list  fastlut_lse (exp19, quantised)
```

No client change, no new Python dependency. **Not deployed by this commit** — the artifact is
committed for review; the rebuild/rsync is nucstar's call.

## Provenance and reproducibility

The three source files that produced this artifact are committed **here**, so the `.npz` can
be regenerated from a checkpoint without reaching into another branch:

| file (this folder) | what |
|---|---|
| `export_quantised.py` | `.pt` → `.npz` + meta, with the numpy/torch parity gate (worst **5.68e-08**) |
| `obs_quant.py` | the input quantiser exactly as used in training |
| `act_quant.py` | the output quantiser exactly as used in training |

```sh
python export_quantised.py --ckpt <the w=0.3 .pt> --out .
```

⚠️ **The training run itself is NOT on this branch.** It lives in the exp23 QAT work
(`experiments/walker2d-lut/exp23_qat_obs_quant/`, run dirs `qat_n22_full/` and
`qat_n22_l2/`, with the launcher `run_l2.sh`, the penalty-weight sweep `run_sweep.sh`, and
the analysis figures under `qat_n22_l2/analysis/`), which is uncommitted working-tree
material on the gpustar box at the time of writing. Only the deployable artifact and its
exporter are committed here. **The `.pt` checkpoint this was exported from is likewise not
committed** — the chapter's policy is that checkpoints are never tracked
(`experiments/walker2d-lut/README.md`) — but it is backed up outside the git tree at
`~/projects/ckpt_backups/`.

| elsewhere | what |
|---|---|
| `../deploy_matched/actor_s2.pt` | the parent checkpoint this was fine-tuned from |
| `../export_for_viz.py` | the continuous-model exporter this one is a sibling of |
