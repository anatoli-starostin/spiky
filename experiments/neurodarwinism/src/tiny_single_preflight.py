"""exp012 pre-flight for the single-output task.

Two things have to hold before a 50-minute run is worth starting:

  1 NOTHING CHANGED for the six-dimension case. set_target_dims defaults to None, so a K=8
    six-target genome must score bit-identically to what it scored before the patch.
  2 The single-dimension path is coherent: geometry, the 48-neuron output budget on one
    target, the per-dimension baseline, and one full evaluation that returns a finite MSE.

It also prints the per-dimension constant-predictor baselines, which are the only honest
yardstick for these runs -- the 6-dim 34.15 is not.
"""
import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

CK = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
      "exp012_tiny-direct-genome/run_diagls_k8/ck_P0.npz")
BATCH, SEED = 1024, 0


def base_setup():
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.FANOUT_CAP = 16
    G.MAX_EPISODE_BATCH = 128


_, _, Xp, Yp, Xv, Yv = load(BATCH, seed=SEED)
T.fit_target_stats(Yp)
enc = LatencyEncoder(Xp)
Xb, Yb, _ = sample_batch(Xp, Yp, BATCH, SEED, 12345)

# ---------------------------------------------------------------- per-dimension baselines
tv = T.target_offsets(Yv)
print("per-dimension constant-predictor baselines on the held-out split")
for d in range(6):
    print(f"  dim {d}:  chance {((tv[:, d] - tv[:, d].mean()) ** 2).mean():8.3f}   "
          f"sd {tv[:, d].std():6.3f}")
print(f"  all six (the 6-dim number): {T.constant_baseline(Yv):.3f}\n")

# ---------------------------------------------------------------- 1 regression: 6 dims
base_setup()
G.set_out_per_target(8, "mean")
assert G.TARGET_DIMS is None and G.N_TARGET == 6 and G.N_OUT_NEURONS == 48
from tiny_grow_evolve import load_ckpt                                  # noqa: E402
pool, ewma, *_ = load_ckpt(CK)
fin = np.where(np.isfinite(ewma))[0]
g = pool[int(fin[np.argmin(ewma[fin])])]
H = G.build([g], device="cuda")
st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls", readout_map=st["readout_map"])
six = float(sv["mse"][0])
# The pre-patch reference is the 25.918 that tiny_delay_ab printed, which is all the
# precision I have. So rather than assert against a digit I do not know, measure how much
# this evaluation moves on its own: re-score the identical genome on the identical batches
# and take the spread as the resolution of the comparison.
rep = []
for _ in range(3):
    H2 = G.build([g], device="cuda")
    s2 = G.score(H2, Xb, Yb, enc, genomes=[g], readout="diagls")
    v2 = G.score(H2, Xv, Yv, enc, genomes=[g], readout="diagls", readout_map=s2["readout_map"])
    rep.append(float(v2["mse"][0]))
    del H2, s2, v2
    torch.cuda.empty_cache()
spread = max(rep) - min(rep)
print(f"1 REGRESSION  six-dim K=8 leader held-out {six:.6f}  (pre-patch reference 25.918)")
print(f"  re-scored 3x: {['%.6f' % r for r in rep]}  spread {spread:.2e}")
assert abs(six - 25.918) < max(5e-3, 10 * spread), "the six-dimension path MOVED"
print("  per-dim held-out MSE:",
      " ".join(f"{v:.2f}" for v in ((sv["calibrated"][:, 0, :] - G.task_targets(Yv)) ** 2).mean(0)))
del H, st, sv
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2 the single-dim path
for dim in (5, 1, 0):
    base_setup()
    G.set_out_per_target(48, "mean")            # the whole 48-neuron budget on ONE target
    G.set_target_dims([dim])
    assert G.N_TARGET == 1 and G.OUT_PER_TARGET == 48 and G.N_OUT_NEURONS == 48, "geometry"
    assert G.N_TGT == 40 + 10 + 48
    rng = np.random.default_rng(SEED)
    gg = G.random_genome(rng, p_init=0.10, n_exc=40, n_inh=10)
    G.enforce(gg)
    assert gg["aff_a"].shape == (1,), gg["aff_a"].shape
    assert bool((gg["delay"][G.PIN_DELAY & gg["mask"]] == 1).all()), "delay pin"
    assert not gg["mask"][G.R_IN, G.C_OUT].any() and not gg["mask"][G.R_INH, G.C_OUT].any()
    ch = G.task_baseline(Yv)
    H = G.build([gg], device="cuda")
    st = G.score(H, Xb, Yb, enc, genomes=[gg], readout="diagls")
    sv = G.score(H, Xv, Yv, enc, genomes=[gg], readout="diagls", readout_map=st["readout_map"])
    m = float(sv["mse"][0])
    print(f"2 SINGLE dim {dim}: chance {ch:7.3f}  random-genome held-out {m:7.3f}  "
          f"ratio {m / ch:5.3f}  silent {float(sv['silent'][0]):.3f}  syn {int(gg['mask'].sum())}"
          f"  tau {sv['tau'][0]}")
    assert np.isfinite(m)
    # crossover and mutation must survive the 1-length affine
    h = G.mutate(G.crossover(gg, gg, rng), rng, p_affine=0.25, p_inhcoeff=0.25, p_gain=0.25)
    assert G.enforce(h)["aff_a"].shape == (1,)
    del H, st, sv
    torch.cuda.empty_cache()

print("\nPRE-FLIGHT GREEN")
