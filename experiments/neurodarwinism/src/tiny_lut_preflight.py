"""exp012 pre-flight for the LUT target / LUT-decode readout.

Two things to prove before anything is launched:
  1 DEFAULT OFF. With no LUT installed, the six-dimension K=8 path scores exactly what it
    scored before -- the flag must be inert, so exp001-011 and the 40/10 path stay
    byte-identical.
  2 The LUT path runs end to end on the 17-8-1 substrate: target, decode, baseline, and a
    real forward pass, with the decode fixed in advance and nothing fitted.
"""
import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder
from tiny_lut_target import build_lut

DIM = 0
_, _, Xp, Yp, Xv, Yv = load(1024, seed=0)
T.fit_target_stats(Yp)
enc = LatencyEncoder(Xp)
Xb, Yb, _ = sample_batch(Xp, Yp, 1024, 0, 12345)

# ---------------------------------------------------------------- 1 the flag is inert
assert G.LUT_TABLE is None, "a LUT is installed at import time -- it must default to off"
G.set_weight_levels([-1.0, 0.0, 1.0])
G.set_delay_levels(list(range(1, 64, 2)))
G.QUANTIZED = True
G.FANOUT_CAP = 16
G.MAX_EPISODE_BATCH = 128
G.set_out_per_target(8, "mean")
from tiny_grow_evolve import load_ckpt                                    # noqa: E402
CK = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
      "exp012_tiny-direct-genome/run_diagls_k8/ck_P0.npz")
pool, ewma, *_ = load_ckpt(CK)
g = pool[int(np.where(np.isfinite(ewma))[0][np.argmin(ewma[np.isfinite(ewma)])])]
H = G.build([g], device="cuda")
st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls", readout_map=st["readout_map"])
six = float(sv["mse"][0])
print(f"1 DEFAULT-OFF  six-dim K=8 leader {six:.6f}   (reference 25.918407)")
assert abs(six - 25.918407) < 1e-6, "the LUT patch changed the default path"
del H, st, sv
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2 the LUT path
G.set_hidden_capacity(8, 0)
G.set_out_per_target(1, "mean")
G.set_weight_levels([round(0.1 * i, 10) for i in range(11)])
edges, lut, bt = build_lut(Yp[:, DIM])
tbl = np.concatenate([lut, [float(lut[np.digitize(Yp[:, DIM], edges)].mean())]])
G.set_lut_task(edges, tbl, DIM)
print(f"\n2 LUT installed: {len(lut)} decode values + silence -> {tbl[-1]:.4f}")
# the topology must actually be 17-8-1: ONE output neuron, not six broadcast against one
# target. This is the assertion the first version of this pre-flight was missing.
print(f"  N_TARGET {G.N_TARGET}  OUT_PER_TARGET {G.OUT_PER_TARGET}  "
      f"N_OUT_NEURONS {G.N_OUT_NEURONS}  N_SRC x N_TGT {G.N_SRC} x {G.N_TGT}")
assert (G.N_TARGET, G.N_OUT_NEURONS) == (1, 1), "the LUT task did not bind to one output"
assert (G.N_SRC, G.N_TGT) == (25, 9)

tv = G.task_targets(Yv)
assert tv.shape == (len(Yv), 1)
assert np.isin(tv.ravel(), tbl).all(), "a target value is not a LUT entry"
print(f"  target takes {len(np.unique(tv))} distinct values, all of them LUT entries")
print(f"  own-chance of the LUT target {G.task_baseline(Yv):.6f}")

# the decode really is exact: feed the TRUE bin in as the first-spike time and the MSE is 0
bins_v = G.lut_bins(Yv)
assert np.allclose(tbl[bins_v], tv.ravel()), "decoding the true bin does not reproduce y'"
print("  decoding the TRUE bin reproduces the target exactly (MSE 0) -- "
      "the readout can represent it")

rng = np.random.default_rng(0)
gg = G.random_genome(rng, p_init=0.5, n_exc=8, n_inh=0)
G.enforce(gg)
H = G.build([gg], device="cuda")
s = G.score(H, Xv, Yv, enc, genomes=[gg], readout="lut")
print(f"  a random 17-8-1 genome, LUT readout: held-out {float(s['mse'][0]):.6f}  "
      f"silent {float(s['silent'][0]):.3f}  distinct first-spikes {int(s['n_distinct'][0])}")
assert np.isfinite(float(s["mse"][0]))
assert s["readout_map"] is None, "the LUT readout must fit nothing"
del H, s
torch.cuda.empty_cache()

G.set_lut_task(None, None, None)
assert G.LUT_TABLE is None
print("\nPRE-FLIGHT GREEN -- LUT readout wired, default-off, and reversible")
