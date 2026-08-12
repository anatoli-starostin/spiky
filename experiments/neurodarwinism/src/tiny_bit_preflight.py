"""exp012 pre-flight for the sign-comparison bit task.

  1 DEFAULT OFF -- the six-dimension K=8 path must still score 25.918407 exactly.
  2 The bit path is coherent on the 17-8-1 substrate: target, baseline, topology, one real
    forward pass, and the accuracy channel.
"""
import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

IA, IB = 0, 16
_, _, Xp, Yp, Xv, Yv = load(1024, seed=0)
T.fit_target_stats(Yp)
enc = LatencyEncoder(Xp)
Xb, Yb, _ = sample_batch(Xp, Yp, 1024, 0, 12345)

# ---------------------------------------------------------------- 1 the flag is inert
assert G.BIT_TASK is None, "a bit task is installed at import time -- must default to off"
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
assert abs(six - 25.918407) < 1e-6, "the bit-task patch changed the default path"
del H, st, sv
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2 the bit path
G.set_hidden_capacity(8, 0)
G.set_out_per_target(1, "mean")
G.set_weight_levels([round(0.1 * i, 10) for i in range(11)])
G.set_bit_task(IA, IB)
print(f"\n2 BIT task 1[x_norm[{IA}] > x_norm[{IB}]]")
print(f"  N_TARGET {G.N_TARGET}  N_OUT_NEURONS {G.N_OUT_NEURONS}  "
      f"matrix {G.N_SRC} x {G.N_TGT}")
assert (G.N_TARGET, G.N_OUT_NEURONS) == (1, 1) and (G.N_SRC, G.N_TGT) == (25, 9)

tv = G.task_targets(Yv, Xv)
assert tv.shape == (len(Xv), 1) and set(np.unique(tv).tolist()) <= {0.0, 1.0}
print(f"  target is 0/1, P(1) held-out {float(tv.mean()):.4f}   "
      f"own chance {G.task_baseline(Yv, Xv):.4f}")
assert abs(G.task_baseline(Yv, Xv) - tv.mean() * (1 - tv.mean())) < 1e-9, "chance != p(1-p)"

rng = np.random.default_rng(0)
gg = G.random_genome(rng, p_init=0.5, n_exc=8, n_inh=0)
G.enforce(gg)
H = G.build([gg], device="cuda")
stt = G.score(H, Xb, Yb, enc, genomes=[gg], readout="diagls")
s = G.score(H, Xv, Yv, enc, genomes=[gg], readout="diagls", readout_map=stt["readout_map"])
print(f"  a random 17-8-1 genome: held-out MSE {float(s['mse'][0]):.4f}  "
      f"error rate {float(s['mse_action'][0]):.4f}  "
      f"(accuracy {100 * (1 - float(s['mse_action'][0])):.2f}%)  "
      f"silent {float(s['silent'][0]):.3f}")
assert np.isfinite(float(s["mse"][0]))
del H, stt, s
torch.cuda.empty_cache()

G.set_bit_task(None, None)
assert G.BIT_TASK is None
print("\nPRE-FLIGHT GREEN -- bit task wired, default-off, reversible")
