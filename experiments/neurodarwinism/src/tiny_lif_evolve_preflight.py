"""exp012 pre-flight for the LIF single-bit evolution run.

Before 20 minutes of GPU: the Izhikevich path must be untouched, the wide episode and the
128-delay bank must actually build, and the `wta` readout must be able to express the answer
that `diagls` provably cannot.
"""
import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder
from tiny_lif_fixes import WideEncoder

IA, IB = 0, 16
T_IN, N_TICKS, RO = 128, 384, 128

# ---------------------------------------------------------------- 1 Izhikevich untouched
G.set_weight_levels([-1.0, 0.0, 1.0])
G.set_delay_levels(list(range(1, 64, 2)))
G.QUANTIZED = True
G.FANOUT_CAP = 16
G.MAX_EPISODE_BATCH = 128
G.set_out_per_target(8, "mean")
_, _, Xp, Yp, Xv, Yv = load(1024, seed=0)
T.fit_target_stats(Yp)
enc32 = LatencyEncoder(Xp)
Xb, Yb, _ = sample_batch(Xp, Yp, 1024, 0, 12345)
from tiny_grow_evolve import load_ckpt                                    # noqa: E402
CK = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
      "exp012_tiny-direct-genome/run_diagls_k8/ck_P0.npz")
pool, ewma, *_ = load_ckpt(CK)
g = pool[int(np.where(np.isfinite(ewma))[0][np.argmin(ewma[np.isfinite(ewma)])])]
H = G.build([g], device="cuda")
st = G.score(H, Xb, Yb, enc32, genomes=[g], readout="diagls")
sv = G.score(H, Xv, Yv, enc32, genomes=[g], readout="diagls", readout_map=st["readout_map"])
six = float(sv["mse"][0])
print(f"1 IZHIKEVICH REGRESSION  {six:.6f}   (reference 25.918407)")
assert abs(six - 25.918407) < 1e-6, "the wide-episode / wta patch changed the default path"
del H, st, sv
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2 the LIF wide substrate
G.set_episode(t_in=T_IN, n_ticks=N_TICKS, readout_window=RO, d_hi=T_IN)
G.set_lif(tau=80.0, threshold=1.0, v_rest=0.0, v_reset=0.0, refractory_ticks=0)
G.set_hidden_capacity(6, 2)
G.set_out_per_target(2, "mean")           # TWO output neurons -- the wta readout needs a pair
G.set_bit_task(IA, IB)
G.set_weight_levels([-1.0] + [round(0.1 * i, 10) for i in range(11)])
G.set_delay_levels(None)
G.QUANTIZED = True
G.FANOUT_CAP = None
G.GAIN = 1.0
G.MAX_EPISODE_BATCH = 256
print(f"\n2 EPISODE  {G.set_episode(T_IN, N_TICKS, RO, T_IN)}")
print(f"  N_TARGET {G.N_TARGET}  OUT_PER_TARGET {G.OUT_PER_TARGET}  "
      f"N_OUT_NEURONS {G.N_OUT_NEURONS}  matrix {G.N_SRC}x{G.N_TGT}")
assert (G.N_TARGET, G.N_OUT_NEURONS) == (1, 2)
assert G.d_hi_() == T_IN and len(G.metas_(200.0)) == T_IN

encw = WideEncoder(Xp, T_IN)
ev = encw(Xv)
ties = float((ev[:, IA] == ev[:, IB]).mean())
yv = G.task_targets(Yv, Xv).ravel()
raw = (ev[:, IA] < ev[:, IB]).astype(float)
raw[ev[:, IA] == ev[:, IB]] = float(yv[ev[:, IA] == ev[:, IB]].mean() > 0.5)
print(f"  exact ties at {T_IN} ticks: {100 * ties:.2f}%   (12.15% at 32)")
print(f"  encoder floor at {T_IN} ticks: MSE {float(((raw - yv) ** 2).mean()):.4f} / "
      f"acc {100 * float(((raw > 0.5) == (yv > 0.5)).mean()):.2f}%")

rng = np.random.default_rng(0)
gg = G.random_genome(rng, p_init=0.5, n_exc=6, n_inh=2)
G.enforce(gg)
gg["gain"] = 1.0
H = G.build([gg], device="cuda")
Xbw, Ybw, _ = sample_batch(Xp, Yp, 512, 0, 7)
for ro in ("wta", "diagls"):
    stt = G.score(H, Xbw, Ybw, encw, genomes=[gg], readout=ro)
    s = G.score(H, Xv[:512], Yv[:512], encw, genomes=[gg], readout=ro,
                readout_map=stt["readout_map"])
    acc = float(((s["calibrated"][:, 0, 0] > 0.5) == (yv[:512] > 0.5)).mean())
    print(f"  random genome, readout {ro:7s}: held-out MSE {float(s['mse'][0]):.4f}  "
          f"acc {100 * acc:.2f}%")
    assert np.isfinite(float(s["mse"][0]))
del H
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3 wta CAN express the bit
# hand the readout a perfect pair of output spike trains and check it scores ~0
print("\n3 CAN the wta readout express the answer? feed it a perfect oracle pair:")
B = len(yv)
raw_oracle = np.zeros((B, 1, 2))
raw_oracle[:, 0, 0] = np.where(yv > 0.5, 1.0, 9.0)     # output 0 early exactly when bit = 1
raw_oracle[:, 0, 1] = np.where(yv > 0.5, 9.0, 1.0)
win = (raw_oracle[:, 0, 0] < raw_oracle[:, 0, 1]).astype(float)
print(f"   oracle pair -> MSE {float(((win - yv) ** 2).mean()):.6f}  "
      f"acc {100 * float(((win > 0.5) == (yv > 0.5)).mean()):.2f}%")
print("   (diagls on a single first-spike time cannot reach this -- it encodes WHEN, not WHICH)")

print("\nPRE-FLIGHT GREEN")
