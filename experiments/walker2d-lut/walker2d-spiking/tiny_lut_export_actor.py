"""exp012: export everything the walker2d-viz actor needs into ONE .npz.

The affine decode coefficients are fitted HERE, using the very pipeline the actor will run,
so the deployed numbers are identical by construction rather than by re-derivation.
"""
import os

import numpy as np
import torch

import tiny_lut_full_pipeline as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                   "landing", "walker2d-viz", "server", "models", "spiking_lut_actor.npz")


def main():
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    ntr = len(x) - 4000
    n_cal = 1024
    # LEVER A: one SHARED quantile-equalised value->tick map, fitted on the TRAINING pool.
    # It has to be shared across all 17 coordinates -- the teacher uses all C(17,2)=136 pairs,
    # so the comparison graph is complete and a per-coordinate scale would stop tick(a)<tick(b)
    # meaning x[a]>x[b]. Same 128-tick window, so the tick budget is untouched.
    qtab = np.quantile(x[:ntr].ravel(), np.linspace(0, 1, F.T_IN))

    def qencode(v):
        r = np.searchsorted(qtab, np.asarray(v).ravel(), side="left").reshape(np.shape(v))
        return np.clip(F.T_IN - 1 - r, 0, F.T_IN - 1).astype(np.int64)

    ticks = qencode(x[ntr:ntr + n_cal])
    dims = [0, 1, 2, 3, 4, 5]
    F.GATE_TICK, F.EMIT = 140, 143
    net, ids, nsyn, n_ticks, nneur = F.build(Z, dims, True, settle=75)
    T = []
    for i in range(0, n_cal, 32):
        T.append(F.run(net, ids, ticks[i:i + 32], n_ticks, dims)[6])
    T = np.concatenate(T)
    del net
    torch.cuda.empty_cache()

    # the quantised-input LUT is the thing the circuit reproduces; fit the affine to it
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    W = Z["weights"].astype(np.float64)
    bits = (ticks[:, A_] < ticks[:, B_])
    code = (bits * (2 ** np.arange(5, -1, -1))).sum(-1)
    ws = W.reshape(32 * 64, 6)[code + (np.arange(32) * 64)[None, :]]
    lutq = 32 * float(Z["tau"]) * (np.log(np.exp(np.clip(ws / float(Z["tau"]), -60, 60)).sum(1))
                                   - np.log(32))
    coef = np.zeros((6, 2))
    for oi, o in enumerate(dims):
        t = T[:, oi].astype(float)
        f = t < n_ticks
        coef[o] = np.polyfit(t[f], lutq[f, o], 1)
        print(f"  dim {o}: affine slope {coef[o, 0]:+.6f} intercept {coef[o, 1]:+.6f} "
              f"(fired {100 * f.mean():.2f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        anchor_a=Z["anchor_a"], anchor_b=Z["anchor_b"],
        weights=Z["weights"].astype(np.float32),
        tau=np.float64(Z["tau"]), tables_per_head=np.int64(Z["tables_per_head"]),
        obs_mean=Z["obs_mean"].astype(np.float64), obs_var=Z["obs_var"].astype(np.float64),
        affine=coef,
        # the encoder's percentile calibration, so the actor reproduces encode() exactly
        # Lever A: the shared quantile table replaces the linear enc_lo/enc_hi map. The old
        # fields are still written so an older actor build keeps working.
        qtable=qtab.astype(np.float64),
        enc_lo=np.float64(np.percentile(x, 0.5)), enc_hi=np.float64(np.percentile(x, 99.5)),
        t_in=np.int64(F.T_IN), gate_tick=np.int64(140), settle=np.int64(75))
    print(f"\nwrote {os.path.abspath(OUT)}  ({os.path.getsize(OUT) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
