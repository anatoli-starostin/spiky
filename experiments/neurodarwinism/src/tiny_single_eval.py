"""exp012: score the single-output runs against their OWN per-dimension baseline.

The 6-dim 34.152 is meaningless here. A run trained on one target dimension has to be judged
against the best constant predictor OF THAT DIMENSION, which ranges from 24.5 (dim 1) to
47.4 (dim 5) -- so the raw MSEs of two single-output runs are not comparable to each other
either. The ratio MSE/chance is.

For each run: the final EWMA leader, its held-out MSE, its own chance, the ratio, the
bias^2 / scale / residual decomposition, and the correlation r between prediction and target.
The six-dimension K=8 leader is re-scored per dimension the same way, so the question the
brief asks -- did isolating the dimension help? -- has a matched comparison on both sides.
"""
import argparse
import json
import os

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

SIX = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/"
       "exp012_tiny-direct-genome/run_diagls_k8/ck_P0.npz")


def base_setup():
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.set_delay_levels(list(range(1, 64, 2)))
    G.QUANTIZED = True
    G.FANOUT_CAP = 16
    G.MAX_EPISODE_BATCH = 128


def leader(ckpt):
    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def decompose(pred, tgt):
    """MSE = bias^2 + (sd mismatch)^2 + the residual no affine could remove."""
    p, q = np.asarray(pred, float).ravel(), np.asarray(tgt, float).ravel()
    r = float(np.corrcoef(p, q)[0, 1]) if p.std() > 1e-12 else 0.0
    b2 = float((p.mean() - q.mean()) ** 2)
    se = float((p.std() - q.std()) ** 2)
    mse = float(((p - q) ** 2).mean())
    return dict(mse=mse, bias2=b2, scale_err=se, residual=mse - b2 - se, r=r,
                pred_sd=float(p.std()), target_sd=float(q.std()),
                pred_mean=float(p.mean()), target_mean=float(q.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated dim:ckpt pairs, e.g. '5:/path/ck_S0.npz,1:/path/..'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 12345)

    tv = T.target_offsets(Yv)
    chance = {d: float(((tv[:, d] - tv[:, d].mean()) ** 2).mean()) for d in range(6)}
    R = dict(chance_per_dim=chance, chance_six=T.constant_baseline(Yv), single={}, six={})

    # ---- the six-dimension K=8 leader, per dimension, for the matched comparison
    base_setup()
    G.set_out_per_target(8, "mean")
    G.set_target_dims(None)
    g = leader(SIX)
    H = G.build([g], device=a.device)
    st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
    sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls", readout_map=st["readout_map"])
    y6 = sv["calibrated"][:, 0, :]
    for d in range(6):
        R["six"][str(d)] = dict(decompose(y6[:, d], tv[:, d]), chance=chance[d],
                                ratio=float(((y6[:, d] - tv[:, d]) ** 2).mean() / chance[d]))
    R["six_overall_mse"] = float(sv["mse"][0])
    del H, st, sv
    torch.cuda.empty_cache()

    # ---- each single-output run
    for spec in a.runs.split(","):
        dim, ck = spec.split(":", 1)
        dim = int(dim)
        base_setup()
        G.set_out_per_target(48, "mean")
        G.set_target_dims([dim])
        gg = leader(ck)
        H = G.build([gg], device=a.device)
        st = G.score(H, Xb, Yb, enc, genomes=[gg], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[gg], readout="diagls",
                     readout_map=st["readout_map"])
        y = sv["calibrated"][:, 0, 0]
        dec = decompose(y, tv[:, dim])
        R["single"][str(dim)] = dict(
            dec, ckpt=ck, chance=chance[dim], ratio=dec["mse"] / chance[dim],
            n_synapses=int(gg["mask"].sum()), gain=float(G.gain_of(gg)),
            inh_coeff=float(G.inh_coeff_of(gg)), silent=float(sv["silent"][0]),
            n_distinct=int(sv["n_distinct"][0]),
            six_way_mse=R["six"][str(dim)]["mse"], six_way_ratio=R["six"][str(dim)]["ratio"])
        print(f"dim {dim}: chance {chance[dim]:6.2f}  single {dec['mse']:6.2f} "
              f"(ratio {dec['mse'] / chance[dim]:.3f}, r {dec['r']:.3f})   "
              f"vs 6-way {R['six'][str(dim)]['mse']:6.2f} "
              f"(ratio {R['six'][str(dim)]['ratio']:.3f}, r {R['six'][str(dim)]['r']:.3f})",
              flush=True)
        del H, st, sv
        torch.cuda.empty_cache()

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
