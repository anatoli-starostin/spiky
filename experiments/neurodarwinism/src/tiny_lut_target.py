"""exp012: reshape the TARGET so a first-spike readout can represent it exactly.

The input side already cost us 3.8 MSE (the latency code). This is the output-side
complement: instead of asking the net to hit a centred/clipped/uniformly-quantised offset,
make the target literally a 32-entry lookup table indexed by the output's first-spike time.

  y'  =  LUT[ bin(y) ],   32 EQUAL-POPULATION bins,   LUT[b] = mean of y over bin b

LUT[b] as the bin mean is the MSE-optimal per-bin decode, so y' is the best any 32-level
first-spike readout could possibly emit.

A UNIT WARNING that governs how these numbers may be read. y' lives in the units of the raw
action, while the old target lives in offset units 0..32. Their MSEs are NOT comparable and
neither are their chances (the old one is 29.384; the new one is whatever var(y') is). The
only honest cross-target comparison is the RATIO MSE/own-chance -- the fraction of that
target's variance left unexplained -- so every number below is reported both ways.

Bins and LUT are fitted on the TRAINING pool only and applied unchanged to held-out.
"""
import argparse
import json

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_mlp_ceiling import apply_affine, fit_affine, mlp_forward, mse, snap_abs, train_mlp

N_LEVELS = 32


def build_lut(y_train, n=N_LEVELS):
    """Equal-population bins on the training pool -> (interior edges, per-bin mean)."""
    q = np.quantile(y_train, np.linspace(0, 1, n + 1))
    edges = q[1:-1].copy()                       # n-1 interior edges
    b = np.digitize(y_train, edges)
    lut = np.array([y_train[b == k].mean() if (b == k).any() else y_train.mean()
                    for k in range(n)])
    return edges, lut, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _, _, Xp, Yp, Xv, Yv = load(1024, seed=a.seed)
    T.fit_target_stats(Yp)
    yc_t, yc_v = Yp[:, a.dim], Yv[:, a.dim]                    # the continuous source target

    edges, lut, bt = build_lut(yc_t)
    bv = np.digitize(yc_v, edges)
    yt, yv = lut[bt], lut[bv]                                  # the NEW target y'

    chance = float(((yv - yv.mean()) ** 2).mean())
    old_chance = float(((T.target_offsets(Yv)[:, a.dim]
                         - T.target_offsets(Yv)[:, a.dim].mean()) ** 2).mean())
    R = dict(dim=a.dim, n_levels=N_LEVELS, chance_new_target=chance,
             chance_old_offset_target=old_chance,
             var_continuous=float(yc_v.var()), var_lut=float(yv.var()),
             within_bin_var=float(((yc_v - yv) ** 2).mean()),
             lut=lut.tolist(), edges=edges.tolist(),
             bin_counts_val=np.bincount(bv, minlength=N_LEVELS).tolist())
    print(f"dim {a.dim}   NEW target y' = LUT[bin(y)], {N_LEVELS} equal-population bins")
    print(f"  own-chance of y'            {chance:.6f}   (var of y' on held-out)")
    print(f"  var of the continuous y     {yc_v.var():.6f}")
    print(f"  within-bin variance lost    {R['within_bin_var']:.6f}  "
          f"= {100 * R['within_bin_var'] / yc_v.var():.2f}% of the continuous variance")
    print(f"  held-out bin occupancy      min {min(R['bin_counts_val'])}  "
          f"max {max(R['bin_counts_val'])}  (all {N_LEVELS} levels used: "
          f"{all(c > 0 for c in R['bin_counts_val'])})")
    print(f"  [the old offset target's own-chance was {old_chance:.3f}, "
          f"in DIFFERENT units -- compare ratios, not MSEs]\n")

    enc = LatencyEncoder(Xp)
    Ep_raw, Ev_raw = enc(Xp).astype(np.float64), enc(Xv).astype(np.float64)
    mu, sd = Ep_raw.mean(0), Ep_raw.std(0) + 1e-9
    Ep, Ev = (Ep_raw - mu) / sd, (Ev_raw - mu) / sd

    # The sine penalty is O(1) whatever the target is, but this target's MSE is ~0.5 where the
    # old offset target's was ~16. An absolute lambda of 0.5 is therefore negligible on the
    # old target and utterly dominant on this one -- which is what made the Dale arm score
    # WORSE than the arm that adds the encoder on top of it, an impossible ordering. Scaling
    # lambda by the target variance makes it dimensionless and comparable across targets.
    var_t = float(yt.var())

    def qat(xt_, xv_, **kw):
        """QAT on the 0.1 grid in grid x free-gain form, hard snap, then affine refit."""
        best = None
        for lam in (0.005, 0.02, 0.05, 0.2):
            m = train_mlp(xt_, yt, xv_, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed,
                          lam_max=lam * var_t, step=0.1, clamp=1.0, gain=True, **kw)
            s1, s2 = snap_abs(m["w1"]), snap_abs(m["w2"])
            pt, pv = mlp_forward(m, xt_, s1, s2), mlp_forward(m, xv_, s1, s2)
            v = mse(apply_affine(pv, fit_affine(pt, yt)), yv)
            if best is None or v < best[0]:
                best = (v, lam, float(np.corrcoef(pv, yv)[0, 1]))
        return best

    # ---- 1 free MLP, full precision, signed weights
    m = train_mlp(Xp, yt, Xv, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed)
    ab = fit_affine(m["pred_train"], yt)
    R["free_mlp"] = mse(apply_affine(m["pred_val"], ab), yv)
    R["free_mlp_r"] = float(np.corrcoef(m["pred_val"], yv)[0, 1])

    # ---- 2 + Dale and the 0.1 grid, quantization-aware
    v, lam, _ = qat(Xp, Xv, dale=True)
    R["dale_grid"] = v
    R["dale_grid_lambda"] = lam

    # ---- 3 + the latency-encoded input = the matched ceiling
    v, lam, r_ = qat(Ep, Ev, dale=True)
    R["matched_ceiling"] = v
    R["matched_ceiling_lambda"] = lam
    R["matched_ceiling_r"] = r_

    # the same ladder on the OLD offset target, so the comparison is like-for-like
    R["old"] = dict(chance=old_chance, free_mlp=10.659, dale_grid=16.566,
                    matched_ceiling=20.341)

    print(f"{'':34s}{'MSE':>12s}{'ratio to own chance':>22s}")
    for nm, key in (("free MLP 17→8→1 (full precision)", "free_mlp"),
                    ("+ Dale + 0.1 grid (QAT)", "dale_grid"),
                    ("+ latency input = MATCHED ceiling", "matched_ceiling")):
        print(f"{nm:34s}{R[key]:12.6f}{R[key] / chance:22.4f}")
    print(f"{'own chance':34s}{chance:12.6f}{1.0:22.4f}")
    print(f"  free-MLP r {R['free_mlp_r']:.4f}   matched-ceiling r {R['matched_ceiling_r']:.4f}")

    print(f"\nthe SAME ladder on the OLD offset target (ratios are the comparable column):")
    for nm, key in (("free MLP 17→8→1", "free_mlp"),
                    ("+ Dale + 0.1 grid", "dale_grid"),
                    ("+ latency input = MATCHED", "matched_ceiling")):
        print(f"{nm:34s}{R['old'][key]:12.3f}{R['old'][key] / old_chance:22.4f}")

    R["ratio_new"] = {k: R[k] / chance
                      for k in ("free_mlp", "dale_grid", "matched_ceiling")}
    R["ratio_old"] = {k: R["old"][k] / old_chance
                      for k in ("free_mlp", "dale_grid", "matched_ceiling")}
    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
