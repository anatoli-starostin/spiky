"""exp012 diagnostic: WHERE does the tiny net's MSE go, given its tau is already healthy?

The smoke run scored held-out tau +0.31 -- as good as exp009's 800-excitatory reservoir --
while its MSE sat well ABOVE the constant predictor. Those two facts together can only mean
the ordering is informative but the offsets are on the wrong scale: MSE is
`bias^2 + spread mismatch + residual`, and rank correlation is blind to the first two.

This prints the decomposition per output dimension so the go/no-go can say which term
dominates, and how much of it a per-dimension affine recalibration of the READOUT (not of the
network) would recover -- i.e. how much of the gap is a coding-convention artefact rather
than a modelling failure.
"""
import argparse
import json

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="a *_final.json from tiny_evolve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, summary = [], {}

    J = json.load(open(a.final))
    g = dict(mask=np.array(J["best"]["genome"]["mask"], bool),
             delay=np.array(J["best"]["genome"]["delay"], np.int64),
             weight=np.array(J["best"]["genome"]["weight"], np.float64))

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)

    tgt_pool = T.target_offsets(Yp)
    tgt = T.target_offsets(Yv)
    print(f"TARGET   pool mean {tgt_pool.mean():.2f} sd {tgt_pool.std():.2f} "
          f"| val mean {tgt.mean():.2f} sd {tgt.std():.2f}")
    print(f"CONSTANT baseline  pool {T.constant_baseline(Yp):.2f}   "
          f"val {T.constant_baseline(Yv):.2f}   (exp009 quoted 39.19)")

    H = T.build([g], device=a.device, w_ceiling=J["config"]["w_ceiling"])
    s = T.score(H, Xv, Yv, enc)
    first = s["first"][:, 0, :]
    print(f"\nPREDICTED offsets  mean {first.mean():.2f} sd {first.std():.2f}  "
          f"silent {s['silent'][0]:.3f}  distinct {s['n_distinct'][0]}")
    print(f"MSE {s['mse'][0]:.2f}   tau {s['tau'][0]:+.4f}\n")

    print("per dimension:")
    print("  d   pred_mean  pred_sd   tgt_mean  tgt_sd      r      mse    bias^2   "
          "scale_err   resid   mse_after_affine")
    tot = np.zeros(4)
    for d in range(first.shape[1]):
        p, t = first[:, d], tgt[:, d]
        r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        mse = float(((p - t) ** 2).mean())
        bias2 = float((p.mean() - t.mean()) ** 2)
        scale = float((p.std() - t.std()) ** 2)
        resid = mse - bias2 - scale
        # the best per-dimension affine rescaling of THIS prediction: t ~ alpha*p + beta
        aff = float(t.var() * (1 - r ** 2)) if p.std() > 1e-9 else float(t.var())
        tot += [mse, bias2, scale, aff]
        rows.append(dict(dim=d, pred_mean=float(p.mean()), pred_sd=float(p.std()),
                         tgt_mean=float(t.mean()), tgt_sd=float(t.std()), r=r, mse=mse,
                         bias2=bias2, scale_err=scale, resid=resid, mse_after_affine=aff))
        print(f"  {d}   {p.mean():8.2f} {p.std():8.2f}   {t.mean():8.2f} {t.std():7.2f}  "
              f"{r:+.3f}  {mse:7.2f}  {bias2:7.2f}  {scale:9.2f}  {resid:7.2f}  {aff:10.2f}")
    tot /= first.shape[1]
    summary = dict(mse=float(tot[0]), bias2=float(tot[1]), scale_err=float(tot[2]),
                   resid=float(tot[0] - tot[1] - tot[2]), mse_after_affine=float(tot[3]),
                   constant_val=T.constant_baseline(Yv), constant_pool=T.constant_baseline(Yp),
                   tau=float(s["tau"][0]), silent=float(s["silent"][0]),
                   n_distinct=int(s["n_distinct"][0]), mean_abs_r=float(np.mean(np.abs(
                       [x["r"] for x in rows]))))
    print(f"\n  MEAN            mse {tot[0]:7.2f}   bias^2 {tot[1]:7.2f}   "
          f"scale_err {tot[2]:7.2f}   after per-dim affine {tot[3]:7.2f}")
    print(f"  constant baseline on val {T.constant_baseline(Yv):7.2f}")
    print(f"  => recalibrating the READOUT alone would take {tot[0]:.2f} -> {tot[3]:.2f}, "
          f"which is {'BELOW' if tot[3] < T.constant_baseline(Yv) else 'still above'} "
          "the constant predictor.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(dict(source=a.final, per_dim=rows, summary=summary)), f,
                      indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
