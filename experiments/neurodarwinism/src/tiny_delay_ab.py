"""exp012: the fine-delay A/B. Does halving the delay step buy anything?

Two runs, identical in every respect except the delay grid:
  odd    1,3,..,63  (32 levels, a delay mutation moves 2 ticks)
  fine   1,2,..,64  (64 levels, a delay mutation moves 1 tick)

Beyond the headline MSE, the question the dissection raised: are the EVOLVED delays any less
prior-like on the finer grid? The prior is uniform over the grid, so 'prior-like' is measured
as the total-variation distance between the evolved delay histogram and uniform, plus a
chi-square against uniform. The inh->exc class is excluded -- it is pinned to 1 by the
operator and carries no information.
"""
import argparse
import json

import numpy as np

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

ODD = list(range(1, 64, 2))
FINE = list(range(1, 65))


def cls_of(i, j):
    src = "in" if i < G.N_IN else ("exc" if i < G.N_IN + G.N_EXC_MAX else "inh")
    tgt = ("exc" if j < G.N_EXC_MAX else
           "inh" if j < G.N_EXC_MAX + G.N_INH_MAX else "out")
    return f"{src}->{tgt}"


def leader(ckpt):
    from tiny_grow_evolve import load_ckpt
    pool, ewma, *_ = load_ckpt(ckpt)
    fin = np.where(np.isfinite(ewma))[0]
    return pool[int(fin[np.argmin(ewma[fin])])]


def prior_gap(g, levels):
    """How far the evolved, unpinned delays are from the uniform prior over the grid."""
    m = g["mask"] & ~G.PIN_DELAY
    d = g["delay"][m]
    obs = np.array([(d == L).sum() for L in levels], float)
    n = obs.sum()
    exp = n / len(levels)
    tv = 0.5 * np.abs(obs / n - 1.0 / len(levels)).sum()      # 0 = exactly the prior
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(levels) - 1
    # TV against a finite sample is NOT comparable across grid sizes -- with twice the bins and
    # the same n, a sample drawn from the exact prior looks further from uniform. So simulate
    # the null: n draws from the uniform grid, 2000 times, and report where the evolved
    # histogram falls in that null distribution.
    rng = np.random.default_rng(0)
    sim = rng.integers(0, len(levels), (2000, int(n)))
    cnt = np.stack([(sim == k).sum(1) for k in range(len(levels))], 1).astype(float)
    tv_null = 0.5 * np.abs(cnt / n - 1.0 / len(levels)).sum(1)
    return dict(n=int(n), tv_from_uniform=float(tv), chi2=chi2, dof=dof,
                chi2_over_dof=chi2 / dof, mean=float(d.mean()), median=float(np.median(d)),
                sd=float(d.std()), frac_ge_32=float((d >= 32).mean()),
                tv_null_mean=float(tv_null.mean()), tv_null_p95=float(np.percentile(tv_null, 95)),
                tv_percentile_in_null=float((tv_null < tv).mean() * 100),
                hist=obs.astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odd", required=True)
    ap.add_argument("--fine", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    G.set_out_per_target(8, "mean")
    G.set_weight_levels([-1.0, 0.0, 1.0])
    G.QUANTIZED = True
    G.FANOUT_CAP = 16
    G.MAX_EPISODE_BATCH = 256

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 12345)
    chance = T.constant_baseline(Yv)

    R = dict(chance=chance, arms={})
    for name, ck, levels in (("odd", a.odd, ODD), ("fine", a.fine, FINE)):
        G.set_delay_levels(levels)
        g = leader(ck)
        H = G.build([g], device=a.device)
        st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                     readout_map=st["readout_map"])
        mse = float(sv["mse"][0])
        del H, st, sv
        torch.cuda.empty_cache()

        # per-class delay stats on the unpinned classes
        ii, jj = np.where(g["mask"])
        per = {}
        for i, j in zip(ii, jj):
            per.setdefault(cls_of(i, j), []).append(int(g["delay"][i, j]))
        per = {k: dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                       frac_ge_32=float(np.mean(np.array(v) >= 32)))
               for k, v in sorted(per.items())}

        R["arms"][name] = dict(
            ckpt=ck, n_levels=len(levels), heldout_mse=mse,
            n_synapses=int(g["mask"].sum()),
            gain=float(G.gain_of(g)), inh_coeff=float(G.inh_coeff_of(g)),
            delay_prior_gap=prior_gap(g, levels), per_class=per,
            levels=list(map(int, levels)))
        print(f"{name:5s} {len(levels):3d} levels  held-out {mse:7.3f}  "
              f"syn {int(g['mask'].sum()):4d}  "
              f"TV-from-uniform {R['arms'][name]['delay_prior_gap']['tv_from_uniform']:.4f}  "
              f"chi2/dof {R['arms'][name]['delay_prior_gap']['chi2_over_dof']:.3f}", flush=True)

    R["delta_fine_minus_odd"] = R["arms"]["fine"]["heldout_mse"] - R["arms"]["odd"]["heldout_mse"]
    print(f"\ndelta (fine - odd) = {R['delta_fine_minus_odd']:+.3f}   chance {chance:.3f}")

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
