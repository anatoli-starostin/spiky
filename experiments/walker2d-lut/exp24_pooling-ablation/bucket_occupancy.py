"""Do the two arms' weights quantise differently? Bucket-occupancy comparison.

WHAT THE DEPLOY PATH ACTUALLY QUANTISES — checked, because the framing matters:

  * the ACTION mean is snapped to 22 LINEAR levels, `linspace(-clip, +clip, 22)`,
    step 2/21 = 0.0952  (src/act_quant.py, UniformActionQuantizer).
  * the WEIGHTS are snapped to 256 levels — 8 bits — uniform in `L = W / tau`, i.e.
    uniform in the LOG domain (stage3_cd_bigdata.py:74-78):
        L0 = W0 / tau ;  step = (L0.max() - L0.min()) / 255 ;  L = lo + round((L0-lo)/step)*step
    The grid is log-domain because the readout exponentiates: S = sum_t exp(L_t).

So "22 buckets" is the action grid, not a weight grid, and the real weight grid is
8-bit log-domain. `L = W/tau` is only defined for the log-sum-exp arm; a plain-sum
readout decodes linearly, so its natural weight grid is uniform in W itself.

This script therefore reports three things:
  1. each arm on ITS OWN natural weight grid (8-bit; log-domain for A, linear for B),
  2. both arms on a common 22-bucket grid over each arm's own weight range — the closest
     literal reading of the request, and a like-for-like evenness comparison,
  3. entropy and near-empty bucket counts for each, plus an overlaid bar chart.

Evenness is reported as normalised entropy H/log(B): 1.0 = perfectly uniform occupancy,
0.0 = everything in one bucket.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"A": "log-sum-exp (exp19)", "B": "plain sum (ablation)"}


def occupancy(vals, n_buckets, lo=None, hi=None):
    lo = vals.min() if lo is None else lo
    hi = vals.max() if hi is None else hi
    step = (hi - lo) / (n_buckets - 1)
    idx = np.clip(np.rint((vals - lo) / step).astype(int), 0, n_buckets - 1)
    counts = np.bincount(idx, minlength=n_buckets)
    p = counts / counts.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum() / np.log(n_buckets))
    return counts, H, float(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--empty-frac", type=float, default=0.001,
                    help="a bucket is 'near-empty' below this fraction of the weights")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    W, TAU = {}, {}
    for arm in ARMS:
        ws, taus = [], []
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}_s*.pt"))):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            ws.append(ck["state_dict"]["actor_lut.weights"].numpy().astype(np.float64))
            if "actor_lut.exp_outputs_tau_raw" in ck["state_dict"]:
                r = float(ck["state_dict"]["actor_lut.exp_outputs_tau_raw"])
                taus.append(float(np.log1p(np.exp(-abs(r))) + max(r, 0.0)))
        if ws:
            W[arm] = np.concatenate([w.reshape(-1) for w in ws])
            TAU[arm] = float(np.mean(taus)) if taus else None
    report = {}

    print("=" * 92)
    print("1. EACH ARM ON ITS OWN NATURAL 8-BIT WEIGHT GRID (the deploy path's)")
    print("=" * 92)
    for arm, w in W.items():
        if TAU[arm]:
            v, dom = w / TAU[arm], f"log domain, L = W/tau  (tau = {TAU[arm]:.6f})"
        else:
            v, dom = w, "linear domain, L = W  (no tau: plain sum decodes linearly)"
        c, H, step = occupancy(v, 256)
        empty = int((c < a.empty_frac * c.sum()).sum())
        print(f"\n{ARMS[arm]}  —  {dom}")
        print(f"   grid: 256 levels over [{v.min():+.5f}, {v.max():+.5f}], step {step:.6f}")
        print(f"   normalised entropy {H:.4f}   near-empty buckets {empty}/256   "
              f"occupied {int((c > 0).sum())}/256")
        print(f"   busiest bucket holds {c.max() / c.sum() * 100:.2f}% of weights")
        report[f"{arm}_own256"] = dict(domain=dom, entropy=H, near_empty=empty,
                                       occupied=int((c > 0).sum()),
                                       max_bucket_frac=float(c.max() / c.sum()))

    print("\n" + "=" * 92)
    print("2. BOTH ARMS ON A 22-BUCKET GRID over each arm's own weight range")
    print("=" * 92)
    occ22 = {}
    for arm, w in W.items():
        c, H, step = occupancy(w, 22)
        empty = int((c < a.empty_frac * c.sum()).sum())
        occ22[arm] = c
        print(f"\n{ARMS[arm]}   range [{w.min():+.5f}, {w.max():+.5f}]  step {step:.6f}")
        print(f"   normalised entropy {H:.4f}   near-empty buckets {empty}/22   "
              f"occupied {int((c > 0).sum())}/22")
        print(f"   busiest bucket {c.max() / c.sum() * 100:.2f}%   "
              f"outer two buckets each side: "
              f"{(c[:2].sum() + c[-2:].sum()) / c.sum() * 100:.2f}%")
        print("   counts: " + " ".join(f"{x:,}" for x in c))
        report[f"{arm}_22"] = dict(entropy=H, near_empty=empty,
                                   occupied=int((c > 0).sum()),
                                   max_bucket_frac=float(c.max() / c.sum()),
                                   counts=[int(x) for x in c])

    json.dump(report, open(os.path.join(a.out, "bucket_occupancy.json"), "w"), indent=1)

    # ---- figure --------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    CA, CB = "#2f6feb", "#d1730a"
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for x in ax:
        x.set_facecolor("white")
        x.grid(True, color=GRID, lw=0.8, zorder=0)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=9)

    b = np.arange(22)
    wdt = 0.4
    for arm, c, col, off in (("A", occ22.get("A"), CA, -wdt / 2),
                             ("B", occ22.get("B"), CB, +wdt / 2)):
        if c is None:
            continue
        ax[0].bar(b + off, c / c.sum() * 100, wdt, color=col, label=ARMS[arm])
    ax[0].set_title("22-bucket occupancy (each arm over its own weight range)",
                    color=INK, fontsize=11)
    ax[0].set_xlabel("bucket", color=MUTED)
    ax[0].set_ylabel("% of weights", color=MUTED)
    ax[0].set_xticks(b[::3])
    leg = ax[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    for arm, col in (("A", CA), ("B", CB)):
        if arm not in W:
            continue
        v = W[arm] / TAU[arm] if TAU[arm] else W[arm]
        c, _, _ = occupancy(v, 256)
        ax[1].plot(np.linspace(0, 255, 256), c / c.sum() * 100, lw=1.6, color=col,
                   label=f"{ARMS[arm]}"
                         + (" — log domain" if TAU[arm] else " — linear domain"))
    ax[1].set_yscale("log")
    ax[1].set_title("8-bit (256) occupancy on each arm's own deploy grid",
                    color=INK, fontsize=11)
    ax[1].set_xlabel("bucket index", color=MUTED)
    ax[1].set_ylabel("% of weights (log)", color=MUTED)
    leg = ax[1].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle("Pooling ablation — how the learned weights fall into quantisation buckets",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(a.out, "bucket_occupancy.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
