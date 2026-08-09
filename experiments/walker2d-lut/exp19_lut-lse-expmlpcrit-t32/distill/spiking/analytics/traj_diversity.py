"""(1) Trajectory + diversity overlay across experiments.

Overlays EWMA best, EWMA mean and the pool spread proxy (best - mean) on shared axes, and
quantifies DIVERSITY COLLAPSE: the round after which the best member stops being meaningfully
better than the average one, i.e. the point selection runs out of material to choose between.

    python traj_diversity.py                 # all experiments with data
    python traj_diversity.py k128 small8020  # a subset
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, available, load_genomes, load_history, summarise

COL = {"k32_baseline": "#4E79A7", "k128": "#E15759",
       "small8020": "#59A14F", "delay148": "#B07AA1"}

want = sys.argv[1:] or None
exps = {k: v for k, v in available().items() if not want or k in want}

fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
rows, summary = [], {}
for name, meta in exps.items():
    h = load_history(meta["history"])
    r = np.array([x["rnd"] for x in h])
    best = np.array([x["best"] for x in h])
    mean = np.array([x["mean"] for x in h])
    c = COL.get(name, "#777777")
    ax[0].plot(r, best, color=c, label=name, lw=1.4)
    ax[1].plot(r, mean, color=c, label=name, lw=1.4)
    ax[2].plot(r, best - mean, color=c, label=name, lw=1.4)

    s = summarise(h)
    if s["convergence_round"] is not None:
        for a in ax:
            a.axvline(s["convergence_round"], color=c, ls=":", alpha=0.55, lw=1.1)
    # true pool spread, available only at the final round (checkpoint holds the ewma vector)
    if meta["ckpt"]:
        _, ewma, _, _, nxt = load_genomes(meta["ckpt"])
        fin = ewma[~np.isnan(ewma)]
        s["final_pool_std"] = float(fin.std())
        s["final_pool_min"] = float(fin.min())
        s["final_pool_max"] = float(fin.max())
        s["final_pool_n"] = int(fin.size)
    summary[name] = s
    rows.append((name, s))

ax[0].set_ylabel("EWMA best (corrected tau)")
ax[1].set_ylabel("EWMA mean")
ax[2].set_ylabel("spread proxy: best - mean")
ax[2].set_xlabel("round")
ax[0].set_title("Steady-state distillation: trajectory and diversity collapse\n"
                "(dotted verticals = convergence round, best-mean < 0.01 for 10 rounds)")
for a in ax:
    a.grid(alpha=0.25)
    a.legend(fontsize=9)
fig.tight_layout()
p = f"{OUT}/trajectory_diversity.png"
fig.savefig(p, dpi=130)
print(f"wrote {p}")

hdr = (f"{'experiment':14s} {'rounds':>6s} {'peak':>8s} {'@rnd':>5s} {'final':>8s} "
       f"{'fin gap':>8s} {'converged@':>10s} {'syn/net':>9s} {'wall':>7s}")
print("\n" + hdr)
print("-" * len(hdr))
for name, s in rows:
    k = exps[name]["k"]
    conv = s["convergence_round"]
    print(f"{name:14s} {s['rounds']:6d} {s['peak']:+8.4f} {s['peak_round']:5d} "
          f"{s['final_best']:+8.4f} {s['final_gap']:8.4f} "
          f"{(str(conv) if conv is not None else 'never'):>10s} "
          f"{s['syn_last']//k:9,d} {s['wall_s']:7.0f}s")
    if "final_pool_std" in s:
        print(f"{'':14s} final pool spread: std {s['final_pool_std']:.4f}  "
              f"range [{s['final_pool_min']:+.4f}, {s['final_pool_max']:+.4f}]  "
              f"n={s['final_pool_n']}")
json.dump(summary, open(f"{OUT}/trajectory_diversity.json", "w"), indent=1)
print(f"\nwrote {OUT}/trajectory_diversity.json")
