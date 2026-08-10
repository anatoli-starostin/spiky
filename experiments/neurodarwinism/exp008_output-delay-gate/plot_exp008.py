"""exp008 figure: best-EWMA trajectories and paired held-out tau, gated vs ungated."""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
ARMS = ("ungated", "gated")
SEEDS = (0, 1, 2)
# chapter analytics palette (src/analytics/*)
COLOR = {"ungated": "#4E79A7", "gated": "#E15759"}
BEST_RE = re.compile(r"^BEST member (\d+): EWMA ([+-][\d.]+)\s+HELD-OUT corrected ([+-][\d.]+)")


def final(arm, seed):
    out = None
    for line in open(os.path.join(D, f"{arm}_seed{seed}", "run.log"), errors="replace"):
        m = BEST_RE.match(line)
        if m:
            out = float(m.group(3))
    return out


def traj(arm, seed):
    p = os.path.join(D, f"{arm}_seed{seed}", f"steady_state_{arm}_s{seed}.json")
    h = json.load(open(p))
    return np.array([r["rnd"] for r in h]), np.array([r["best"] for r in h])


fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw=dict(width_ratios=[2.15, 1]), layout="constrained")

for arm in ARMS:
    curves = []
    for s in SEEDS:
        try:
            r, b = traj(arm, s)
        except FileNotFoundError:
            continue
        ax.plot(r, b, color=COLOR[arm], lw=0.9, alpha=0.45, zorder=2)
        curves.append((r, b))
    if curves:
        n = min(len(b) for _, b in curves)
        m = np.mean([b[:n] for _, b in curves], axis=0)
        ax.plot(curves[0][0][:n], m, color=COLOR[arm], lw=2.2, zorder=3,
                label=f"{arm} (mean of {len(curves)})")

ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("best EWMA fitness in pool", fontsize=10)
ax.set_title("Best-genome fitness trajectory\nthin = individual seeds, thick = arm mean",
             fontsize=10.5, loc="left")
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower right")

pairs = [(s, final("ungated", s), final("gated", s)) for s in SEEDS]
pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
for i, (s, u, g) in enumerate(pairs):
    bx.plot([0, 1], [u, g], color="0.75", lw=1.1, zorder=1)
    bx.scatter([0], [u], s=64, color=COLOR["ungated"], zorder=3)
    bx.scatter([1], [g], s=64, color=COLOR["gated"], zorder=3)
    bx.annotate(f"seed {s}", (1, g), textcoords="offset points", xytext=(9, -3),
                fontsize=8, color="0.35")
if pairs:
    u = np.array([p[1] for p in pairs])
    g = np.array([p[2] for p in pairs])
    bx.plot([0, 1], [u.mean(), g.mean()], color="0.2", lw=2.4, zorder=4)
    bx.scatter([0, 1], [u.mean(), g.mean()], s=90, color="0.2", marker="_", zorder=5)
    bx.text(0.5, 0.03, f"paired diff {np.mean(g - u):+.4f} +/- {np.std(g - u, ddof=1):.4f}",
            transform=bx.transAxes, ha="center", fontsize=9, color="0.25")
bx.set_xticks([0, 1])
bx.set_xticklabels(["ungated", "gated"], fontsize=10)
bx.set_xlim(-0.35, 1.5)
bx.set_ylabel("held-out corrected tau-b", fontsize=10)
bx.set_title(f"Held-out tau, paired by seed\nblack = mean of {len(pairs)} pairs",
             fontsize=10.5, loc="left")
bx.grid(axis="y", color="0.92", lw=0.6)
bx.set_axisbelow(True)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)

fig.suptitle("exp008 — output delay gate (output synapses in delay bank [64,80]) vs ungated "
             "control · K=32, 300 rounds, stdp_lr 0.01, batch 64", fontsize=11, x=0.005,
             ha="left")
p = os.path.join(D, "exp008_gated_vs_ungated.png")
fig.savefig(p, dpi=150)
print(f"wrote {p}")
