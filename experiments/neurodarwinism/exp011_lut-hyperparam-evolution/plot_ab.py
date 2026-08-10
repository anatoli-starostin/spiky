"""exp011 A/B: warm-start (Lamarckian weights) vs cold-start, on cumulative backprop steps.

Warm start's promise is sample-efficiency, so the axis is total backprop steps, not rounds and
not final fitness alone.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(D, "ab_warm_vs_cold")
SEEDS, ARMS = (0, 1), ("cold", "warm")
INK, MUTE = "#2b2b2b", "#6b6b6b"
C = {"cold": "#4E79A7", "warm": "#E1575A"}
LS = {0: "-", 1: "--"}

H = {(a, s): json.load(open(os.path.join(AB, f"{a}_seed{s}", f"lut_evolve_{a}_s{s}.json")))
     for s in SEEDS for a in ARMS}
F = {(a, s): json.load(open(os.path.join(AB, f"{a}_seed{s}",
                                         f"lut_evolve_{a}_s{s}_final.json")))
     for s in SEEDS for a in ARMS}
TEACH = F[("cold", 0)]["teacher"]

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(15.2, 4.7), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.9]),
    layout="constrained")

# ---------------------------------------------------------------- A: fitness vs steps
for s in SEEDS:
    for a in ARMS:
        h = H[(a, s)]
        ax.plot([r["total_backprop_steps"] / 1e6 for r in h], [r["best"] for r in h],
                LS[s], color=C[a], lw=2.0, label=f"{a}, seed {s}", zorder=3)
ax.set_xlabel("cumulative backprop steps (millions)", fontsize=10)
ax.set_ylabel("best fitness (EWMA)", fontsize=10)
ax.set_title("A · Fitness against training spent\n"
             "warm ends ahead on both seeds — but only just, and on\n"
             "seed 1 it TRAILS cold for two thirds of the run",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")

# ---------------------------------------------------------------- B: minimal LUT vs steps
tol = 0.002
for s in SEEDS:
    for a in ARMS:
        f, h = F[(a, s)], H[(a, s)]
        best = {}
        for x in f["seen"]:
            if x["mse"] <= TEACH["mse"] + tol:
                best[x["rnd"]] = min(best.get(x["rnd"], 10 ** 9), x["params"])
        run, cur, xs, ys = [], None, [], []
        for r, rec in enumerate(h):
            if r in best:
                cur = best[r] if cur is None else min(cur, best[r])
            if cur:
                xs.append(rec["total_backprop_steps"] / 1e6)
                ys.append(cur)
        bx.plot(xs, ys, LS[s], color=C[a], lw=2.0, label=f"{a}, seed {s}", zorder=3)
bx.axhline(TEACH["params"], color="#B4453C", ls=":", lw=1.5, zorder=2)
bx.text(0.05, TEACH["params"] * 1.04, f"teacher {TEACH['params']:,} params",
        fontsize=8.5, color="#B4453C")
bx.set_yscale("log")
bx.set_xlabel("cumulative backprop steps (millions)", fontsize=10)
bx.set_ylabel("smallest LUT matching the teacher's fit", fontsize=10)
bx.set_title("B · Minimal teacher-matching LUT\n"
             "both arms get well below the teacher's parameter count;\n"
             "cold ends marginally smaller on both seeds",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- C: steps to threshold
ths = [-0.0210, -0.0200, -0.0195, -0.0190]
w = 0.35
for i, th in enumerate(ths):
    for j, s in enumerate(SEEDS):
        vals = {}
        for a in ARMS:
            vals[a] = next((r["total_backprop_steps"] / 1e6
                            for r in H[(a, s)] if r["best"] >= th), np.nan)
        for k, a in enumerate(ARMS):
            cx.bar(i + (j - 0.5) * 0.44 + (k - 0.5) * 0.2, vals[a], width=0.19,
                   color=C[a], alpha=1.0 if s == 0 else 0.55, zorder=3)
cx.set_xticks(range(len(ths)))
cx.set_xticklabels([f"{t:+.4f}" for t in ths], fontsize=8.5)
cx.set_xlabel("fitness threshold", fontsize=10)
cx.set_ylabel("steps to reach it (millions)", fontsize=10)
cx.set_title("C · Steps to a given fitness\n"
             "no early advantage; a late one on seed 0 only\n"
             "solid = seed 0, faded = seed 1; no bar = never reached",
             fontsize=9.5, loc="left", color=INK)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp011 A/B — warm-start vs cold-start · K=16, 30 rounds, 2000 steps/candidate, "
             "2 paired seeds, 0.96M backprop steps each",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp011_ab_warm_vs_cold.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
