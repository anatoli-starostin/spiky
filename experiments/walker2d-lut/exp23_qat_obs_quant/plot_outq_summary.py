"""exp23 output-QAT smoke summary: return vs N, the grid before/after, and the KL cost.

All numbers are read from the saved smoke artefacts in outq/ (eval logs + run JSONs);
nothing is re-run. The teacher output distribution in panel B is the stored
`y_action_mean_f64`, and the "today" grid is the SHIPPED decode affine, not a nominal step.
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OQ = os.path.join(HERE, "outq")
OUT = os.path.join(HERE, "analysis", "exp23_outq_summary.png")
NPZ = ("/home/astarostin/projects/spiky/landing/walker2d-viz/server/models/"
       "spiking_lut_actor.npz")
DIS = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/data/"
       "distill_exp19_100k.npz")

INK, MUTE = "#2b2b2b", "#6b6b6b"
C_ON, C_OFF, C_WARN, C_GRID = "#4E79A7", "#59A14F", "#B4453C", "#9aa5b1"

# ---------------------------------------------------------------- read the smoke back
NS = [6, 8, 16, 22, 32]
ret = {"on": {}, "off": {}}
for n in NS:
    for line in open(os.path.join(OQ, f"eval_n{n}.log")):
        m = re.search(r"in-quant (\S+).*out-quant (\d+)\s+n=\s*\d+\s+mean\s+([\d.]+)", line)
        if m and int(m.group(2)) == n:
            ret["off" if m.group(1) == "OFF," else "on"][n] = float(m.group(3))
CTL = {"on": 6001.3, "off": 6206.0}          # 6 repeats each, from the eval logs
SD = {"on": 34.0, "off": 16.0}

kl, ep = {}, {}
for arm, key in (("ctl", "ctl"), ("n32", 32), ("n22", 22), ("n16", 16),
                 ("n8", 8), ("n6", 6)):
    h = json.load(open(os.path.join(OQ, f"{arm}.json")))["history"][0]
    kl[key], ep[key] = h["kl"], h["epochs_done"]

fig = plt.figure(figsize=(17.5, 10.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.20,
                      left=0.055, right=0.985, top=0.855, bottom=0.075)
ax = fig.add_subplot(gs[0, :])
bl = fig.add_subplot(gs[1, 0])
cx = fig.add_subplot(gs[1, 1])

# ---------------------------------------------------------------- A return vs N
# categorical x: the N values are 6..32 but what matters is their ORDER, and linear
# spacing crushes 6/8 together exactly where the interesting change happens.
XP = np.arange(len(NS))
for key, col, lab in (("off", C_OFF, "input-quant OFF"),
                      ("on", C_ON, "input-quant ON (128 buckets, sigma 1.0)")):
    ax.axhspan(CTL[key] - SD[key], CTL[key] + SD[key], color=col, alpha=0.16, zorder=1)
    ax.axhline(CTL[key], color=col, ls="--", lw=1.5, alpha=0.75, zorder=2)
    ax.plot(XP, [ret[key][n] for n in NS], "-o", lw=2.2, ms=9, color=col, zorder=4,
            label=lab)
    ax.annotate(f"no output quantization  ({CTL[key]:,.0f})", xy=(4.06, CTL[key]),
                fontsize=8.5, color=col, va="center", ha="left", fontweight="bold",
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5))

for key, col in (("off", C_OFF), ("on", C_ON)):
    for x, n in zip(XP, NS):
        d = ret[key][n] - CTL[key]
        ax.annotate(f"{d:+.0f}", xy=(x, ret[key][n]), xytext=(0, -18 if key == "on" else 12),
                    textcoords="offset points", ha="center", fontsize=8.5, color=col)

ax.annotate("N = 8\nshipped hardware\n(step 0.28)", xy=(1, ret["on"][8]),
            xytext=(1.28, 5822), fontsize=9.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                            connectionstyle="arc3,rad=0.25"))
ax.annotate("N = 22\nneeds ~3x TAU_M_OUT\n(and ~1.6x the episode)", xy=(3, ret["off"][22]),
            xytext=(2.35, 6320), fontsize=9.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xticks(XP)
ax.set_xticklabels([str(n) for n in NS])
ax.set_xlim(-0.35, 5.5)
ax.set_ylim(5780, 6420)
ax.set_xlabel("N — output quantization levels across [-1, 1]", fontsize=10)
ax.set_ylabel("deterministic episode return", fontsize=10)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("A · Return vs output resolution — shaded band is the +-1sd noise floor from "
             "6 repeats of the identical control\n"
             "N = 16, 22 and 32 sit INSIDE noise. Only N = 6 clearly falls below; the "
             "shipped N = 8 costs about one standard deviation.",
             fontsize=10.5, loc="left", color=INK)

# ---------------------------------------------------------------- B the grid
Q = np.load(NPZ)
aff = Q["affine"].astype(np.float64)
v = np.load(DIS)["y_action_mean_f64"][:, 0].astype(np.float64)
LO, HI = -3.8, 3.8
bl.remove()                                    # replaced by a 1x2 sub-grid
gsb = gs[1, 0].subgridspec(1, 2, wspace=0.08)
for panel, (title, lv, note) in enumerate([
        ("today · step 0.283",
         aff[0, 0] * np.arange(-400, 400) + aff[0, 1],
         "7 in-band levels\n~15 ticks wasted on\nvalues that get clipped"),
        ("N = 22 · step 0.095",
         np.linspace(-1, 1, 22),
         "22 in-band levels\n3x finer, by dropping\ntail ordering")]):
    a = fig.add_subplot(gsb[0, panel])
    lv = lv[(lv >= LO) & (lv <= HI)]
    a.axvspan(-1, 1, color="#eef2f7", zorder=0)
    n_, _, _ = a.hist(v, bins=170, range=(LO, HI), color=C_ON, alpha=0.9, zorder=3)
    ym = n_.max()
    for L in lv:
        ins = abs(L) <= 1.0
        a.plot([L, L], [0, ym * 1.02], color=C_GRID, lw=1.5 if ins else 0.7,
               alpha=0.9 if ins else 0.32, zorder=2)
    for s in (-1.0, 1.0):
        a.plot([s, s], [0, ym * 1.10], color=C_WARN, lw=2.0, zorder=5)
    a.annotate(note, xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
               color=MUTE, bbox=dict(facecolor="white", edgecolor="none", alpha=0.92))
    a.set_title(title, fontsize=9.5, loc="left", color=INK)
    a.set_xlim(LO, HI)
    a.set_ylim(0, ym * 1.18)
    a.set_yticks([])
    a.tick_params(labelsize=8.5)
    a.set_xlabel("action value (dim 0)", fontsize=8.5)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    if panel == 0:
        a.set_ylabel("B · Where the output levels go", fontsize=10.5, color=INK)

# ---------------------------------------------------------------- C KL / epochs
keys = ["ctl", 32, 22, 16, 8, 6]
labs = ["control\n(off)", "N=32", "N=22", "N=16", "N=8\n(hardware)", "N=6"]
xs = np.arange(len(keys))
vals = [kl[k] for k in keys]
cols = [C_OFF if ep[k] == 4 else C_WARN for k in keys]
cx.bar(xs, vals, width=0.6, color=cols, zorder=3)
cx.axhline(0.03, color=INK, ls="--", lw=1.6, zorder=4)
cx.annotate("early-stop fires at 1.5 x target-kl = 0.03", xy=(-0.42, 0.0315),
            fontsize=9, color=INK, ha="left", va="bottom", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5))
for x, k, vv in zip(xs, keys, vals):
    cx.annotate(f"{vv:.4f}", xy=(x, vv), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK)
    # above the bar, in ink: the control bar is far too short to hold white text inside it
    cx.annotate(f"{ep[k]}/4 epochs", xy=(x, vv), xytext=(0, 17),
                textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                color=(C_OFF if ep[k] == 4 else C_WARN))
cx.set_xticks(xs)
cx.set_xticklabels(labs, fontsize=9)
cx.set_ylabel("approx_kl at update 1", fontsize=10)
cx.set_ylim(0, 0.058)
cx.set_title("C · The real cost of a coarse grid is the KL early-stop\n"
             "N <= 16 blows past the threshold and completes 1 of 4 epochs per update — "
             "a 384-update run\nwould deliver the gradient budget of roughly 100-150",
             fontsize=10.5, loc="left", color=INK)

for p in (ax, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)
    p.tick_params(labelsize=9)

fig.suptitle("exp23 · output quantization-aware training — 20-update smoke from "
             "deploy_matched seed 2, matched physics, frozen normaliser",
             fontsize=13, x=0.005, ha="left", color=INK)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
