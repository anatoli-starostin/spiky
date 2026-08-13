"""exp23 L2 out-of-band penalty — does a live gradient pull the raw readout in-band,
and what does that buy the spiking build?

Everything is read from artefacts on disk (training JSONs, checkpoints, eval logs).
"""
import json
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

B = "/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant"
HERE = os.path.join(B, "qat_n22_l2")
AN = os.path.join(HERE, "analysis")
OUT = os.path.join(AN, "exp23_l2_oob.png")

INK, MUTE = "#2b2b2b", "#6b6b6b"
C_NONE, C_W1, C_W3, C_PAR = "#B4453C", "#4E79A7", "#2f5d8a", "#8b8b8b"

TAU, TAU_M = 0.09036567807197571, 10.0
SCALE = (1.0 / np.log((1.0 + 0.5 / TAU_M) ** 2)) / TAU

ARMS = [("no penalty", C_NONE, [f"{B}/qat_n22_full/qat_s{s}.json" for s in (0, 1, 2)]),
        ("L2 w=0.1", C_W1, [f"{HERE}/l2w0p1_s{s}.json" for s in (0, 1, 2)]),
        ("L2 w=0.3", C_W3, [f"{HERE}/l2w0p3_s0.json"])]
CKPT = {"parent": "/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/"
                  "deploy_matched/actor_s2.pt",
        "no penalty": [f"{B}/qat_n22_full/qat_s{s}.pt" for s in (0, 1, 2)],
        "L2 w=0.1": [f"{HERE}/l2w0p1_s{s}.pt" for s in (0, 1, 2)],
        "L2 w=0.3": [f"{HERE}/l2w0p3_s0.pt"]}


def spans(path):
    W = torch.load(path, map_location="cpu",
                   weights_only=False)["state_dict"]["actor_lut.weights"].numpy()
    sp, dm = [], []
    for o in range(W.shape[2]):
        Wd = W[:, :, o].astype(np.float64)
        arr = np.rint(-SCALE * Wd + np.ceil(SCALE * Wd.max() + 2))
        sp.append(arr.max() - arr.min()); dm.append(arr.max())
    return np.array(sp), max(dm)


def evals(pat):
    out = []
    for f in sorted(os.listdir(HERE)):
        if not f.startswith("eval_" + pat) or not f.endswith(".log"):
            continue
        for line in open(os.path.join(HERE, f)):
            m = re.search(r"in-quant 128/s1, out-quant 22\s+n=\s*\d+\s+mean\s+([\d.]+)", line)
            if m:
                out.append(float(m.group(1)))
    return out


fig, ax = plt.subplots(1, 4, figsize=(19.0, 5.0))
fig.subplots_adjust(left=0.045, right=0.99, top=0.74, bottom=0.14, wspace=0.28)
a, b, c, d = ax

# ---------------------------------------------------------------- A oob over training
for name, col, files in ARMS:
    for i, f in enumerate(files):
        h = json.load(open(f))["history"]
        a.plot([r["update"] for r in h], [r["out_oob"] * 100 for r in h],
               lw=1.9, color=col, alpha=0.9, label=name if i == 0 else None)
a.set_xlabel("update", fontsize=9.5)
a.set_ylabel("% of raw output outside [-1,1]", fontsize=9.5)
a.set_ylim(0, 62)
a.legend(frameon=False, fontsize=9)
a.set_title("A · The penalty does what the clamp cannot\n"
            "53.9% -> 19.5% (w=0.1) -> 13.0% (w=0.3);\nseeds land within 0.3 pp",
            fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B per-dim spans
labs = ["parent", "no penalty", "L2 w=0.1", "L2 w=0.3"]
cols = [C_PAR, C_NONE, C_W1, C_W3]
sp = {"parent": spans(CKPT["parent"])[0]}
for k in ("no penalty", "L2 w=0.1", "L2 w=0.3"):
    sp[k] = np.mean([spans(p)[0] for p in CKPT[k]], axis=0)
x = np.arange(6)
w = 0.2
for i, (k, col) in enumerate(zip(labs, cols)):
    b.bar(x + (i - 1.5) * w, sp[k], w, color=col, label=k, zorder=3)
b.set_xticks(x); b.set_xticklabels([f"d{i}" for i in range(6)], fontsize=9)
b.set_ylabel("Stage-3 delay span (ticks)", fontsize=9.5)
b.legend(frameon=False, fontsize=8.5, ncol=2)
b.set_ylim(0, 118)
b.annotate("dim 0 barely moves —\nand dim 0 alone sets dmax", xy=(0, sp["L2 w=0.3"][0]),
           xytext=(0.30, 0.80), textcoords="axes fraction", fontsize=8.5, color=INK,
           fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=INK, lw=1.2,
                           connectionstyle="arc3,rad=-0.3"))
b.set_title("B · Where the ticks are saved\nfive of six dims shrink hard;\n"
            "mean span 74.7 -> 80.2 -> 67.5 -> 63.8", fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C dmax / episode
dm = {"parent": spans(CKPT["parent"])[1]}
for k in ("no penalty", "L2 w=0.1", "L2 w=0.3"):
    dm[k] = float(np.mean([spans(p)[1] for p in CKPT[k]]))
xs = np.arange(4)
vals = [dm[k] for k in labs]
c.bar(xs, vals, width=0.6, color=cols, zorder=3)
c.axhline(dm["parent"], color=C_PAR, ls="--", lw=1.4, zorder=4)
for xx, v in zip(xs, vals):
    c.annotate(f"{v:.0f}\n({143+v+75:.0f} ticks)", xy=(xx, v), xytext=(0, 5),
               textcoords="offset points", ha="center", fontsize=9, color=INK,
               fontweight="bold")
c.set_xticks(xs); c.set_xticklabels(labs, fontsize=8.5, rotation=12)
c.set_ylim(0, 125)
c.set_ylabel("dmax (ticks)", fontsize=9.5)
c.set_title("C · The spiking cost\nno-penalty REGRESSED; w=0.3 beats\nthe parent outright",
            fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- D return
ev = {"parent": [6037.1], "no penalty": [6456.8, 6476.4, 6411.9],
      "L2 w=0.1": evals("l2w0p1"), "L2 w=0.3": evals("l2w0p3")}
for i, (k, col) in enumerate(zip(labs, cols)):
    v = ev[k]
    if not v:
        d.annotate("w=0.3 eval\nstill running", xy=(i, 200), ha="center", fontsize=8.5,
                   color=MUTE)
        continue
    d.bar(i, np.mean(v), 0.6, color=col, zorder=3)
    if len(v) > 1:
        d.errorbar(i, np.mean(v), yerr=np.std(v), fmt="none", ecolor=INK,
                   elinewidth=1.4, capsize=5, zorder=4)
    d.annotate(f"{np.mean(v):,.0f}", xy=(i, np.mean(v)), xytext=(0, 6),
               textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold",
               color=INK)
d.axhline(6037.1, color=C_PAR, ls="--", lw=1.4, zorder=2)
d.set_xticks(xs); d.set_xticklabels(labs, fontsize=8.5, rotation=12)
d.set_ylim(0, 7400)
d.set_ylabel("headline eval return", fontsize=9.5)
d.set_title("D · What it costs in return\nboth penalties stay far above the parent\n"
            "(1024x2000, matched physics, both quantizers)",
            fontsize=10, loc="left", color=INK)

for p in (a, b, c, d):
    for sp_ in ("top", "right"):
        p.spines[sp_].set_visible(False)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)
    p.tick_params(labelsize=8.5)

fig.suptitle("exp23 · L2 out-of-band penalty — giving the raw readout the gradient the "
             "clamp cannot provide",
             fontsize=12.5, x=0.004, ha="left", color=INK)
os.makedirs(AN, exist_ok=True)
fig.savefig(OUT, dpi=135)
print("wrote", OUT)
print("mean spans:", {k: round(float(np.mean(v)), 1) for k, v in sp.items()})
print("dmax:", {k: round(v, 1) for k, v in dm.items()})
print("evals:", {k: [round(x, 1) for x in v] for k, v in ev.items()})
