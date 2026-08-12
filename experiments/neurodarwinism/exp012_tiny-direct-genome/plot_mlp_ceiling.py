"""exp012: is the spiking net's held-out MSE a substrate/search failure or the task's ceiling?

A conventional MLP of the SAME 17-8-1 shape, on the SAME data, answers it -- and adding the
spiking net's constraints one at a time says where the error actually comes from.

  A  the ladder for target dim 0, from a free MLP up to the spiking net.
  B  the same decomposition as a waterfall: quantisation is free; Dale's law and the latency
     encoder are what cost, and the spiking search is only ~2.4 short of its own ceiling.
  C  target dim 5, the same story at a different scale.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(D, "analysis")
R0 = json.load(open(os.path.join(A, "mlp_ceiling_t0.json")))
R5 = json.load(open(os.path.join(A, "mlp_ceiling_t5.json")))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
C_FREE, C_CON, C_SNN = "#59A14F", "#4E79A7", "#E1A03C"

SPIKING = {0: 22.783, 5: None}          # dim 5 still running at report time


def ladder(R, spiking):
    q = R["qat"]
    rows = [("128-wide MLP", R["mlp_17_128_1_affine"], "#b7d3a8"),
            ("free MLP 17→8→1", R["mlp_17_8_1"], C_FREE),
            ("+ 0.1 grid (QAT)", q["grid_x_gain"]["after_snap_affine"], C_FREE),
            ("+ Dale", q["dale_grid_x_gain"]["after_snap_affine"], C_CON),
            ("+ encoding = ceiling",
             q["dale_grid_x_gain_encoded"]["after_snap_affine"], C_CON),
            ("linear 17→1 (LS)", R["linear_17_1"], "#c9c9c9")]
    if spiking is not None:
        rows.insert(5, ("the SPIKING net", spiking, C_SNN))
    return rows


fig = plt.figure(figsize=(16.6, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], left=0.045, right=0.99,
                      top=0.735, bottom=0.175, wspace=0.26)
ax, bx, cx = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))

# ---------------------------------------------------------------- A the ladder, dim 0
rows = ladder(R0, SPIKING[0])
xs = np.arange(len(rows))
ax.bar(xs, [r[1] for r in rows], 0.62, color=[r[2] for r in rows], zorder=3)
for x, (nm, v, c) in zip(xs, rows):
    ax.text(x, v + 0.55, f"{v:.2f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.axhline(R0["chance"], color=C_T, ls="--", lw=1.9, zorder=5)
ax.text(len(rows) - 0.45, R0["chance"] + 0.6, f"own chance {R0['chance']:.1f}", fontsize=9.5,
        color=C_T, fontweight="bold", ha="right")
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in rows], fontsize=8.6, rotation=20, ha="right")
ax.set_ylim(0, 33)
ax.set_ylabel("held-out MSE", fontsize=10)
ax.set_title("A · Target dim 0 — the spiking net is near its OWN ceiling\n"
             "the same 17-8-1 shape and the same data, with the spiking net's constraints\n"
             "added one at a time. Quantisation is free; the constraints are what cost",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B the waterfall
q = R0["qat"]
steps = [("free MLP\n17→8→1", R0["mlp_17_8_1"], None),
         ("0.1 grid\n(QAT)", q["grid_x_gain"]["after_snap_affine"], "quantisation"),
         ("Dale\nno inhibition", q["dale_grid_x_gain"]["after_snap_affine"], "Dale"),
         ("latency\nencoding", q["dale_grid_x_gain_encoded"]["after_snap_affine"], "encoder"),
         ("the spiking\nnet", SPIKING[0], "search + spikes")]
prev = steps[0][1]
bx.bar(0, prev, 0.6, color=C_FREE, zorder=3)
bx.text(0, prev / 2, f"{prev:.2f}", ha="center", va="center", fontsize=10, color="white",
        fontweight="bold")
for i, (nm, v, lab) in enumerate(steps[1:], start=1):
    d = v - prev
    col = C_CON if i < 4 else C_SNN
    bx.bar(i, d, 0.6, bottom=prev, color=col, zorder=3)
    bx.plot([i - 0.5, i + 0.5], [prev, prev], color="0.55", lw=0.9, ls=":", zorder=4)
    bx.text(i, max(v, prev) + 0.5, f"{d:+.2f}", ha="center", fontsize=9.5, color=INK,
            fontweight="bold")
    prev = v
bx.axhline(R0["chance"], color=C_T, ls="--", lw=1.9, zorder=5)
bx.text(4.45, R0["chance"] + 0.55, f"own chance {R0['chance']:.1f}", fontsize=9.5, color=C_T,
        fontweight="bold", ha="right")
bx.set_xticks(range(len(steps)))
bx.set_xticklabels([s[0] for s in steps], fontsize=8.6)
bx.set_ylim(0, 33)
bx.set_ylabel("held-out MSE", fontsize=10)
bx.set_title("B · Where the error actually comes from\n"
             "the 0.1 weight grid costs NOTHING once training is grid-aware. Removing\n"
             "inhibition costs 5.9 and the latency code 3.8 — both are design choices",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C dim 5
q5 = R5["qat"]
rows5 = [("free MLP", R5["mlp_17_8_1"], C_FREE),
         ("+0.1 grid\nQAT", q5["grid_x_gain"]["after_snap_affine"], C_FREE),
         ("+Dale", q5["dale_grid_x_gain"]["after_snap_affine"], C_CON),
         ("+encoding\n= ceiling", q5["dale_grid_x_gain_encoded"]["after_snap_affine"], C_CON)]
xs = np.arange(len(rows5))
cx.bar(xs, [r[1] for r in rows5], 0.6, color=[r[2] for r in rows5], zorder=3)
for x, (nm, v, c) in zip(xs, rows5):
    cx.text(x, v + 0.12, f"{v:.2f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
cx.set_xticks(xs)
cx.set_xticklabels([r[0] for r in rows5], fontsize=8.4)
cx.set_ylim(0, 8.6)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.text(0.03, 0.965, f"own chance {R5['chance']:.1f} — off the top of this axis",
        transform=cx.transAxes, fontsize=8.8, color=C_T, fontweight="bold", va="top")
cx.set_title("C · Target dim 5 — same shape, 8× lower\n"
             "an easier dimension: matched ceiling 5.67 vs\n"
             "chance 47.4; the spiking run was at ~9.0",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)

fig.suptitle("exp012 — an MLP ceiling for the single-output task: the spiking net is close to "
             "its own matched ceiling; the losses are Dale's law and the latency code",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_mlp_ceiling.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
