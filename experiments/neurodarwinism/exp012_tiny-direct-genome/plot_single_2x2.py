"""exp012: the single-output capacity test and the asymmetric-grid 2x2.

  A  every arm against ITS OWN chance. Raw MSEs are not comparable across target dimensions
     -- dim 5's constant predictor scores 47.4 and dim 1's scores 24.5 -- so the bar is the
     RATIO, and 1.0 is chance.
  B  what the error is made of. bias^2 / scale mismatch / the residual no affine can remove.
  C  the 2x2 itself: {-1,0,1} vs the 11-exc/2-inh grid, on targets 0 and 1.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, "analysis", "single_2x2.json")))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
C_SIX, C_SYM, C_ASYM = "#9db8d1", "#4E79A7", "#59A14F"

DIMS = sorted(int(d) for d in R["single"])
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.6, 5.9))
fig.subplots_adjust(left=0.05, right=0.99, top=0.755, bottom=0.13, wspace=0.26)

# ---------------------------------------------------------------- A ratio to own chance
w = 0.36
xs = np.arange(len(DIMS))
six = [R["six"][str(d)]["ratio"] for d in DIMS]
sng = [R["single"][str(d)]["ratio"] for d in DIMS]
ax.bar(xs - w / 2, six, w, color=C_SIX, zorder=3, label="that dim inside the 6-way run")
ax.bar(xs + w / 2, sng, w, color=C_SYM, zorder=3, label="the whole net on that dim alone")
ax.axhline(1.0, color=C_T, ls="--", lw=1.9, zorder=5)
ax.text(len(DIMS) - 0.45, 1.03, "its own chance", fontsize=9.5, color=C_T, fontweight="bold",
        ha="right")
for x, d in zip(xs, DIMS):
    ax.text(x - w / 2, six[x] + 0.03, f"{six[x]:.2f}", ha="center", fontsize=9, color=INK)
    ax.text(x + w / 2, sng[x] + 0.03, f"{sng[x]:.2f}", ha="center", fontsize=9.5, color=INK,
            fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"target {d}\nown chance "
                    f"{R['chance_per_dim'][str(d)]:.1f}" for d in DIMS], fontsize=9)
ax.set_ylabel("held-out MSE ÷ that dimension's own chance", fontsize=10)
ax.set_ylim(0, max(max(six), max(sng)) * 1.28)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("A · Isolating one dimension helps — a lot, and unevenly\n"
             "every arm judged against the constant predictor OF ITS OWN dimension.\n"
             "The 6-dim 34.15 is not the yardstick for any of these",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B error decomposition
xs = np.arange(len(DIMS))
res = [R["single"][str(d)]["residual"] for d in DIMS]
se = [R["single"][str(d)]["scale_err"] for d in DIMS]
b2 = [R["single"][str(d)]["bias2"] for d in DIMS]
bx.bar(xs, res, 0.55, color="#9db8d1", label="residual (no affine can remove it)", zorder=3)
bx.bar(xs, se, 0.55, bottom=res, color=C_SYM, label="scale mismatch", zorder=3)
bx.bar(xs, b2, 0.55, bottom=np.array(res) + np.array(se), color=C_T, label="bias²", zorder=3)
for x, d in zip(xs, DIMS):
    ch = R["chance_per_dim"][str(d)]
    bx.plot([x - 0.36, x + 0.36], [ch, ch], color=C_T, ls="--", lw=1.8, zorder=6)
    tot = res[x] + se[x] + b2[x]
    bx.text(x, tot + 0.9, f"r={R['single'][str(d)]['r']:.2f}", ha="center", fontsize=9.5,
            color=INK, fontweight="bold")
bx.text(xs[-1] + 0.4, R["chance_per_dim"][str(DIMS[-1])], "own\nchance", fontsize=8.5,
        color=C_T, va="center", ha="left", fontweight="bold")
bx.set_xticks(xs)
bx.set_xticklabels([f"target {d}" for d in DIMS], fontsize=9)
bx.set_ylabel("held-out MSE", fontsize=10)
bx.legend(frameon=False, fontsize=8.5, loc="upper left")
bx.set_title("B · Where the remaining error lives\n"
             "single-output arms. Bias and scale are what a readout could fix; the\n"
             "residual is what the network genuinely fails to represent",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C the 2x2
GRIDS = [("sym", "{−1, 0, +1}", C_SYM), ("asym", "11 exc / 2 inh", C_ASYM)]
TD = [d for d in (0, 1) if str(d) in R["single"]]
xs = np.arange(len(TD))
w = 0.36
for k, (key, lab, col) in enumerate(GRIDS):
    src = R["single"] if key == "sym" else R.get("asym", {})
    v = [src.get(str(d), {}).get("ratio", np.nan) for d in TD]
    bars = cx.bar(xs + (k - 0.5) * w, v, w, color=col, zorder=3, label=lab)
    for x, val in zip(xs, v):
        if np.isfinite(val):
            cx.text(x + (k - 0.5) * w, val + 0.02, f"{val:.2f}", ha="center", fontsize=9.5,
                    color=INK, fontweight="bold")
cx.axhline(1.0, color=C_T, ls="--", lw=1.9, zorder=5)
cx.text(len(TD) - 0.45, 1.02, "its own chance", fontsize=9.5, color=C_T, fontweight="bold",
        ha="right")
cx.set_xticks(xs)
cx.set_xticklabels([f"target {d}\nown chance {R['chance_per_dim'][str(d)]:.1f}" for d in TD],
                   fontsize=9)
cx.set_ylabel("held-out MSE ÷ own chance", fontsize=10)
cx.legend(frameon=False, fontsize=9, loc="upper left", title="weight grid",
          title_fontsize=8.5)
cx.set_title("C · The 2×2 — does a finer excitatory grid pay?\n"
             "same single-output task, same seed; only the weight grid differs.\n"
             "The dissection said the weights carry the solution — this tests it",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — the single-output capacity test: what the substrate can do when the "
             "whole network is aimed at one number",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_single_2x2.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
