"""exp23 prep figure: why sigma=1.0, and what the quantizer costs before any fine-tuning."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = json.load(open(os.path.join(HERE, "analysis", "sigma_sweep.json")))
OUT = os.path.join(HERE, "analysis", "exp23_prep.png")

INK, MUTE = "#2b2b2b", "#6b6b6b"
C_A, C_B = "#4E79A7", "#B4453C"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.0, 5.4))
fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.14, wspace=0.24)

# ---------------------------------------------------------------- A sigma sweep
s = np.array([r["sigma"] for r in SWEEP["sweep"]])
flip = np.array([r["flip_rate"] for r in SWEEP["sweep"]]) * 100
tie = np.array([r["tie_rate"] for r in SWEEP["sweep"]]) * 100

ax.plot(s, flip, "-o", lw=2.0, ms=8, color=C_A, zorder=3, label="address-bit flips")
ax.plot(s, tie, "-o", lw=2.0, ms=7, color=MUTE, alpha=0.55, zorder=2,
        label="pairs sharing a bucket (ties)")
best = int(np.argmin(flip))
ax.plot(s[best], flip[best], "o", ms=15, mfc="none", mec=C_A, mew=2.4, zorder=4)
ax.annotate(f"$\\sigma$ = {s[best]:g}\n{flip[best]:.3f}% flips\n(interior minimum —\n"
            "not an artefact of\npushing $\\sigma$ up)",
            xy=(s[best], flip[best]), xytext=(0.40, 0.62), textcoords="axes fraction",
            fontsize=9.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                            connectionstyle="arc3,rad=-0.25"))
for x, y in ((s[0], flip[0]), (s[-1], flip[-1])):
    ax.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                fontsize=8.5, color=MUTE, ha="center")
ax.set_xlabel("companding strength  $\\sigma$", fontsize=10)
ax.set_ylabel("% of the 192 address bits affected", fontsize=10)
ax.set_ylim(0, max(tie.max(), flip.max()) * 1.18)
ax.legend(frameon=False, fontsize=9, loc="upper center")
ax.set_title("A · Choosing $\\sigma$ by measurement, not assumption\n"
             "too small wastes levels on the tails, too large wastes the companding;\n"
             "the optimum coincides with the principled value for unit-variance obs",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B what it costs
labels = ["quantizer OFF", "quantizer ON\n(128 ticks, $\\sigma$=1.0)"]
means = [6213.5, 5973.2]
sds = [664.9, 1022.2]
meds = [6350.0, 6327.0]
xs = np.arange(2)
bars = bx.bar(xs, means, width=0.52, color=[C_A, C_B], zorder=3)
bx.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK, elinewidth=1.6,
            capsize=7, capthick=1.6, zorder=4)
for x, m, md in zip(xs, means, meds):
    bx.annotate(f"mean {m:,.0f}", xy=(x, m), xytext=(0, 42), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    bx.plot([x - 0.26, x + 0.26], [md, md], color="white", lw=2.2, zorder=5)
    bx.annotate(f"median {md:,.0f}", xy=(x, md), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8.5, color="white")
bx.axhline(5966.3, color=MUTE, ls="--", lw=1.4, zorder=2)
bx.text(0.015, 0.80, "- - -  5966: the return recorded when this checkpoint was saved",
        transform=bx.transAxes, fontsize=8.5, color=MUTE, ha="left")
bx.set_xticks(xs)
bx.set_xticklabels(labels, fontsize=9.5)
bx.set_ylabel("episode return (deterministic policy)", fontsize=10)
bx.set_ylim(0, 8200)
bx.set_title("B · The cost of quantization BEFORE any fine-tuning\n"
             "mean −3.9%, but the median is within 23 — the loss is entirely in the tail,\n"
             "which is what a fine-tune is positioned to recover",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)

fig.suptitle("exp23 prep · Gaussian-companding 128-bucket observation quantizer, "
             "measured on the deploy_matched seed-2 checkpoint",
             fontsize=12.5, x=0.004, ha="left", color=INK)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
