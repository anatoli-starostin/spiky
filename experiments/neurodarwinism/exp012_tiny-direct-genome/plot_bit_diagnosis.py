"""exp012: where the 0.13 floor on a single sign-comparison bit comes from.

b* = 1[x_norm[0] > x_norm[16]], held-out chance 0.2499.

  A  the localisation ladder, from the true floor up to what the run achieved.
  B  the same as a budget: what each suspect actually costs.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
CH = 0.2499

ROWS = [("two-input rule\n1[x0 > x16]", 0.0000, 100.00, "#b7d3a8", "the true floor"),
        ("encoder TIE floor\nperfect order reader", 0.0273, 95.85, "#8c8c8c", "hard cap"),
        ("matched model\nWITH inhibition", 0.0265, 96.75, "#59A14F", "saturates it"),
        ("matched model\nNO inhibition", 0.0785, 91.70, "#4E79A7", "2.9× the floor"),
        ("best decode of the\nrun's own output tick", 0.1143, 85.70, "#E1A03C", ""),
        ("the RUN\n(diagls readout)", 0.1283, 85.70, C_T, "")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.2, 6.0))
fig.subplots_adjust(left=0.055, right=0.985, top=0.745, bottom=0.185, wspace=0.24)

# ---------------------------------------------------------------- A the ladder
xs = np.arange(len(ROWS))
ax.bar(xs, [r[1] for r in ROWS], 0.62, color=[r[3] for r in ROWS], zorder=3)
for x, (nm, v, acc, c, note) in zip(xs, ROWS):
    ax.text(x, v + 0.006, f"{v:.4f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(x, v + 0.018, f"{acc:.1f}%", ha="center", fontsize=8.5, color=MUTE)
ax.axhline(CH, color=C_T, ls="--", lw=1.9, zorder=5)
ax.text(len(ROWS) - 0.45, CH + 0.006, f"chance {CH:.4f}  (50.8% acc)", fontsize=9.5,
        color=C_T, fontweight="bold", ha="right")
ax.axhline(0.0273, color="#8c8c8c", ls=":", lw=1.6, zorder=4)
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in ROWS], fontsize=8.4, rotation=18, ha="right")
ax.set_ylim(0, 0.285)
ax.set_ylabel("held-out MSE   (accuracy above each bar)", fontsize=10)
ax.set_title("A · A single comparison bit, localised\n"
             "with inhibition the matched model reaches 0.0265 — the encoder's own\n"
             "tie floor (0.0273). Without it, 0.0785. The run got 0.1283",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B the budget
STEPS = [("encoder ties\n(32 ticks, unavoidable)", 0.0273, "#8c8c8c"),
         ("NO inhibition", 0.0785 - 0.0273, "#4E79A7"),
         ("search / wiring\n(x16 never reaches the output)", 0.1143 - 0.0785, "#E1A03C"),
         ("readout\nscale+shift", 0.1283 - 0.1143, C_T)]
bottom = 0.0
mids = []
for nm, d, c in STEPS:
    bx.bar(0, d, 0.5, bottom=bottom, color=c, zorder=3)
    mids.append(bottom + d / 2)
    bottom += d
# the readout band is only 0.014 tall, so labels anchored at band midpoints collide. Spread
# them evenly and draw a connector to each band instead.
label_y = np.linspace(0.035, 0.20, len(STEPS))[::-1]
for (nm, d, c), my, ly in zip(STEPS, mids, label_y):
    bx.plot([0.26, 0.40], [my, ly], color=c, lw=1.2, zorder=4)
    bx.text(0.44, ly, f"{nm}\n{d:+.4f}   ({100 * d / 0.1283:.0f}% of the floor)",
            va="center", fontsize=9.2, color=INK)
bx.axhline(CH, color=C_T, ls="--", lw=1.8, zorder=5)
bx.text(-0.32, CH + 0.005, f"chance {CH:.4f}", fontsize=9.5, color=C_T, fontweight="bold")
bx.text(0, -0.012, "the run's 0.1283", ha="center", fontsize=10, color=INK,
        fontweight="bold")
bx.set_xlim(-0.38, 1.35)
bx.set_ylim(0, 0.285)
bx.set_xticks([])
bx.set_ylabel("held-out MSE", fontsize=10)
bx.set_title("B · What each suspect costs\n"
             "the readout is nearly free and the encoder floor is small; the damage is\n"
             "no-inhibition plus a concrete wiring failure",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)

fig.suptitle("exp012 — the 0.13 floor on a single sign-comparison bit is NOT the readout: "
             "it is no-inhibition (0.051) plus a wiring failure (0.036)",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_bit_diagnosis.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
