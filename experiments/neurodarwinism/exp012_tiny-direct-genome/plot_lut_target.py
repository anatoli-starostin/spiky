"""exp012: does reshaping the target into a 32-entry first-spike LUT lower the ceiling?

The two targets live in different units -- offsets 0..32 versus raw action -- so their MSEs
are not comparable and only the RATIO to each target's own chance is. That is the whole
point of panel A: same ladder, both targets, on the one axis where they can be compared.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, "analysis", "lut_target_t0.json")))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
C_OLD, C_NEW = "#9db8d1", "#59A14F"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.6, 5.9))
fig.subplots_adjust(left=0.06, right=0.985, top=0.755, bottom=0.145, wspace=0.24)

# ---------------------------------------------------------------- A the two ladders
KEYS = [("free MLP\n17→8→1", "free_mlp"),
        ("+ Dale\n+ 0.1 grid (QAT)", "dale_grid"),
        ("+ latency input\n= MATCHED ceiling", "matched_ceiling")]
xs = np.arange(len(KEYS))
w = 0.36
old = [R["ratio_old"][k] for _, k in KEYS]
new = [R["ratio_new"][k] for _, k in KEYS]
ax.bar(xs - w / 2, old, w, color=C_OLD, zorder=3,
       label=f"OLD offset target  (own chance {R['chance_old_offset_target']:.1f})")
ax.bar(xs + w / 2, new, w, color=C_NEW, zorder=3,
       label=f"NEW 32-level LUT target  (own chance {R['chance_new_target']:.3f})")
for x, (o, n) in enumerate(zip(old, new)):
    ax.text(x - w / 2, o + 0.015, f"{o:.3f}", ha="center", fontsize=9.5, color=INK)
    ax.text(x + w / 2, n + 0.015, f"{n:.3f}", ha="center", fontsize=9.5, color=INK,
            fontweight="bold")
ax.axhline(1.0, color=C_T, ls="--", lw=1.9, zorder=5)
ax.text(len(KEYS) - 0.45, 1.02, "own chance", fontsize=9.5, color=C_T, fontweight="bold",
        ha="right")
ax.annotate("", xy=(2 + w / 2, new[2] + 0.02), xytext=(2 - w / 2, old[2] + 0.02),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
ax.text(0.0, 0.93, f"the MATCHED ceiling improves {old[2]:.3f} → {new[2]:.3f}\n"
                   f"— {100 * (old[2] - new[2]) / old[2]:.0f}% less unexplained variance",
        fontsize=9.2, color=INK, fontweight="bold", va="top", ha="left")
ax.set_xticks(xs)
ax.set_xticklabels([k[0] for k in KEYS], fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_ylabel("held-out MSE ÷ that target's own chance", fontsize=10)
ax.legend(frameon=False, fontsize=8.8, loc="upper left")
ax.set_title("A · The LUT target IS easier — but only compare ratios\n"
             "the two targets are in different units (offsets 0..32 vs raw action), so\n"
             "their MSEs are not comparable. The fraction of variance left is",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B the LUT itself
lut = np.array(R["lut"])
cnt = np.array(R["bin_counts_val"])
b = np.arange(len(lut))
bx.bar(b, lut, 0.78, color=C_NEW, zorder=3)
bx.set_xlabel("output first-spike tick  →  LUT bin", fontsize=10)
bx.set_ylabel("decoded value (mean of the bin)", fontsize=10)
bx.set_title("B · The 32-entry decode table\n"
             "equal-population bins, so every tick is used. Binning discards only\n"
             f"{100 * R['within_bin_var'] / R['var_continuous']:.2f}% of the continuous "
             "target's variance",
             fontsize=10, loc="left", color=INK)
bx.text(0.03, 0.95, f"every one of the 32 levels occupied on held-out "
                    f"(min {cnt.min()} samples, max {cnt.max()})",
        transform=bx.transAxes, fontsize=8.8, color=MUTE, ha="left")

for p in (ax, bx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)
bx.spines["right"].set_visible(False)

fig.suptitle("exp012 — reshaping the target into a first-spike LUT: the matched ceiling "
             "improves from 0.692 to 0.567 of chance",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_lut_target.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
