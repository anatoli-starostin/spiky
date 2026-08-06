"""exp_c46 — the shared-ladder penalty grows with table count, and it breaks the
detector-count reading.

Run in the SPIKY venv (matplotlib).

LEFT — the penalty at two scales. At 64 detectors, c44 (per-table ladders) vs c45 (shared)
costs 1,163. At 128 detectors, c36 (per-table) vs c46 (shared) costs 3,298. The penalty
does not saturate; it GROWS with the number of tables forced onto one ladder, which is the
opposite of what "more detectors recovers it" would predict.

RIGHT — return against total LIF detectors, with the two shared-ladder configurations
marked. Within the per-table family the ordering is clean (Spearman +0.812 over 9); adding
the two shared points drops it to +0.437, because both sit far below their detector count.
The per-table ladder is a PRECONDITION for the detector-count reading, not a minor term in
it.

Usage:
  python plot_c46.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, RED, MUTED2 = "#2a78d6", "#eb6834", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

# (label, detectors, mean, per-table ladder?)
RUNS = [
    ("c43", 32, 1177.2, True), ("c33", 32, 1536.2, True), ("c32b", 32, 2041.2, True),
    ("c45", 64, 1198.0, False), ("c44", 64, 2360.9, True), ("c37", 64, 2531.1, True),
    ("c39", 96, 2030.2, True), ("c42+b", 96, 3043.7, True),
    ("c36", 128, 4246.1, True), ("c38", 192, 3213.9, True),
]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    r = json.load(open(os.path.join(HERE, "results.json")))
    m46 = r["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5000)

    # ---- LEFT: the penalty at two scales ---------------------------------
    ax = axes[0]
    pairs = [("64 detectors\nc44 → c45", 2360.9, 1198.0),
             ("128 detectors\nc36 → c46", 4246.1, m46)]
    for i, (lab, per, sh) in enumerate(pairs):
        ax.plot([i - 0.16, i + 0.16], [per, sh], color=MUTED2, linewidth=1.8, zorder=3)
        ax.scatter([i - 0.16], [per], s=140, color=BLUE, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.scatter([i + 0.16], [sh], s=140, color=RED, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.annotate(f"{per:.0f}", xy=(i - 0.16, per), xytext=(0, 12),
                    textcoords="offset points", color=BLUE, fontsize=9.0,
                    fontweight="bold", ha="center")
        ax.annotate(f"{sh:.0f}", xy=(i + 0.16, sh), xytext=(0, -20),
                    textcoords="offset points", color=RED, fontsize=9.0,
                    fontweight="bold", ha="center")
        ax.annotate(f"−{per - sh:.0f}\n(−{100*(per-sh)/per:.0f}%)",
                    xy=(i, (per + sh) / 2), xytext=(30, 0),
                    textcoords="offset points", color=INK, fontsize=9.2,
                    fontweight="bold", ha="left", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([p[0] for p in pairs], fontsize=9.0)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    h1, = ax.plot([], [], "o", color=BLUE, label="per-table ladders")
    h2, = ax.plot([], [], "o", color=RED, label="ONE shared ladder")
    ax.legend(handles=[h1, h2], frameon=False, fontsize=8.6, labelcolor=INK,
              loc="lower left")
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(-0.45, BASE_M + BASE_SD),
                xytext=(0, 6), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", fontweight="bold")
    ax.set_title("The shared-ladder penalty GROWS with table count",
                 color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: detector count, family-split -----------------------------
    ax = axes[1]
    for lab, nd, mm, per in RUNS:
        ax.scatter([nd], [mm], s=100 if per else 130,
                   color=MUTED2 if per else RED, alpha=0.8 if per else 0.95, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(lab, xy=(nd, mm), xytext=(0, 11 if per else -20),
                    textcoords="offset points", color=MUTED if per else RED,
                    fontsize=8.2, ha="center",
                    fontweight="normal" if per else "bold")
    ax.scatter([128], [m46], s=150, color=RED, alpha=0.95, zorder=6,
               edgecolor="white", linewidth=1.8)
    ax.annotate("c46\nshared", xy=(128, m46), xytext=(0, -30),
                textcoords="offset points", color=RED, fontsize=8.6, ha="center",
                fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 192])
    ax.set_xticklabels(["32", "64", "128", "192"])
    ax.set_xlim(26, 240)
    ax.set_xlabel("total LIF detectors  (n_tables × n_det)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean 100-ep CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("ρ = +0.81 within the per-table family (9 configs);\n"
                 "+0.44 once the two shared-ladder points are added",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c46 — 128 tables × 1 detector × 16 buckets with a shared ladder",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c46 {m46:.0f} ± {r['sd']:.0f} (1578, 681, 586), takeoff 0/3, 29,328 params. "
             f"(a) vs c45 — same handicap, half the detectors: −250, |t| 0.40, so doubling "
             f"detectors recovered NOTHING. (b) vs c36 — same 128 detectors, per-table "
             f"ladders: −3298, |t| 9.17.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "So the per-table bucket ladder is a PRECONDITION for the detector-count "
             "reading, not a minor term in it: sharing it costs 49% at 64 detectors and "
             "78% at 128, and destroys the only configuration that ever reached the "
             "baseline.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c46_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
