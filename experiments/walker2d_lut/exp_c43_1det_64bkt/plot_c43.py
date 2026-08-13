"""exp_c43 — the width-vs-count axis completed, and the detector-count hypothesis.

Run in the SPIKY venv (matplotlib).

LEFT — the controlled triple. c38, c39 and c43 all have 32 tables, 64 cells per table and
the same 24,576-entry table; only the split of those 64 cells into (detectors x buckets)
differs. Per-seed points, because every configuration in this line is bimodal and a mean
over a bimodal sample hides the thing that matters.

RIGHT — every configuration in the LIF/bucket line plotted against its TOTAL number of LIF
detectors (n_tables x n_det). exp_c36 concluded that return tracks the number of
independent indices SUMMED, meaning tables; c43 completes the other axis and suggests the
unifying quantity is detectors, however they are distributed. Spearman is printed rather
than a fitted line: eight configurations with 3-9 seeds each do not support a regression,
and the claim being made is ordinal.

Usage:
  python plot_c43.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

TRIPLE = [
    ("c43\n1 det × 64 bkt\n27,808", [1554.5, 1375.5, 601.7], RED),
    ("c39\n3 det × 4 bkt\n28,384", [4217.3, 982.3, 890.8], BLUE),
    ("c38\n6 det × 2 bkt\n31,744", [4117.4, 4072.3, 1452.1], GREEN),
]
# (label, total detectors, mean, is-this-run)
SCATTER = [
    ("c33", 32, 1536.2, False), ("c32b", 32, 2041.2, False),
    ("c37", 64, 2531.1, False), ("c39", 96, 2030.2, False),
    ("c42+b", 96, 3043.7, False), ("c36", 128, 4246.1, False),
    ("c38", 192, 3213.9, False),
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
    c43 = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    triple = [(TRIPLE[0][0], c43, RED)] + [(a, b, c) for a, b, c in TRIPLE[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5000)

    # ---- LEFT: the triple ------------------------------------------------
    ax = axes[0]
    for i, (lab, vals, col) in enumerate(triple):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=105, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(34, -4), textcoords="offset points",
                    color=col, fontsize=9.0, fontweight="bold", ha="center")
        k = sum(1 for v in vals if v >= 3000)
        ax.annotate(f"{k}/3", xy=(i, 130), color=col, fontsize=9.0, ha="center",
                    fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([t[0] for t in triple], fontsize=8.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(-0.55, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    ax.set_title("Identical capacity — 32 tables, 64 cells/table, 24,576-entry table.\n"
                 "Only the detector/bucket split differs.",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: detectors vs return --------------------------------------
    ax = axes[1]
    pts = SCATTER + [("c43", 32, r["mean"], True)]
    for lab, nd, mm, mine in pts:
        ax.scatter([nd], [mm], s=125 if mine else 95,
                   color=RED if mine else MUTED, alpha=0.95 if mine else 0.75,
                   zorder=5, edgecolor="white", linewidth=1.6)
        ax.annotate(lab, xy=(nd, mm), xytext=(0, -20 if mine else 11),
                    textcoords="offset points", color=RED if mine else MUTED,
                    fontsize=8.2, ha="center",
                    fontweight="bold" if mine else "normal")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 192])
    ax.set_xticklabels(["32", "64", "128", "192"])
    ax.set_xlim(26, 240)
    ax.set_xlabel("total LIF detectors  (n_tables × n_det)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean 100-ep CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title(f"Spearman ρ = {r['spearman_detectors']:+.2f} against total detectors,\n"
                 f"vs {r['spearman_tables']:+.2f} against table count alone",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c43 — one detector × 64 buckets: the pure-width end of the axis",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c43 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c43)}), takeoff {r['takeoff']}/3 — the "
             f"weakest configuration in the line, at 27,808 params (99.2% of baseline). "
             f"Replicates the old c33 (same 27,808 params, 1536 ± 1417) within noise.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "At fixed capacity, return and takeoff both rise monotonically with the "
             "number of DETECTORS: 1 → 1177 (0/3), 3 → 2030 (1/3), 6 → 3214 (2/3). And "
             "c43 had the BETTER table init, so its deficit is understated.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c43_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
