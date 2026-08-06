"""exp_c37 — the bucket line ordered by TABLE COUNT, the variable that turned out to matter.

Run in the SPIKY venv.

LEFT — per-seed returns for every bucket configuration, ordered by number of tables. The
pattern is monotone in tables and indifferent to buckets: 32 tables (two configurations, 16
and 64 buckets) sit near 1,800; 64 tables reaches 2,531; 128 tables reaches 4,246 and is the
only one on the baseline band.

RIGHT — the two quantities that move with table count: the 3-seed mean and the BEST seed.
Both rise. The best seed going 3,234 -> 3,166 -> 3,992 -> 4,528 says the CEILING itself
lifts, not merely the odds of reaching a fixed one — which is what distinguishes "more
tables makes takeoff more likely" from "more tables makes a better actor". Both appear to
be happening, and three seeds cannot separate them.

Usage:
  python plot_c37.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f9e5a"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"

# (label, dir, tables, buckets, params, colour)
RUNS = [
    ("c32b", "exp_c32b_bucket_fixed", 32, 16, 7840, BLUE),
    ("c33", "exp_c33_bucket64", 32, 64, 27808, BLUE),
    ("c37", "exp_c37_bucket_32x64", 64, 32, 28992, BLUE),
    ("c36", "exp_c36_bucket_tables", 128, 16, 31360, GREEN),
]
BASE_M, BASE_SD = 4308.0, 500.1


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    runs = []
    for lab, d, nt, nb, par, col in RUNS:
        r = json.load(open(os.path.join(D, d, "results.json")))
        pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
        runs.append((lab, nt, nb, par, col, pts, r["mean"], r["sd"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.2, 1]))
    for ax in axes:
        style(ax)

    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.2, zorder=2)
    ax.annotate("exp_c18 hyperplane baseline  4308 ± 500", xy=(-0.42, BASE_M),
                xytext=(0, 8), textcoords="offset points", color=ORANGE,
                fontsize=8.6, ha="left", fontweight="bold")
    for i, (lab, nt, nb, par, col, pts, m, sd) in enumerate(runs):
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.6, zorder=3)
        ax.scatter([i] * len(pts), pts, s=95, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(32, -4),
                    textcoords="offset points", color=col, fontsize=8.8,
                    fontweight="bold", ha="center")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{r[0]}\n{r[1]} tab × {r[2]} bkt\n{r[3]:,} par" for r in runs],
                       fontsize=8.0)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5200)
    ax.set_title("Ordered by TABLE count — bucket count varies within",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    ax = axes[1]
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, alpha=0.9)
    # Left-anchored: the c36 best-seed marker sits at the right end, on the baseline.
    ax.annotate("baseline 4308", xy=(29, BASE_M), xytext=(0, 7),
                textcoords="offset points", color=ORANGE, fontsize=8.4, ha="left")
    xs = [r[1] for r in runs]
    means = [r[6] for r in runs]
    bests = [r[5][0] for r in runs]
    ax.plot(xs, means, "o-", color=BLUE, linewidth=2.0, markersize=9,
            markeredgecolor="white", markeredgewidth=1.6, label="3-seed mean", zorder=4)
    ax.plot(xs, bests, "s--", color=GREEN, linewidth=1.8, markersize=8,
            markeredgecolor="white", markeredgewidth=1.4, label="best seed", zorder=4)
    for x, y, r in zip(xs, means, runs):
        ax.annotate(f"{r[2]} bkt", xy=(x, y), xytext=(0, -17),
                    textcoords="offset points", color=MUTED, fontsize=7.8, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128])
    ax.set_xticklabels(["32", "64", "128"])
    ax.set_xlim(28, 150)
    ax.set_xlabel("tables (log scale)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5200)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="upper left")
    ax.set_title("Both the mean AND the ceiling rise with tables", color=INK,
                 fontsize=11.5, loc="left", pad=10)

    fig.suptitle("exp_c37 — 32 buckets × 64 tables: the middle point of the table-count "
                 "sweep", color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.030,
             "c37 is the closest param-match in the line: 28,992 = 103.4% of the "
             "baseline's 28,032, with the standard 24,576 table. At 32 tables the two "
             "bucket counts (16, 64) give 2,041 and 1,536 — indistinguishable.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "c37 vs baseline −1777 (|t| 2.34); vs c36 (128 tables) −1715 (|t| 2.28); "
             "vs c32b (32 tables) +490 (|t| 0.48). Seeds clearing 3,000: 1/3, 1/3, 1/3, "
             "3/3 as tables go 32, 32, 64, 128.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(D, "exp_c37_bucket_32x64", "c37_result.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
