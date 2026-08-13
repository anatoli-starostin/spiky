"""The bucket-LIF line: three levers moved the addressing statistics, none moved return.

Run in the SPIKY venv. Reads the four bucket runs' results.json plus the spike-distribution
analyses, and answers one question: does anything about the bucket ADDRESSING predict the
return?

LEFT — per-seed returns. Deliberately points, not bars. Every configuration produces the
same shape: exactly one seed reaches ~3,200 and the rest stall under 1,100. The best seed
is 3234 / 3166 / 3245 across three very different configurations, a spread of 79 points.
Means and error bars hide that; the points make it obvious that what varies between runs is
HOW MANY seeds take off, not how high they get.

RIGHT — the addressing statistic each experiment was designed to move, against the return
it produced. The levers worked: effective buckets went 4.5 -> 8.8 with 4x the capacity, and
quantile init started at 7.8. Return did not follow. The exp_c34 arrow is the finding of
the whole line -- it STARTS at 7.8 effective buckets and TRAINS DOWN to 3.1, so the
collapse is an attractor of the dynamics rather than a bad initialisation.

Usage:
  python plot_bucket_line.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#b03030"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"

# (label, dir, colour, per-seed, mean, sd, effective buckets final, n_buckets, init eff)
RUNS = [
    ("c32 BROKEN\nfree-signed w", None, RED,
     [1739.9, 725.5, 1057.7], 1174.4, 517.2, None, 16, None),
    ("c32b\n16 uniform", "exp_c32b_bucket_fixed", BLUE,
     None, None, None, 4.5, 16, None),
    ("c33\n64 uniform", "exp_c33_bucket64", BLUE,
     None, None, None, 8.8, 64, None),
    ("c34\n16 quantile", "exp_c34_quantile_boundaries", BLUE,
     None, None, None, 3.1, 16, 7.8),
    ("c35\nquantile FROZEN", "exp_c35_frozen_boundaries", "#7a4fbf",
     None, None, None, 7.8, 16, None),
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
    for lab, d, col, pts, m, sd, eff, nb, eff0 in RUNS:
        if d is not None:
            r = json.load(open(os.path.join(D, d, "results.json")))
            pts = [r["seeds"][k] for k in sorted(r["seeds"])]
            m, sd = r["mean"], r["sd"]
        runs.append((lab, col, sorted(pts, reverse=True), m, sd, eff, nb, eff0))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax in axes:
        style(ax)

    # --- left: per-seed returns ------------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.2, zorder=2)
    ax.annotate("exp_c18 hyperplane baseline  4308 ± 500", xy=(4.42, BASE_M),
                xytext=(0, 7), textcoords="offset points", color=ORANGE,
                fontsize=8.6, ha="right", fontweight="bold")
    for i, (lab, col, pts, m, sd, eff, nb, eff0) in enumerate(runs):
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=2.4, alpha=0.9,
                zorder=3)
        ax.scatter([i] * len(pts), pts, s=95, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.annotate(f"{pts[0]:.0f}", xy=(i, pts[0]), xytext=(9, -3),
                    textcoords="offset points", color=col, fontsize=8.6,
                    fontweight="bold", va="center")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r[0] for r in runs], fontsize=8.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5000)
    ax.set_title("One seed reaches ~3,200 — unless the boundaries are frozen",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    ax.annotate("horizontal bar = 3-seed mean\ndots = individual seeds",
                xy=(0.02, 0.055), xycoords="axes fraction", color=MUTED, fontsize=8.2)

    # --- right: addressing statistic vs return ---------------------------------
    ax = axes[1]
    for lab, col, pts, m, sd, eff, nb, eff0 in runs:
        if eff is None:
            continue
        if eff0 is not None:
            ax.annotate("", xy=(eff, m), xytext=(eff0, m),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.8,
                                        alpha=0.85))
            ax.scatter([eff0], [m], s=60, facecolor="white", edgecolor=col,
                       linewidth=1.8, zorder=4)
            ax.annotate("c34 starts here (7.8, quantile init)\nand trains DOWN to 3.1",
                        xy=(eff0, m), xytext=(-10, 22), textcoords="offset points",
                        color=col, fontsize=8.2, ha="right")
        ax.scatter([eff], [m], s=130, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.8)
        ax.annotate(f"  {lab.splitlines()[0]} ({nb} bkt)", xy=(eff, m),
                    xytext=(8, -10), textcoords="offset points", color=col,
                    fontsize=8.4, va="center")
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, alpha=0.9)
    ax.annotate("baseline 4308", xy=(10.2, BASE_M), xytext=(0, 6),
                textcoords="offset points", color=ORANGE, fontsize=8.4, ha="right")
    ax.set_xlabel("EFFECTIVE buckets in use at the end of training  (2**entropy)",
                  color=MUTED, fontsize=9.5)
    ax.set_ylabel("3-seed mean CPU reference", color=MUTED, fontsize=9.5)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5000)
    ax.set_title("The lever moved. The return did not — and pinning it HURT.",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    fig.suptitle("The bucket-LIF line — three levers on the addressing, one flat result",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.030,
             "Capacity (16→64 buckets, param-matched at 99.2% of baseline) and boundary "
             "PLACEMENT (uniform → equal-mass quantiles) varied independently. "
             "Every pairwise difference among the three: |t| ≤ 0.47.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.006, 0.008,
             "Best seed 3234 / 3166 / 3245 with FREE boundaries. Freezing them (c35) is "
             "the only change that mattered — and it made things WORSE: 0/300 full "
             "episodes. The boundaries have to track the policy.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(D, "exp_c34_quantile_boundaries", "bucket_line_summary.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
