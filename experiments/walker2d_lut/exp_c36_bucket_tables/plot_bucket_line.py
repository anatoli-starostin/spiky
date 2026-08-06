"""The bucket-LIF line, exp_c32 → exp_c36: what actually predicts the return.

Run in the SPIKY venv.

LEFT — per-seed returns. Points, not bars, because the shape is the finding: every
32-table configuration produces exactly one seed near ~3,200 and the rest stalled under
1,100, while exp_c36 (128 tables) lifts ALL THREE onto the baseline band.

RIGHT — the hypothesis test. x is the per-table addressing entropy actually realised at
the end of training; marker area is the number of tables. Four configurations at 32 tables
span 1.45–2.54 bits and return 1,213–2,041 in no particular order — per-table addressing
predicts nothing. exp_c36 has among the LOWEST per-table entropy in the set (1.72 bits,
3.7 of 16 effective buckets) and the HIGHEST return, because it has 128 independent
addresses instead of 32.

That is the whole result: the lever is the NUMBER OF INDEPENDENT INDICES, not the
resolution of each one. exp_c33 spent a nearly identical parameter budget on the opposite
choice — 64 buckets × 32 tables, 27,808 params against exp_c36's 31,360 — and landed
2,710 points lower.

Usage:
  python plot_bucket_line.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLUE, ORANGE, RED, PURPLE = "#2a78d6", "#eb6834", "#b03030", "#7a4fbf"
GREEN = "#1f9e5a"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"

# (label, dir, colour, fallback per-seed, mean, sd, per-table entropy bits, n_tables,
#  n_buckets)
RUNS = [
    ("c32 BROKEN\n16×32", None, RED, [1739.9, 725.5, 1057.7], 1174.4, 517.2,
     None, 32, 16),
    ("c32b\n16×32", "exp_c32b_bucket_fixed", BLUE, None, None, None, 2.02, 32, 16),
    ("c33\n64×32", "exp_c33_bucket64", BLUE, None, None, None, 2.54, 32, 64),
    ("c34\n16×32 quant", "exp_c34_quantile_boundaries", BLUE, None, None, None,
     1.45, 32, 16),
    ("c35\n16×32 frozen", "exp_c35_frozen_boundaries", PURPLE, None, None, None,
     1.68, 32, 16),
    ("c36\n16×128", "exp_c36_bucket_tables", GREEN, None, None, None, 1.72, 128, 16),
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
    for lab, d, col, pts, m, sd, ent, nt, nb in RUNS:
        if d is not None:
            r = json.load(open(os.path.join(D, d, "results.json")))
            pts = [r["seeds"][k] for k in sorted(r["seeds"])]
            m, sd = r["mean"], r["sd"]
        runs.append((lab, col, sorted(pts, reverse=True), m, sd, ent, nt, nb))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax in axes:
        style(ax)

    # --- left: per-seed returns ------------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.2, zorder=2)
    # Left-aligned: c36's mean now sits inside the band on the right-hand side, where a
    # right-aligned caption lands on top of its label.
    ax.annotate("exp_c18 hyperplane baseline  4308 ± 500", xy=(-0.42, BASE_M),
                xytext=(0, 8), textcoords="offset points", color=ORANGE,
                fontsize=8.6, ha="left", fontweight="bold")
    for i, (lab, col, pts, m, sd, ent, nt, nb) in enumerate(runs):
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.6, alpha=0.95,
                zorder=3)
        ax.scatter([i] * len(pts), pts, s=95, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.7)
        # c36's mean sits inside the baseline band, where a label below the bar lands on
        # its own seed points; put that one above instead.
        dy = 9 if nt == 128 else -16
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(26, dy),
                    textcoords="offset points", color=col, fontsize=8.8,
                    fontweight="bold", ha="center")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([r[0] for r in runs], fontsize=8.4)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5200)
    ax.set_title("128 tables lifts ALL THREE seeds onto the baseline band",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    ax.annotate("bar = 3-seed mean · dots = seeds\nlabels are buckets × tables",
                xy=(0.015, 0.045), xycoords="axes fraction", color=MUTED, fontsize=8.2)

    # --- right: per-table entropy vs return, sized by table count ---------------
    ax = axes[1]
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, alpha=0.9)
    ax.annotate("baseline 4308", xy=(2.68, BASE_M), xytext=(0, 6),
                textcoords="offset points", color=ORANGE, fontsize=8.4, ha="right")
    for lab, col, pts, m, sd, ent, nt, nb in runs:
        if ent is None:
            continue
        ax.scatter([ent], [m], s=40 + 2.6 * nt, color=col, alpha=0.85, zorder=5,
                   edgecolor="white", linewidth=1.8)
        # c33 sits at the right edge; label it leftward so it does not run off the axes.
        left = ent > 2.3
        ax.annotate(f"{lab.splitlines()[0]} ({nt} tables)", xy=(ent, m),
                    xytext=(-12 if left else 12, -12 if nt == 32 else 4),
                    textcoords="offset points", color=col, fontsize=8.4,
                    va="center", ha="right" if left else "left")
    ax.set_xlabel("per-table addressing entropy realised at end of training (bits of 4)",
                  color=MUTED, fontsize=9.5)
    ax.set_ylabel("3-seed mean CPU reference", color=MUTED, fontsize=9.5)
    ax.set_xlim(1.2, 2.72)
    ax.set_ylim(0, 5200)
    ax.set_title("Per-table addressing predicts nothing. Table COUNT does.",
                 color=INK, fontsize=11.5, loc="left", pad=10)
    ax.annotate("marker area ∝ number of tables", xy=(0.02, 0.045),
                xycoords="axes fraction", color=MUTED, fontsize=8.2)

    fig.suptitle("The bucket-LIF line — the lever is the NUMBER of independent indices, "
                 "not the resolution of each",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.030,
             "Four configurations at 32 tables span 1.45–2.54 bits of per-table entropy "
             "and return 1,213–2,041 in no order. exp_c36 has among the LOWEST "
             "per-table entropy (1.72) and the highest return.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.006, 0.008,
             "c33 and c36 spend almost the same budget on opposite choices — 64 buckets×32 "
             "tables (27,808 params) vs 16 buckets×128 tables (31,360) — and differ by "
             "+2710 (|t| 3.24). c36 vs the baseline: −62, |t| 0.23.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(D, "exp_c36_bucket_tables", "bucket_line_summary.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
