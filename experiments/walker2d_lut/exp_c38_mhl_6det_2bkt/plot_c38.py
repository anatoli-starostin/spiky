"""exp_c38 — the matched control against exp_c31, and the addressing diagnostic that broke
the chapter's pattern.

Run in the SPIKY venv (matplotlib).

LEFT — exp_c38 against its matched control. c31 and c38 have the SAME 32 tables, the SAME
64 rows per table, the SAME 24,576-entry table, and totals within 1.1%. The only thing that
differs is where the six address bits come from: six deadlines on ONE LIF (c31) versus six
INDEPENDENT LIFs (c38). Both come out bimodal, and the honest read is per-seed, not the
mean -- which is why the seeds are drawn individually and the means are drawn thin.

RIGHT — the finding. `eff` is 2**entropy of the per-table cell-occupancy distribution: how
many rows a table actually uses. Every bucket configuration from c32b to c37 converged to
1.7-2.5 REGARDLESS of bucket count, placement or freezing -- that was the c36 conclusion,
that per-table addressing entropy is pinned and predicts nothing. c38 is the first
configuration in the chapter to leave that band, reaching 7.6-10.8 of 64. Six independent
detectors buy addressing diversity in a way that more buckets on one detector never did.

Whether that diversity BUYS RETURN is a separate question and the left panel is equivocal
about it. Both are shown rather than only the flattering one.

Usage:
  python plot_c38.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

# (label, sublabel, per-seed returns, mean, colour)
RUNS = [
    ("c32b", "1 det × 16 bkt\n32 tab · 7,840", [3234.0, 1697.0, 1192.6], 2041.2, MUTED),
    ("c37", "1 det × 32 bkt\n64 tab · 28,992", [3992.3, 1842.2, 1758.8], 2531.1, MUTED),
    ("c36", "1 det × 16 bkt\n128 tab · 31,360", [4528.0, 4181.0, 4029.3], 4246.1, GREEN),
    ("c31", "6 deadlines, 1 LIF\n32 tab · 31,392", [4262.1, 4073.3, 518.2], 2951.2, BLUE),
    ("c38", "6 LIF det × 2 bkt\n32 tab · 31,744", None, None, RED),
]


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
    r = json.load(open(os.path.join(HERE, "results.json")))
    c38_pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    runs = [list(x) for x in RUNS]
    runs[-1][2], runs[-1][3] = c38_pts, r["mean"]

    hist = {s: json.load(open(os.path.join(HERE, f"mhl_sac_c38_s{s}.json")))["history"]
            for s in (0, 1, 2)}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax in axes:
        style(ax)

    # ---- LEFT: per-seed returns ------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.2, zorder=2)
    ax.annotate("exp_c18 hyperplane baseline  4308 ± 500", xy=(-0.45, BASE_M),
                xytext=(0, 8), textcoords="offset points", color=ORANGE,
                fontsize=8.6, ha="left", fontweight="bold")
    for i, (lab, sub, pts, m, col) in enumerate(runs):
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=1.6, alpha=0.75,
                zorder=3)
        ax.scatter([i] * len(pts), pts, s=105, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(34, -4), textcoords="offset points",
                    color=col, fontsize=8.8, fontweight="bold", ha="center")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{a}\n{b}" for a, b, _, _, _ in runs], fontsize=7.9)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5200)
    ax.set_title("c38 vs its matched control c31 — same 32 tables, same 64 rows,\n"
                 "same 24,576 table, totals within 1.1%", color=INK, fontsize=11,
                 loc="left", pad=10)

    # ---- RIGHT: effective cells per table --------------------------------
    ax = axes[1]
    ax.axhspan(1.7, 2.5, color=MUTED, alpha=0.18, zorder=1)
    ax.annotate("every bucket config c32b–c37 converged\nin this band: 1.7–2.5 of its rows",
                xy=(4200, 0.85), color=MUTED, fontsize=8.4, ha="center", va="center")
    for s, col in zip((0, 1, 2), (RED, "#e8873b", "#7b3f9d")):
        h = hist[s]
        ax.plot([e["iter"] for e in h], [e["eff_cells"] for e in h], color=col,
                linewidth=1.9, alpha=0.9, label=f"seed {s} → {h[-1]['eff_cells']:.1f}")
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("effective cells used per table  (2^entropy, of 64)",
                  color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    ax.set_title("The finding: c38 is the first config to LEAVE that band",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c38 — LIFMultiHeadLUT: 32 tables × 6 independent LIF detectors × "
                 "2 buckets", color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.030,
             f"c38 {r['mean']:.0f} ± {r['sd']:.0f} (4117, 4072, 1452) — bimodal, like c31 "
             f"(4262, 4073, 518). vs c18 −1094 (|t| 1.21); vs c31 +263 (|t| 0.17); vs c36 "
             f"−1032 (|t| 1.15). 31,744 params = 113.2% of baseline.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Six INDEPENDENT detectors buy addressing diversity that more buckets on one "
             "detector never did (eff 1.7–2.5 → 7.6–10.8), but on 3 seeds that diversity "
             "does not separate from c31 or reach the baseline band.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(HERE, "c38_result.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
