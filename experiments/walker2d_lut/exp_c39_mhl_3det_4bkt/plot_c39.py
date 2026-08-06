"""exp_c39 — the width-against-count control, and whether ordered digits change addressing.

Run in the SPIKY venv (matplotlib).

LEFT — the controlled triple. c31, c38 and c39 all have 32 tables, 64 rows per table and
the same 24,576-entry table, under an identical SAC recipe. They differ only in how those
64 rows are addressed: 6 deadlines on ONE LIF (c31), 6 independent LIFs with a binary test
each (c38), or 3 independent LIFs with a 4-way ORDERED quantisation each (c39). c39 also
does it on half of c38's front-end, which puts it at 101.3% of the hyperplane baseline --
the closest parameter match in the chapter. Seeds are drawn individually and means thin,
because every model in this triple has come out bimodal and the mean is the weaker read.

RIGHT — the addressing diagnostic. `eff` is 2**entropy of the per-table cell-occupancy
distribution: how many of the 64 rows a table actually uses. Every bucket configuration
c32b-c37 converged to 1.7-2.5 regardless of bucket count, placement or freezing; exp_c38
was the first to break out, reaching 7.6-10.8. c38's seeds are drawn faded behind c39's so
the question this experiment asks is visible directly: do three ORDERED 4-way digits buy
the same addressing diversity as six independent bits, or does the diversity track the
number of independent detectors?

Usage:
  python plot_c39.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C38 = os.path.join(D, "exp_c38_mhl_6det_2bkt")
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b"
PURPLE = "#7b3f9d"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

# (label, sublabel, per-seed returns, mean, colour)
RUNS = [
    ("c32b", "1 det × 16 bkt\n32 tab · 7,840", [3234.0, 1697.0, 1192.6], 2041.2, MUTED),
    ("c37", "1 det × 32 bkt\n64 tab · 28,992", [3992.3, 1842.2, 1758.8], 2531.1, MUTED),
    ("c36", "1 det × 16 bkt\n128 tab · 31,360", [4528.0, 4181.0, 4029.3], 4246.1, GREEN),
    ("c31", "6 deadlines, 1 LIF\n32 tab · 31,392", [4262.1, 4073.3, 518.2], 2951.2, BLUE),
    ("c38", "6 LIF det × 2 bkt\n32 tab · 31,744", [4117.4, 4072.3, 1452.1], 3213.9,
     PURPLE),
    ("c39", "3 LIF det × 4 bkt\n32 tab · 28,384", None, None, RED),
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
    c39_pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    runs = [list(x) for x in RUNS]
    runs[-1][2], runs[-1][3] = c39_pts, r["mean"]

    h39 = {s: json.load(open(os.path.join(HERE, f"mhl_sac_c39_s{s}.json")))["history"]
           for s in (0, 1, 2)}
    h38 = {s: json.load(open(os.path.join(C38, f"mhl_sac_c38_s{s}.json")))["history"]
           for s in (0, 1, 2)}

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax in axes:
        style(ax)

    # ---- LEFT: per-seed returns ------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.2, zorder=2)
    ax.annotate("exp_c18 hyperplane baseline  4308 ± 500",
                xy=(-0.45, BASE_M - BASE_SD), xytext=(0, -13),
                textcoords="offset points", color=ORANGE,
                fontsize=8.6, ha="left", va="top", fontweight="bold")
    for i, (lab, sub, pts, m, col) in enumerate(runs):
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=1.6, alpha=0.75,
                zorder=3)
        ax.scatter([i] * len(pts), pts, s=100, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(32, -4), textcoords="offset points",
                    color=col, fontsize=8.6, fontweight="bold", ha="center")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{a}\n{b}" for a, b, _, _, _ in runs], fontsize=7.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 5200)
    ax.set_title("The controlled triple — c31, c38, c39 all have 32 tables × 64 rows\n"
                 "and the same 24,576 table; only the addressing differs",
                 color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: effective cells per table --------------------------------
    ax = axes[1]
    ax.axhspan(1.7, 2.5, color=MUTED, alpha=0.18, zorder=1)
    for s in (0, 1, 2):
        h = h38[s]
        ax.plot([e["iter"] for e in h], [e["eff_cells"] for e in h], color=PURPLE,
                linewidth=1.5, alpha=0.35, zorder=3,
                label="c38 · 6 det × 2 bkt" if s == 0 else None)
    for s in (0, 1, 2):
        h = h39[s]
        ax.plot([e["iter"] for e in h], [e["eff_cells"] for e in h], color=RED,
                linewidth=2.0, alpha=0.95, zorder=4,
                label="c39 · 3 det × 4 bkt" if s == 0 else None)
    ax.annotate("every bucket config c32b–c37 converged\n"
                "in this band: 1.7–2.5 of its rows",
                xy=(4600, 0.85), color=MUTED, fontsize=8.4, ha="center", va="center")
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("effective cells used per table  (2^entropy, of 64)",
                  color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="upper left")
    ax.set_title("Does digit WIDTH buy the diversity that digit COUNT did?",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c39 — LIFMultiHeadLUT: 32 tables × 3 LIF detectors × 4 buckets "
                 "(4³ = 64 cells, same rows as c38's 2⁶)",
                 color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    a38 = next(a for a in r["anchors"] if a["name"].startswith("exp_c38"))
    a18 = next(a for a in r["anchors"] if a["name"].startswith("exp_c18"))
    fig.text(0.006, 0.030,
             f"c39 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c39_pts)}) at 28,384 params = 101.3% of "
             f"the baseline — the closest param match in the chapter, on HALF c38's "
             f"front-end (3,808 vs 7,168).",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             f"vs c38 {a38['delta']:+.0f} (|t| {abs(a38['delta'])/a38['welch_se']:.2f}); "
             f"vs c18 {a18['delta']:+.0f} "
             f"(|t| {abs(a18['delta'])/a18['welch_se']:.2f}). "
             f"Three detectors carry half the delays and synapses of six, so this is the "
             f"cheaper way to address the same 64 rows.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(HERE, "c39_result.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
