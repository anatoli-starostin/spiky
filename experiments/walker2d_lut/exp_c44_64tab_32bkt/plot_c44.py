"""exp_c44 — the detector-count prediction, tested.

Run in the SPIKY venv (matplotlib).

LEFT — the prediction and the outcome. exp_c43 (32 detectors) scored 1177; exp_c37 has the
IDENTICAL shape and parameter count to c44 (64 detectors, 28,992 params) and scored 2531.
The detector-count reading predicted c44 would land near c37 and well above c43. Per-seed
points, because every configuration in this line is bimodal.

RIGHT — return against total LIF detectors across all nine configurations, c44 highlighted.
Spearman rather than a fitted line: nine configurations at n=3 support an ordinal claim
about direction, not a regression.

Usage:
  python plot_c44.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, MUTED2 = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

BARS = [
    ("c43\n32 detectors\n1 det × 64 bkt", [1554.5, 1375.5, 601.7], RED),
    ("c37\n64 detectors\nstock table init", [3992.3, 1842.2, 1758.8], MUTED2),
    ("c44\n64 detectors\nfan-in table init", None, BLUE),
]
SCATTER = [
    ("c43", 32, 1177.2), ("c33", 32, 1536.2), ("c32b", 32, 2041.2),
    ("c37", 64, 2531.1), ("c39", 96, 2030.2), ("c42+b", 96, 3043.7),
    ("c36", 128, 4246.1), ("c38", 192, 3213.9),
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
    c44 = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    bars = [(BARS[0][0], BARS[0][1], BARS[0][2]),
            (BARS[1][0], BARS[1][1], BARS[1][2]),
            (BARS[2][0], c44, BARS[2][2])]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5000)

    ax = axes[0]
    for i, (lab, vals, col) in enumerate(bars):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=105, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(36, -4), textcoords="offset points",
                    color=col, fontsize=9.0, fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= 3000)}/3", xy=(i, 140), color=col,
                    fontsize=9.0, ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(-0.55, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    ax.set_title("Doubling the detectors (c43 → c44) at the same parameter count,\n"
                 "and replicating c37's identical shape",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    ax = axes[1]
    for lab, nd, mm in SCATTER:
        ax.scatter([nd], [mm], s=95, color=MUTED, alpha=0.75, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(lab, xy=(nd, mm), xytext=(0, 11), textcoords="offset points",
                    color=MUTED, fontsize=8.2, ha="center")
    ax.scatter([64], [r["mean"]], s=135, color=BLUE, alpha=0.95, zorder=6,
               edgecolor="white", linewidth=1.7)
    ax.annotate("c44", xy=(64, r["mean"]), xytext=(0, -20), textcoords="offset points",
                color=BLUE, fontsize=8.6, ha="center", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 192])
    ax.set_xticklabels(["32", "64", "128", "192"])
    ax.set_xlim(26, 240)
    ax.set_xlabel("total LIF detectors  (n_tables × n_det)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean 100-ep CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title(f"Spearman ρ = {r['spearman_detectors']:+.2f} over "
                 f"{r['n_configs']} configurations,\nvs "
                 f"{r['spearman_tables']:+.2f} against table count alone",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c44 — 64 tables × 1 detector × 32 buckets: the detector-count "
                 "prediction holds", color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c44 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c44)}), takeoff {r['takeoff']}/3, at "
             f"28,992 params. vs c37 (identical shape and params, stock table init) "
             f"−170, |t| 0.16 — a near-exact replication. vs c43 (half the detectors) "
             f"+1184, |t| 1.47.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "The fan-in table init did NOT lift c44 above c37, consistent with c42b: "
             "that correction is principled and mildly helpful but not resolvable at n=3. "
             "Ordinal claim only — several of these means overlap in their own seed noise.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c44_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
