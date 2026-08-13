"""exp_c45 — does the per-table bucket ladder carry real capacity?

Run in the SPIKY venv (matplotlib).

LEFT — the paired comparison. c44 and c45 are identical in every respect except that c45
ties `beta_base`/`beta_raw` across all 64 tables into one global ladder. Lines connect the
SAME seed, because the seed fixes both the init and the RL stream and so the comparison is
genuinely paired -- which matters here, since all three seeds moved the same way and that
is stronger evidence than the (underpowered) difference of means.

RIGHT — return against total LIF detectors, with c45 marked. c45 has the same 64 detectors
as c44 and c37 but scores far below both, which is the useful qualification: detector count
orders the configurations, but it is not sufficient on its own.

Usage:
  python plot_c45.py
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
C44 = {0: 2217.2, 1: 3730.8, 2: 1134.7}
SCATTER = [
    ("c43", 32, 1177.2), ("c33", 32, 1536.2), ("c32b", 32, 2041.2),
    ("c44", 64, 2360.9), ("c37", 64, 2531.1), ("c39", 96, 2030.2),
    ("c42+b", 96, 3043.7), ("c36", 128, 4246.1), ("c38", 192, 3213.9),
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
    c45 = {int(k): v for k, v in r["seeds"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5000)

    # ---- LEFT: paired ----------------------------------------------------
    ax = axes[0]
    for s in (0, 1, 2):
        ax.plot([0, 1], [C44[s], c45[s]], color=MUTED2, linewidth=1.5, alpha=0.8,
                zorder=3)
        ax.annotate(f"s{s}", xy=(1.04, c45[s]), color=MUTED, fontsize=8.0,
                    va="center", ha="left")
    ax.scatter([0] * 3, list(C44.values()), s=110, color=BLUE, alpha=0.92, zorder=5,
               edgecolor="white", linewidth=1.7)
    ax.scatter([1] * 3, list(c45.values()), s=110, color=RED, alpha=0.92, zorder=5,
               edgecolor="white", linewidth=1.7)
    for i, (m, col) in enumerate(((2360.9, BLUE), (r["mean"], RED))):
        ax.plot([i - 0.22, i + 0.22], [m, m], color=col, linewidth=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(0, 12), textcoords="offset points",
                    color=col, fontsize=9.4, fontweight="bold", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["c44\nper-table ladders\n64×31 betas · 28,992 par",
                        "c45\nSHARED ladder\n31 betas · 26,976 par"], fontsize=8.4)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(-0.45, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    ax.set_title("Paired by seed — ALL THREE got worse when the ladder was tied",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: detector count ------------------------------------------
    ax = axes[1]
    for lab, nd, mm in SCATTER:
        ax.scatter([nd], [mm], s=95, color=MUTED, alpha=0.75, zorder=5,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(lab, xy=(nd, mm), xytext=(0, 11), textcoords="offset points",
                    color=MUTED, fontsize=8.2, ha="center")
    ax.scatter([64], [r["mean"]], s=140, color=RED, alpha=0.95, zorder=6,
               edgecolor="white", linewidth=1.7)
    ax.annotate("c45\nshared", xy=(64, r["mean"]), xytext=(0, -28),
                textcoords="offset points", color=RED, fontsize=8.4, ha="center",
                fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 192])
    ax.set_xticklabels(["32", "64", "128", "192"])
    ax.set_xlim(26, 240)
    ax.set_xlabel("total LIF detectors  (n_tables × n_det)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean 100-ep CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Same 64 detectors as c44 and c37 — and far below both.\n"
                 "Detector count orders the line; it is not sufficient.",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c45 — sharing the bucket ladder across all 64 tables",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c45 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{c45[s]:.0f}' for s in (0, 1, 2))}), takeoff "
             f"{r['takeoff']}/3, at 26,976 params. vs c44 −1163, Welch se 928, |t| 1.25 — "
             f"underpowered at n=3, but PAIRED all three seeds fell.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "It removed 2,016 front-end params (beta 2,048 → 32) and cost ~1,163 return. "
             "Return per 1k params fell 81.4 → 44.4 — not a favourable trade even "
             "per-parameter. The per-table ladder is real capacity, not dead weight.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c45_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
