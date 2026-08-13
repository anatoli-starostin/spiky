"""exp_c48 — reproducing c36's settings on the current module, and what the gap isolates.

Run in the SPIKY venv (matplotlib).

LEFT — the three configurations that share the 128 × 1 × 16 per-table shape and the 31,360
parameter budget, so the only differences are the ones named under each. c47 and c48 differ
in EXACTLY (table init + delays) and land on top of each other; c36 differs from c48 in the
module AND the temperature freeze, and sits ~1,640 above both.

RIGHT — c36's own logged temperature trajectory, which is the reason the c48-vs-c36 gap
cannot be read as "the refactor regressed". c36's T_bkt annealed 1.000 -> 0.018 over its
run; every MHL run since c38 pins both temperatures at exactly 1.000.

Usage:
  python plot_c48.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C36 = os.path.join(D, "exp_c36_bucket_tables")
BLUE, ORANGE, GREEN, RED, MUTED2 = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C47 = [3920.7, 3653.6, 776.1]
C36_SEEDS = [4527.5, 3933.2, 4277.6]


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
    c48 = [r["seeds"][k] for k in sorted(r["seeds"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: the three same-shape configs ------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.set_ylim(0, 5000)
    groups = [("c47\nunified module\nfan-in init + delays", C47, BLUE),
              ("c48\nunified module\nSTOCK init, ZERO delays", c48, RED),
              ("c36 ORIGINAL\nold module\nTRAINABLE temps", C36_SEEDS, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=110, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(38, -4), textcoords="offset points",
                    color=col, fontsize=9.2, fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= 3000)}/3", xy=(i, 150), color=col,
                    fontsize=9.0, ha="center", fontweight="bold")
    ax.annotate("", xy=(1, 2608), xytext=(0, 2784),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate("−175  |t| 0.15\n(fan-in + delays\nworth nothing)", xy=(0.5, 2700),
                xytext=(0, 34), textcoords="offset points", color=INK, fontsize=8.4,
                ha="center", fontweight="bold")
    ax.annotate("", xy=(2, 4246), xytext=(1, 2608),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate("−1638  |t| 2.46\n(module AND\ntemperature freeze)", xy=(1.62, 3600),
                xytext=(0, 0), textcoords="offset points", color=INK, fontsize=8.4,
                ha="left", va="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("All three: 128 tables × 1 detector × 16 buckets, per-table\n"
                 "ladders, 31,360 params, seeds 0/1/2",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: c36's temperature anneal ---------------------------------
    ax = axes[1]
    h = json.load(open(os.path.join(C36, "bucket_sac_c36_s0.json")))["history"]
    it = [e["iter"] for e in h]
    ax.plot(it, [e["t_bkt"] for e in h], "o-", color=RED, linewidth=2.2, markersize=4,
            label="c36 T_bkt (trainable)")
    ax.plot(it, [e["t_cross"] for e in h], "s-", color=BLUE, linewidth=2.0, markersize=4,
            label="c36 T_cross (trainable)")
    ax.axhline(1.0, color=INK, linewidth=1.8, linestyle="--", alpha=0.75)
    ax.annotate("c47 / c48: BOTH pinned at 1.000 by freeze_temperature",
                xy=(9800, 1.02), color=INK, fontsize=8.4, ha="right", va="bottom",
                fontweight="bold")
    ax.annotate(f"T_bkt → {h[-1]['t_bkt']:.3f}\n(55× sharper)",
                xy=(it[-1], h[-1]["t_bkt"]), xytext=(-8, 34),
                textcoords="offset points", color=RED, fontsize=8.6, ha="right",
                fontweight="bold")
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("temperature", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="center right")
    ax.set_title("Why the c48↔c36 gap is not attributable to the refactor alone",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c48 — c36's two init settings on the CURRENT module",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c48 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in sorted(c48, reverse=True))}), takeoff "
             f"{r['takeoff']}/3. vs c47 −175, |t| 0.15 — reverting to stock init + zero "
             f"delays changed NOTHING measurable, so the fan-in/delay settings are not "
             f"the cause of the c47 gap.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "vs c36 −1638, |t| 2.46, all three seeds lower. But c48 differs from c36 in "
             "TWO ways — the module AND the temperature freeze — so this is not a verdict "
             "on the refactor. The decisive next run is c48 with temperatures UNFROZEN.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c48_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
