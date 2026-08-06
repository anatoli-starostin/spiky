"""exp_c49 — the temperature freeze was NOT the cause, and what is.

Run in the SPIKY venv (matplotlib).

LEFT — the verdict. c49 unfroze the temperatures and they annealed to c36's values almost
exactly, yet the return did not move (-375 vs c48, |t| 0.39) and the gap to c36 remains
(-2013, |t| 2.69). So the freeze is exonerated and the module refactor is implicated.

RIGHT — the mechanism, found by looking at the delays. `LIFMultiHeadLUT` clamps the delay
to [0, t_window]; the old `BucketLIFDetectorsMHL` did not. Starting from delay_init_std=0,
the first updates push most delays below zero, where the clamp makes them functionally zero
AND gives them exactly zero gradient -- permanently dead. c36's delays, unclamped, spread
symmetrically across [-10, +13] with ~40% negative and fully functional.

Usage:
  python plot_c49.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C36 = os.path.join(D, "exp_c36_bucket_tables")
C48 = os.path.join(D, "exp_c48_c36_repro")
BLUE, ORANGE, GREEN, RED, MUTED2 = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C48_S = [3212.5, 1323.0, 3288.9]
C36_S = [4527.5, 3933.2, 4277.6]


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
    c49 = [r["seeds"][k] for k in sorted(r["seeds"])]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: the verdict ------------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.set_ylim(0, 5000)
    groups = [("c48\ntemps FROZEN\nT=1.000", C48_S, RED),
              ("c49\ntemps UNFROZEN\nT_bkt → 0.01–0.07", c49, BLUE),
              ("c36 ORIGINAL\nold module\nT_bkt → 0.018", C36_S, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=110, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(38, -4), textcoords="offset points",
                    color=col, fontsize=9.2, fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= 3000)}/3", xy=(i, 150), color=col,
                    fontsize=9.0, ha="center", fontweight="bold")
    ax.annotate("", xy=(1, 2233), xytext=(0, 2608),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate("−375  |t| 0.39\nUNFREEZING DID NOTHING", xy=(0.5, 2420),
                xytext=(0, -34), textcoords="offset points", color=INK, fontsize=8.4,
                ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("c49 reproduced c36's temperature anneal — and the return\n"
                 "did not move. The freeze is exonerated.",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: the delay distributions -----------------------------------
    ax = axes[1]
    d36 = np.concatenate([np.load(os.path.join(
        C36, f"bucket_sac_c36_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    d49 = np.concatenate([np.load(os.path.join(
        HERE, f"mhl_sac_c49_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    bins = np.linspace(-12, 14, 70)
    ax.hist(d36, bins=bins, color=GREEN, alpha=0.72, label="c36 — NO clamp (old module)")
    ax.hist(d49, bins=bins, color=BLUE, alpha=0.72,
            label="c49 — clamped to [0, 32] (current module)")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 2e4)
    ax.axvline(0.0, color=INK, linewidth=1.8, linestyle="--", alpha=0.85)
    ax.annotate("clamp floor", xy=(0.4, 3.5), color=INK, fontsize=8.4, ha="left",
                fontweight="bold")
    ax.annotate(f"{100*(d49 <= 0).mean():.0f}% of c49's delays pile up at 0\n"
                f"— clamped in the forward AND carrying\nexactly zero gradient: DEAD",
                xy=(1.6, 2200), color=BLUE, fontsize=8.4, ha="left", fontweight="bold")
    ax.annotate(f"c36: {100*(d36 < 0).mean():.0f}% negative and fully\n"
                f"functional (= earlier arrival),\nspanning −10 … +13",
                xy=(-11.5, 60), color=GREEN, fontsize=8.4, ha="left", fontweight="bold")
    ax.set_xlabel("learned delay", color=MUTED, fontsize=9.5)
    ax.set_ylabel("count, log scale (3 seeds pooled, 6,528 delays)",
                  color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="upper left")
    ax.set_title("The mechanism: the delay clamp kills ~95% of the delays",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c49 — temperatures unfrozen: the freeze was not the cause",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c49 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in sorted(c49, reverse=True))}), takeoff "
             f"{r['takeoff']}/3. Temperatures annealed to c36's values almost exactly "
             f"(T_bkt → 0.010–0.067 vs c36's 0.018; T_cross → 0.409–0.479 vs 0.436), yet "
             f"vs c48 only −375, |t| 0.39. vs c36 −2013, |t| 2.69.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Prime suspect is now the delay clamp `clamp(delay, 0, t_window)`, added by "
             "the unified module and absent from the old one: it zeroes both the value "
             "and the gradient below 0, collapsing 2,176 delay params to ~100 live ones.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c49_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
