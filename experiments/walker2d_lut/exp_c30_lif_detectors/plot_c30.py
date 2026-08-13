"""exp_c30 — the LIF-detector actor's learning curves and where it lands.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

TWO PANELS. Left: the training proxy per seed against the eps anneal, because the single
most visible feature of this experiment is that ALL THREE seeds peak before the end and
give back return over the last stretch of sharpening -- and the checkpoint is the final
actor, so the quoted numbers pay that cost. A bar chart of finals would hide it entirely.
Right: the 100-episode CPU reference per seed against the exp_c18 hyperplane cell at the
same nap6/tph32 shape, drawn as a band because that anchor is itself 6 noisy seeds and
plotting it as a line would imply a precision it does not have.

Colors are categorical slots 1 and 2 of the validated default palette (blue #2a78d6 =
LIF, orange #eb6834 = the hyperplane anchor), matching plot_c29.py so the chapter's
figures read as one set. The three seeds share the blue slot at different alphas: this is
a two-category comparison (front-end vs front-end), not a three-category one.

Usage:
  python plot_c30.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
SEEDS = (0, 1, 2)
ANCHOR_MEAN, ANCHOR_SD = 4308.0, 500.1


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    hist = {s: json.load(open(os.path.join(HERE, f"lif_sac_c30_s{s}.json")))["history"]
            for s in SEEDS}
    cpu = {s: json.load(open(os.path.join(HERE, f"lif_sac_c30_s{s}_cpueval.json")))
           for s in SEEDS}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.55, 1]))
    for ax in axes:
        style(ax)

    # --- left: learning curves against the eps anneal -----------------------
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        h = hist[s]
        ax.plot([r["iter"] for r in h], [r["mjx_return"] for r in h],
                color=BLUE, alpha=0.45 + 0.25 * i, linewidth=1.9, zorder=3,
                label=f"seed {s}")
        # Mark the peak and the terminal value: the gap between them IS the finding.
        pk = max(h, key=lambda r: r["mjx_return"])
        ax.scatter([pk["iter"]], [pk["mjx_return"]], s=42, color=BLUE,
                   alpha=0.45 + 0.25 * i, edgecolor="white", linewidth=1.4, zorder=4)
    ax.set_xlabel("SAC iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("20-ep MJX proxy (scored at the current eps)", color=MUTED,
                  fontsize=9.5)
    # Lower right: the eps guide falls diagonally across the upper left, and the curves
    # all finish high, so this is the only quadrant that stays empty.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    ax.set_title("Every seed peaks early and gives return back as eps sharpens",
                 color=INK, fontsize=12, loc="left", pad=10)

    ax2 = ax.twinx()
    ax2.plot([r["iter"] for r in hist[0]], [r["eps"] for r in hist[0]],
             color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax2.set_ylabel("eps (gate sharpness)", color=MUTED, fontsize=9.5)
    ax2.tick_params(colors=MUTED, labelsize=9, length=3)
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.annotate("eps 2.0 → 0.3", xy=(hist[0][len(hist[0]) // 2]["iter"],
                                      hist[0][len(hist[0]) // 2]["eps"]),
                 xytext=(0, 10), textcoords="offset points",
                 color=MUTED, fontsize=8.5, ha="center")

    # --- right: the result against the exp_c18 anchor -----------------------
    ax = axes[1]
    ax.axhspan(ANCHOR_MEAN - ANCHOR_SD, ANCHOR_MEAN + ANCHOR_SD, color=ORANGE,
               alpha=0.16, zorder=1)
    ax.axhline(ANCHOR_MEAN, color=ORANGE, linewidth=2.0, zorder=2)
    # Inside the band and to the right of every data point. Above the band there is no
    # room (the band top is the axis top, so the label lands in the panel title); on the
    # mean line it sits on top of the line itself.
    ax.annotate(f"exp_c18 hyperplane\nnap6/tph32, 6 seeds\n{ANCHOR_MEAN:.0f} ± "
                f"{ANCHOR_SD:.0f}", xy=(2.60, ANCHOR_MEAN - 90),
                fontsize=8.5, color=ORANGE, va="top", linespacing=1.35)
    vals = [cpu[s]["cpu_reference_mean"] for s in SEEDS]
    ax.scatter(range(len(SEEDS)), vals, s=100, color=BLUE, edgecolor="white",
               linewidth=2, zorder=4)
    m = sum(vals) / len(vals)
    ax.plot([-0.35, len(SEEDS) - 0.65], [m, m], color=BLUE, linewidth=2.6, zorder=3)
    ax.annotate(f"mean {m:.0f}", xy=(-0.42, m), fontsize=9, color=BLUE, ha="right",
                va="center")
    for i, s in enumerate(SEEDS):
        ax.annotate(f"{vals[i]:.0f}", xy=(i, vals[i]), xytext=(0, 12),
                    textcoords="offset points", fontsize=8.5, color=BLUE, ha="center")
    ax.set_xlim(-1.05, 4.15)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=9.5)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Within noise of the hyperplane front-end", color=INK, fontsize=12,
                 loc="left", pad=10)

    fig.suptitle("exp_c30 — LIF detectors as the Walker2d SAC actor's index front-end, "
                 "3 seeds, trained from scratch",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             "JAX port of LIFDetectorsMHL, mode=\"st\" (hard forward / full-K softmax "
             "backward); parity with the torch reference 13/13, table gradient "
             "bit-identical.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.008, 0.008,
             "NOT param-matched: 87,361 actor params vs 49,152 for the LUT actors "
             "(the ordered-pair channel P alone is 55,488).",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    out = os.path.join(HERE, "c30_lif_result.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
