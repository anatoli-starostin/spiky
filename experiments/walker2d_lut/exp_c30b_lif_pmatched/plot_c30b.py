"""exp_c30b — the param-matched LIF actor, against both anchors.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

LEFT: learning curves against the eps anneal, same as exp_c30 -- and for the same reason.
All three seeds peak before the end and give return back over the final sharpening, and
the checkpoint is the FINAL actor, so the quoted numbers pay that cost. A chart of finals
alone would hide a pattern that has now repeated across six independent runs.

RIGHT: the comparison the experiment exists for, as three groups with their PARAMETER
COUNTS on the axis, because that is the variable under test. Points are per-seed, the bar
is the mean, the whisker is +/- 1 sd. Drawn on one axis rather than as three separate
panels so the eye compares heights directly.

Colors: orange #eb6834 = the hyperplane front-end, blue #2a78d6 = the LIF front-end, with
the un-matched dense-P variant at lower alpha since it is the same model on a bigger
budget rather than a different treatment. Matches plot_c29/plot_c30 so the chapter reads
as one set.

Usage:
  python plot_c30b.py
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
    hist = {s: json.load(open(os.path.join(HERE, f"lif_sac_c30b_s{s}.json")))["history"]
            for s in SEEDS}
    cpu = [json.load(open(os.path.join(HERE, f"lif_sac_c30b_s{s}_cpueval.json")))
           ["cpu_reference_mean"] for s in SEEDS]
    m = sum(cpu) / len(cpu)
    sd = (sum((v - m) ** 2 for v in cpu) / (len(cpu) - 1)) ** 0.5

    # (label, params, mean, sd, per-seed points or None, color, alpha)
    groups = [("exp_c18\nhyperplane", 49152, 4308.0, 500.1, None, ORANGE, 1.0),
              ("exp_c30\nLIF, dense P", 87361, 3931.3, 585.8, None, BLUE, 0.42),
              ("exp_c30b\nLIF, factorised P", 48193, m, sd, cpu, BLUE, 1.0)]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    for ax in axes:
        style(ax)
    axes[0].grid(True, color=GRID, linewidth=0.8, alpha=0.9)

    # --- left: learning curves ---------------------------------------------
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        h = hist[s]
        ax.plot([r["iter"] for r in h], [r["mjx_return"] for r in h],
                color=BLUE, alpha=0.45 + 0.25 * i, linewidth=1.9, zorder=3,
                label=f"seed {s}")
        pk = max(h, key=lambda r: r["mjx_return"])
        ax.scatter([pk["iter"]], [pk["mjx_return"]], s=42, color=BLUE,
                   alpha=0.45 + 0.25 * i, edgecolor="white", linewidth=1.4, zorder=4)
    ax.set_xlabel("SAC iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("20-ep MJX proxy (scored at the current eps)", color=MUTED,
                  fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    ax.set_title("Peaks marked — every seed gives return back as eps sharpens",
                 color=INK, fontsize=12, loc="left", pad=10)

    ax2 = ax.twinx()
    ax2.plot([r["iter"] for r in hist[0]], [r["eps"] for r in hist[0]],
             color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax2.set_ylabel("eps (gate sharpness)", color=MUTED, fontsize=9.5)
    ax2.tick_params(colors=MUTED, labelsize=9, length=3)
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    mid = hist[0][len(hist[0]) // 2]
    ax2.annotate("eps 2.0 → 0.3", xy=(mid["iter"], mid["eps"]), xytext=(0, 10),
                 textcoords="offset points", color=MUTED, fontsize=8.5, ha="center")

    # --- right: the three-way comparison at stated parameter budgets --------
    ax = axes[1]
    for i, (lab, par, gm, gsd, pts, col, al) in enumerate(groups):
        ax.bar(i, gm, width=0.56, color=col, alpha=al * 0.30, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [gm, gm], color=col, alpha=al, linewidth=2.8,
                zorder=4)
        ax.plot([i, i], [gm - gsd, gm + gsd], color=col, alpha=al, linewidth=1.6,
                zorder=3)
        if pts:
            ax.scatter([i] * len(pts), pts, s=70, color=col, alpha=al,
                       edgecolor="white", linewidth=1.8, zorder=5)
        # Above whichever is higher, the whisker top or the highest seed point. A label
        # at the mean is struck through by the whisker; one at the whisker top collides
        # with a seed that sits above it (c30b's seed 1 does).
        top = max([gm + gsd] + (list(pts) if pts else []))
        ax.annotate(f"{gm:.0f}", xy=(i, top), xytext=(0, 10),
                    textcoords="offset points", fontsize=9.5, color=col, alpha=al,
                    ha="center", fontweight="bold")
        ax.annotate(f"{par:,}\nparams", xy=(i, 0), xytext=(0, 8),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylim(0, 6100)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Same result, 45% fewer parameters", color=INK, fontsize=12,
                 loc="left", pad=10)

    fig.suptitle("exp_c30b — param-matched LIF detectors as the Walker2d SAC actor's "
                 "index front-end",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             "Ordered-pair channel P factorised to rank 2 plus a per-source-channel "
             "bias: 48,193 actor params, 1.95% UNDER the 49,152 of the hyperplane "
             "baseline. Bars are means, whiskers ±1 sd.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.008, 0.008,
             "Neither difference is resolvable at these seed counts: vs hyperplane "
             "−221 (Welch se 608, |t| 0.36); vs dense-P LIF +156 (se 665, |t| 0.23).",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.94))
    out = os.path.join(HERE, "c30b_pmatched_result.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
