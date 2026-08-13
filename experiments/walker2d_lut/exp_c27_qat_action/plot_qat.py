"""exp_c27 — post-hoc rounding vs quantization-aware training, at matched K.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

Both series use the SAME fixed [-1, 1] grid. exp_c26's headline numbers used each joint's
observed min/max range instead, which lands at +/-0.992..0.999 and shifts every level
slightly; that difference is not negligible at coarse K -- it is the whole of exp_c26's
K=5 wobble -- so the post-hoc half is recomputed on the QAT grid by
`posthoc_fixed_grid.py` rather than borrowed.

Two panels, not one chart with two y-axes: return and full-length count are different
measures, and the interesting divergence in this chapter is that one holds while the
other breaks. TWO dashed baselines, one per method, because they are genuinely different
runs -- the post-hoc reference is c21's own @10k checkpoint (5286.6) while the QAT arms
are fresh 10k runs whose unquantized control scored 4513.9. One shared line would credit
post-hoc with a 773-point head start that came from a different run, not from the method.

Colors are categorical slots 1 and 2 of the validated default palette (blue #2a78d6,
orange #eb6834), in fixed order -- the palette's documented passing adjacent pair. The
validator script could not be re-run on this box (no node), so this relies on its
published gate results rather than a fresh run.

Usage:
  python plot_qat.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
KS = [16, 7, 5, 3]


def main():
    ph_all = json.load(open(os.path.join(HERE, "posthoc_fixed_grid.json")))["rows"]
    base = [r for r in ph_all if r["K"] is None][0]
    ph = {r["K"]: r for r in ph_all if r["K"]}
    qat = {K: json.load(open(os.path.join(HERE, f"lut_sac_c27_K{K}_qat_cpueval.json")))
           for K in KS}
    ctl = json.load(open(os.path.join(HERE, "lut_sac_c27_K0_qat_cpueval.json")))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    x = list(range(len(KS)))
    for label, d, color in (("post-hoc (round a trained policy)", ph, BLUE),
                            ("QAT (trained on the grid)", qat, ORANGE)):
        axes[0].plot(x, [d[K]["mean"] for K in KS], color=color, linewidth=2.0,
                     marker="o", markersize=8, markeredgecolor="white",
                     markeredgewidth=2, label=label, zorder=3)
        axes[1].plot(x, [d[K]["full"] for K in KS], color=color, linewidth=2.0,
                     marker="o", markersize=8, markeredgecolor="white",
                     markeredgewidth=2, label=label, zorder=3)

    for ax, v, w in ((axes[0], base["mean"], ctl["mean"]),
                     (axes[1], base["full"], ctl["full"])):
        ax.axhline(v, color=BLUE, linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.55,
                   zorder=1)
        ax.axhline(w, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.55,
                   zorder=1)

    for ax, title, ylab in ((axes[0], "Closed-loop return", "100-episode mean return"),
                            (axes[1], "Episodes reaching full length",
                             "of 100 episodes")):
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("levels per joint  K        (coarser →)", color=MUTED, fontsize=9.5)
        ax.set_ylabel(ylab, color=MUTED, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)

    axes[0].annotate(f"c21 teacher, unquantized  {base['mean']:.0f}",
                     xy=(0.03, base["mean"]), xycoords=("axes fraction", "data"),
                     fontsize=8.5, color=BLUE, va="top")
    axes[0].annotate(f"K=0 control, same trainer  {ctl['mean']:.0f}",
                     xy=(0.03, ctl["mean"]), xycoords=("axes fraction", "data"),
                     fontsize=8.5, color=ORANGE, va="bottom")
    axes[1].annotate(f"{base['full']}/100", xy=(0.03, base["full"]),
                     xycoords=("axes fraction", "data"), fontsize=8.5, color=BLUE,
                     va="top")
    axes[1].annotate(f"{ctl['full']}/100", xy=(0.03, ctl["full"]),
                     xycoords=("axes fraction", "data"), fontsize=8.5, color=ORANGE,
                     va="bottom")
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left")

    fig.suptitle("exp_c27 — does training on the grid rescue what rounding afterwards "
                 "breaks?", color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.015,
             "dashed = each method's own unquantized reference. same fixed [-1,1] grid "
             "for both. 100-episode deterministic CPU reference, seed 4, 10k iters.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out = os.path.join(HERE, "qat_vs_posthoc.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
