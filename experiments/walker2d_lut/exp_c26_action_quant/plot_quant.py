"""exp_c26 — return vs quantization levels. Run in the SPIKY venv (matplotlib lives there;
the mjx venv has jax but no matplotlib, so the sweep and the plot are separate steps).

Two panels rather than one chart with two y-scales: return and full-length-episode count
are different measures, and a dual axis would invite reading a crossing as meaningful.
The second panel is not decoration -- it is where the story actually is, because the mean
holds at K=5 while reliability has already started to break.

Colors are categorical slots 1 and 2 of the validated default palette (blue #2a78d6,
orange #eb6834), used in fixed order. That adjacent pair is the documented passing case;
the validator script itself could not be re-run here (no node on this box), so this relies
on the palette's published gate results rather than a fresh run.

Usage:
  python plot_quant.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
SERIES = [("@10k  (5286.6, 100/100)", "@10k", BLUE),
          ("@20k  (5647.5, 97/100)", "@20k", ORANGE)]


def main():
    d = json.load(open(os.path.join(HERE, "action_quant.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white")

    for ax in axes:
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    ks = [r["K"] for r in d["actors"]["@10k"]["rows"] if r["K"]]
    x = list(range(len(ks)))

    for label, key, color in SERIES:
        rows = [r for r in d["actors"][key]["rows"] if r["K"]]
        base = [r for r in d["actors"][key]["rows"] if r["K"] is None][0]

        axes[0].axhline(base["mean"], color=color, linewidth=1.4, linestyle=(0, (4, 3)),
                        alpha=0.55, zorder=1)
        axes[0].plot(x, [r["mean"] for r in rows], color=color, linewidth=2.0,
                     marker="o", markersize=8, markeredgecolor="white",
                     markeredgewidth=2, label=label, zorder=3)
        axes[1].axhline(base["full"], color=color, linewidth=1.4, linestyle=(0, (4, 3)),
                        alpha=0.55, zorder=1)
        axes[1].plot(x, [r["full"] for r in rows], color=color, linewidth=2.0,
                     marker="o", markersize=8, markeredgecolor="white",
                     markeredgewidth=2, label=label, zorder=3)

    for ax, title, ylab in ((axes[0], "Closed-loop return", "100-episode mean return"),
                            (axes[1], "Episodes reaching full length",
                             "of 100 episodes")):
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("levels per joint  K        (coarser →)", color=MUTED,
                      fontsize=9.5)
        ax.set_ylabel(ylab, color=MUTED, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)

    # The knee, marked once rather than described in a caption nobody reads.
    knee = ks.index(7)
    for ax in axes:
        ax.axvspan(knee - 0.5, knee + 0.5, color="#f4f3ef", zorder=0)
    axes[0].annotate("K = 7 holds", xy=(knee, 5480), fontsize=9, color=MUTED,
                     ha="center", va="bottom")

    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left")
    fig.suptitle("exp_c26 — quantizing the LUT teacher's torque OUTPUTS, per joint, "
                 "in the closed loop", color=INK, fontsize=13.5, x=0.008, ha="left",
                 y=0.985)
    fig.text(0.008, 0.015,
             "dashed line = unquantized baseline for that actor.  midtread uniform "
             "quantizer on each joint's observed min/max range.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out = os.path.join(HERE, "action_quant.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
