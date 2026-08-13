"""exp_c29 — the none-vs-grid contrast and what the added constants actually did.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

TWO PANELS, because the result has two halves that point in opposite directions and a
single chart would hide one of them. Left: the paired per-seed return, drawn as lines
joining the same seed in both arms rather than as two bar means -- the arms share a seed,
so the paired view is the comparison with any power at n=3, and it shows immediately that
one seed dissents. Right: per-constant binary entropy on the states the trained policies
actually visit, averaged over seeds, which is the answer to "which of the sixteen
constants does the LUT key off".

Colors are categorical slots 1 and 2 of the validated default palette (blue #2a78d6,
orange #eb6834). The validator could not be re-run here (no node), so this relies on its
published gate results.

Usage:
  python plot_c29.py [--wave c29]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
SEEDS = (0, 1, 2)
LABEL = {"c29": "balanced · nap6/tph64",
         "c29c": "canonical_full_coverage · nap6/tph64",
         "c29m": "balanced · nap5/tph128",
         "c29mc": "canonical_full_coverage · nap5/tph128"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="c29")
    a = ap.parse_args()
    tag = a.wave

    def ev(arm, s):
        p = os.path.join(HERE, f"lut_sac_{tag}_{arm}_s{s}_cpueval.json")
        return json.load(open(p)) if os.path.exists(p) else None

    none = [ev("none", s) for s in SEEDS]
    grid = [ev("grid", s) for s in SEEDS]
    ok = [i for i in range(len(SEEDS)) if none[i] and grid[i]]
    if not ok:
        raise SystemExit(f"no completed pairs for wave {tag}")

    bits = []
    for s in SEEDS:
        p = os.path.join(HERE, f"lut_sac_{tag}_grid_s{s}_bitusage.json")
        if os.path.exists(p):
            bits.append(json.load(open(p)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)

    # --- left: paired per-seed return -------------------------------------
    ax = axes[0]
    for i in ok:
        y0 = none[i]["cpu_reference_mean"]
        y1 = grid[i]["cpu_reference_mean"]
        up = y1 >= y0
        ax.plot([0, 1], [y0, y1], color=(ORANGE if up else BLUE), linewidth=1.8,
                alpha=0.75, zorder=2)
        ax.annotate(f"seed {SEEDS[i]}  {y1 - y0:+.0f}", xy=(1.30, y1), fontsize=8.5,
                    color=(ORANGE if up else BLUE), va="center")
    for x, arr, col, side in ((0, none, BLUE, -0.30), (1, grid, ORANGE, 0.14)):
        ys = [arr[i]["cpu_reference_mean"] for i in ok]
        ax.scatter([x] * len(ys), ys, s=90, color=col, edgecolor="white",
                   linewidth=2, zorder=3)
        m = sum(ys) / len(ys)
        ax.plot([x - 0.16, x + 0.16], [m, m], color=col, linewidth=2.6, zorder=4)
        # Mean labels sit OUTSIDE the paired lines (left of `none`, right of `grid`) so
        # they cannot collide with the per-seed deltas, which live at x = 1.30.
        ax.annotate(f"mean {m:.0f}", xy=(x + side, m), fontsize=9, color=col,
                    ha=("right" if side < 0 else "left"), va="center")
    ax.set_xlim(-0.75, 2.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["none\n(17-dim, magnitude-blind)", "grid\n(33-dim, 16 thresholds)"],
                       fontsize=9.5)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Paired by seed — same seed, one flag apart", color=INK, fontsize=12,
                 loc="left", pad=10)

    # --- right: which constants the policy keys off ------------------------
    ax = axes[1]
    if bits:
        vals = [c["value"] for c in bits[0]["per_const"]]
        n = len(vals)
        hs = [sum(b["per_const"][k]["mean_H"] for b in bits) / len(bits)
              for k in range(n)]
        dead = [sum(b["per_const"][k]["dead"] for b in bits) /
                max(sum(b["per_const"][k]["bits"] for b in bits), 1) for k in range(n)]
        ax.bar(range(n), hs, color=ORANGE, width=0.72, zorder=3)
        ax.plot(range(n), dead, color=BLUE, linewidth=1.8, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4,
                label="fraction of its bits that are dead")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{v:+.1f}" for v in vals], fontsize=7.5, rotation=90)
        ax.set_xlabel("constant value (standardised units)", color=MUTED, fontsize=9.5)
        ax.set_ylabel("mean binary entropy of its bits (bits)", color=MUTED,
                      fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
        ax.set_title("Which constants the LUT actually keys off",
                     color=INK, fontsize=12, loc="left", pad=10)

    fig.suptitle(f"exp_c29 — do fixed thresholds cure anchor magnitude blindness?  "
                 f"({LABEL.get(tag, tag)})",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             "FastMHL frozen anchors, hard forward, 10k iters, determinism on.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.008, 0.008,
             "Entropy measured on the states each trained policy visits over 100 "
             "episodes; 1.0 bit = the comparator splits those states evenly.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    out = os.path.join(HERE, f"c29_{tag}_contrast.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
