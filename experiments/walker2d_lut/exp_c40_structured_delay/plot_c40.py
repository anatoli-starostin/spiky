"""exp_c40 — structured delay init against stock exp_c39.

Run in the SPIKY venv (matplotlib).

LEFT — per-seed return curves, both configurations overlaid. Identical in every respect
except the per-detector delay bias, so the comparison is as controlled as this chapter
gets: same architecture, same parameter count, same seeds, same trainer, same recipe.

RIGHT — the offset sweep that chose the value, measured at INIT with no training. It is
included because it predicted the outcome before any GPU was spent, and because it shows
WHY the intervention behaves as it does: a uniform per-detector delay bias does not
decorrelate the detectors, it translates them later, and past a few units the later
detectors stop firing at all and fold into the last bucket -- a constant digit, which drives
agreement back UP while entropy collapses.

Usage:
  python plot_c40.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C39 = os.path.join(D, "exp_c39_mhl_3det_4bkt")

BLUE, ORANGE, RED, MUTED2 = "#2a78d6", "#eb6834", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C39_FINAL = {0: 890.8, 1: 982.3, 2: 4217.3}


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
    h40 = {s: json.load(open(os.path.join(HERE, f"mhl_sac_c40_s{s}.json")))["history"]
           for s in (0, 1, 2)}
    h39 = {s: json.load(open(os.path.join(C39, f"mhl_sac_c39_s{s}.json")))["history"]
           for s in (0, 1, 2)}
    sweep = json.load(open(os.path.join(HERE, "offset_sweep.json")))["offsets"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax in axes:
        style(ax)

    # ---- LEFT: return curves --------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    for s in (0, 1, 2):
        h = h39[s]
        ax.plot([e["iter"] for e in h], [e["mjx_return"] for e in h], color=MUTED2,
                linewidth=1.6, alpha=0.75, zorder=3,
                label="stock c39 (offset 0)" if s == 0 else None)
    for s in (0, 1, 2):
        h = h40[s]
        ax.plot([e["iter"] for e in h], [e["mjx_return"] for e in h], color=BLUE,
                linewidth=2.1, alpha=0.95, zorder=4,
                label="c40 structured (offset 2)" if s == 0 else None)
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("MJX return (20-ep proxy)", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(300, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    c40_pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    ax.set_title(f"Per-seed returns — stock {sum(v > 3000 for v in C39_FINAL.values())}/3 "
                 f"took off, structured {sum(v > 3000 for v in c40_pts)}/3",
                 color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: the offset sweep ----------------------------------------
    ax = axes[1]
    offs = [d["offset"] for d in sweep]
    ax.plot(offs, [d["pair_agreement"] for d in sweep], "o-", color=RED, linewidth=2.0,
            markersize=5, label="pairwise digit agreement")
    ax.plot(offs, [d["indep_floor"] for d in sweep], "--", color=RED, linewidth=1.4,
            alpha=0.6, label="independence floor (same marginals)")
    ax.plot(offs, [d["det_entropy"] / 2.0 for d in sweep], "s-", color=BLUE,
            linewidth=2.0, markersize=5, label="detector entropy (bits / 2)")
    ax.plot(offs, [d["dead"] / 96.0 for d in sweep], "^-", color=INK, linewidth=1.6,
            markersize=5, alpha=0.8, label="dead detectors (fraction of 96)")
    ax.axvline(2.0, color=BLUE, linewidth=1.2, linestyle=":", alpha=0.8)
    ax.annotate("chosen: 2.0", xy=(2.2, 0.94), color=BLUE, fontsize=8.4, ha="left")
    ax.set_xlabel("per-detector delay offset", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.0, labelcolor=INK, loc="center right")
    ax.set_title("The init-only sweep, measured before any GPU",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c40 — structured per-detector delay init vs stock exp_c39",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c40 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c40_pts)}) vs stock c39 2030 ± 1895 "
             f"(4217, 982, 891). Params unchanged at 28,384 — an init change only. "
             f"Parity with the structured init on: 83/83.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Why a uniform per-detector delay bias cannot decorrelate: it TRANSLATES each "
             "detector later rather than changing what it reads, so past a few units the "
             "later detectors stop firing and fold into the last bucket.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c40_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
