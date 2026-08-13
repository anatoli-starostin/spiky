"""exp_c41 — structured BOUNDARY init against stock exp_c39 (and c40's delay offset).

Run in the SPIKY venv (matplotlib).

LEFT — per-seed return curves, stock c39 against c41. Identical in every respect except a
per-detector offset on the boundary base, so the comparison is as controlled as this
chapter gets: same architecture, same 28,384 parameters, same seeds, same trainer, same
recipe.

RIGHT — the init-only sweep that chose the offset, and the reason to believe this variant
where c40's was not worth believing. Three things happen at once as the offset grows:
EXCESS agreement over the independence floor falls to zero (the detectors become
statistically independent), detector entropy RISES, and effective cells per table rises.
Meanwhile the no-spike rate is flat at 0.483 across the entire sweep -- a boundary shift
cannot change whether a detector fires, and that is measured rather than asserted. c40's
delay offset moved none of these in the right direction and killed 64 of 96 detectors at
the far end.

Usage:
  python plot_c41.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C39 = os.path.join(D, "exp_c39_mhl_3det_4bkt")

BLUE, ORANGE, RED, GREEN, MUTED2 = "#2a78d6", "#eb6834", "#c0392b", "#1f9e5a", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C39_FINAL = {0: 890.8, 1: 982.3, 2: 4217.3}
CHOSEN = 6.0


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
    h41 = {s: json.load(open(os.path.join(HERE, f"mhl_sac_c41_s{s}.json")))["history"]
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
        h = h41[s]
        ax.plot([e["iter"] for e in h], [e["mjx_return"] for e in h], color=BLUE,
                linewidth=2.1, alpha=0.95, zorder=4,
                label="c41 boundary offset 6" if s == 0 else None)
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("MJX return (20-ep proxy)", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(300, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    c41_pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    ax.set_title(f"Per-seed returns — stock "
                 f"{sum(v > 3000 for v in C39_FINAL.values())}/3 took off, "
                 f"boundary-offset {sum(v > 3000 for v in c41_pts)}/3",
                 color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: the init sweep ------------------------------------------
    ax = axes[1]
    offs = [d["offset"] for d in sweep]
    ax.axhline(0.0, color=GRID, linewidth=1.2)
    ax.plot(offs, [d["excess"] for d in sweep], "o-", color=RED, linewidth=2.2,
            markersize=5, label="EXCESS agreement (want 0)")
    ax.plot(offs, [d["det_entropy"] / 2.0 for d in sweep], "s-", color=BLUE,
            linewidth=2.0, markersize=5, label="detector entropy (bits / 2)")
    ax.plot(offs, [d["eff_cells"] / 12.0 for d in sweep], "d-", color=GREEN,
            linewidth=2.0, markersize=5, label="effective cells / 12")
    ax.plot(offs, [d["nospike"] for d in sweep], "-", color=INK, linewidth=1.6,
            alpha=0.65, label="no-spike rate (FLAT — firing untouched)")
    ax.axvline(CHOSEN, color=BLUE, linewidth=1.2, linestyle=":", alpha=0.85)
    ax.annotate(f"chosen: {CHOSEN:.0f}", xy=(CHOSEN + 0.4, 0.92), color=BLUE,
                fontsize=8.4, ha="left")
    ax.set_xlabel("per-detector boundary offset", color=MUTED, fontsize=9.5)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK, loc="upper left")
    ax.set_title("The init-only sweep — everything improves at once",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c41 — structured per-detector BOUNDARY init vs stock exp_c39",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c41 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c41_pts)}) vs stock c39 2030 ± 1895 "
             f"(4217, 982, 891) and c40 delay-offset 2982 ± 1628. Params unchanged at "
             f"28,384 — an init change only. Parity 85/85.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Takeoff count is unchanged at 1/3, but unlike c40 the seeds did NOT "
             "reshuffle: the same seed wins and ALL THREE improve (891→959, 982→1318, "
             "4217→4708). Seed 2's 4708 is the best single seed in the whole LIF line.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c41_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
