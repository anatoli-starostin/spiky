"""exp_c31 — the seed-0 parameter teardown, as nine panels.

Run in the SPIKY venv (matplotlib lives there). Reads analysis_seed0.npz, written by
analyze_seed0.py in the mjx venv.

Every panel that has an init marks it, because the whole point is displacement from init
rather than the final shape on its own. Log axes where the quantity spans decades
(`temp_bit` moved 251x; row-visit counts span four orders of magnitude) -- on a linear
axis those panels are a spike and a flat line.

Usage:
  python plot_analysis.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import LogNorm    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"


def style(ax, ygrid=True):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def title(ax, t, sub=None, pad=None, sub_y=1.015):
    """`pad`/`sub_y` are overridden only for the panel carrying a twin top axis, whose
    ticks otherwise sit exactly where the subtitle goes."""
    ax.set_title(t, color=INK, fontsize=10.5, loc="left",
                 pad=(pad if pad is not None else (14 if sub else 8)))
    if sub:
        ax.text(0, sub_y, sub, transform=ax.transAxes, color=MUTED, fontsize=8,
                va="bottom")


def main():
    z = np.load(os.path.join(HERE, "analysis_seed0.npz"), allow_pickle=False)
    CH = [str(c) for c in z["channels"]]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.6), facecolor="white")

    # 1 -------------------------------------------------- table distribution
    ax = axes[0, 0]; style(ax)
    bins = np.linspace(-1.2, 1.2, 90)
    ax.hist(z["table_init"].ravel(), bins=bins, color=MUTED, alpha=0.55,
            label="init (0.1·randn)")
    ax.hist(z["table_final"].ravel(), bins=bins, color=BLUE, alpha=0.65, label="final")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("row value", color=MUTED, fontsize=8.5)
    title(ax, "1 · Table values spread 1.8×",
          "std 0.101 → 0.183; tails reach ±2.2. 71.8% of entries moved >0.01")

    # 2 ----------------------------------------------------- table movement
    ax = axes[0, 1]; style(ax, ygrid=False)
    mv = np.abs(z["table_final"] - z["table_init"]).mean(-1)        # [32, 64]
    im = ax.imshow(mv, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("row (0–63)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("table (0–31)", color=MUTED, fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(colors=MUTED, labelsize=7)
    title(ax, "2 · Which rows actually learned",
          "mean |Δ| per row. Bright = trained; dark = never addressed enough to move")

    # 3 -------------------------------------------------- deployed addressing
    ax = axes[0, 2]; style(ax, ygrid=False)
    v = z["visit"].astype(float)
    im = ax.imshow(np.maximum(v, 0.5), aspect="auto", cmap="viridis",
                   norm=LogNorm(vmin=0.5, vmax=max(v.max(), 1)),
                   interpolation="nearest")
    ax.set_xlabel("row (0–63)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("table (0–31)", color=MUTED, fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(colors=MUTED, labelsize=7)
    title(ax, "3 · The deployed policy uses 31.5% of rows",
          "visits per row, log scale (training touched 78.4%)")

    # 4 ------------------------------------------------------ delay per channel
    ax = axes[1, 0]; style(ax, ygrid=False)
    dc = z["delay_ch"]
    o = np.argsort(dc)
    cols = [ORANGE if dc[i] > 0 else BLUE for i in o]
    ax.barh(range(len(o)), dc[o], color=cols, alpha=0.85)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([CH[i] for i in o], fontsize=7.5)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("mean learned delay (+ = arrives later)", color=MUTED, fontsize=8.5)
    title(ax, "4 · vx pushed latest, left-leg angle earliest",
          "init ALL ZERO. Only 0.4% keep the delay=0 order — but 8.3% of pairs flip")

    # 5 -------------------------------- w per channel VS whether it reaches the decision
    # The two series belong on one panel: the tempting reading of panels 4+5 separately
    # is "vx arrives last AND weighs most, so it casts the decisive vote". Measured, the
    # opposite is true, and only the juxtaposition shows it.
    ax = axes[1, 1]; style(ax, ygrid=False)
    wc, reach = z["w_ch"], z["reach"]
    o = np.argsort(wc)
    y = np.arange(len(o))
    ax.barh(y, wc[o], color=BLUE, alpha=0.75, label="mean |w| (bottom axis)")
    ax.set_yticks(y)
    ax.set_yticklabels([CH[i] for i in o], fontsize=7.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("mean |w| — synaptic drive", color=MUTED, fontsize=8.5)
    ax.set_xlim(0, 1.32)
    ax2 = ax.twiny()
    ax2.scatter(100 * reach[o], y, s=42, color=ORANGE, zorder=6,
                edgecolor="white", linewidth=1.1, label="% reaching the decision")
    ax2.set_xlim(0, 100)
    ax2.tick_params(colors=ORANGE, labelsize=7.5, length=3)
    for sp in ("left", "right", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["top"].set_color(GRID)
    ax.annotate("● = % of firing pairs in which that channel\n"
                "arrives BEFORE the spike (top axis)",
                xy=(0.20, 0.19), xycoords="axes fraction", color=ORANGE,
                fontsize=7.8, ha="left", va="top")
    title(ax, "5 · The largest weight is the most ignored",
          "vx |w| = 1.177 (7.4× its init) but reaches the crossing in 10% of firing "
          "pairs, vs 44% mean", pad=36, sub_y=1.105)

    # 6 ------------------------------------------------------------ deadline L
    ax = axes[1, 2]; style(ax)
    ax.hist(z["L_final"], bins=40, color=BLUE, alpha=0.8)
    ax.axvline(16.0, color=ORANGE, linewidth=2.0, label="init = 16.0")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlabel("deadline L (spike-time units, window = 32)", color=MUTED,
                  fontsize=8.5)
    title(ax, "6 · The deadline barely moved",
          "16.000 → 16.288 ± 0.513 — learning went into w and tau")

    # 7 -------------------------------------------------- per-LUT time constants
    ax = axes[2, 0]; style(ax, ygrid=False)
    series = [("temp_bit", z["temp_bit"], 1.0), ("T_cross", z["t_cross"], 1.0),
              ("tau", z["tau"], 1.3143)]
    for i, (nm, val, init) in enumerate(series):
        ax.scatter(val, np.full(len(val), i) + np.linspace(-.13, .13, len(val)),
                   s=17, color=BLUE, alpha=0.65, edgecolor="none")
        ax.scatter([init], [i], s=110, marker="|", color=ORANGE, linewidth=2.4,
                   zorder=5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in series], fontsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("value (log). orange bar = init", color=MUTED, fontsize=8.5)
    title(ax, "7 · Self-sharpening, with no schedule",
          "temp_bit 251× sharper; T_cross 2.2× sharper; tau 3.7× LONGER memory")

    # 8 ------------------------------------------------- detector bit occupancy
    ax = axes[2, 1]; style(ax)
    occ = z["det_occ"]
    ax.hist(occ, bins=40, color=BLUE, alpha=0.8)
    dead = int((occ == 0).sum()); always = int((occ == 1).sum())
    ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xlabel("fraction of visited states where the bit is set", color=MUTED,
                  fontsize=8.5)
    title(ax, f"8 · {dead} of 192 detectors are dead",
          f"{dead} never fire, {always} always fires, 38 never toggle. "
          f"154 carry real information")

    # 9 ------------------------------------------------ what detectors key off
    ax = axes[2, 2]; style(ax, ygrid=False)
    live = z["live"]
    cnt = np.bincount(z["best_ch"][live], minlength=len(CH))
    o = np.argsort(cnt)
    keep = [i for i in o if cnt[i] > 0]
    ax.barh(range(len(keep)), cnt[keep], color=BLUE, alpha=0.8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([CH[i] for i in keep], fontsize=7.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("live detectors whose bit correlates most with this channel",
                  color=MUTED, fontsize=8.5)
    # NOT "asymmetric gait": left chain 56 vs right 47 is z = 0.89 under a fair split,
    # i.e. nothing. The angle-vs-velocity split IS real — 91 to 38, a 2.4x tilt.
    title(ax, "9 · Detectors key off POSITION, not velocity",
          "angles 91 · velocities 38 · other 25.  L-vs-R 56:47 = noise")

    fig.suptitle("exp_c31 — inside the seed-0 PureLIF actor (CPU-ref 4262.1, "
                 "31,392 params: 6,816 front-end + 24,576 table)",
                 color=INK, fontsize=14, x=0.006, ha="left", y=0.996)
    fig.text(0.006, 0.005,
             "All statistics over 4,096 states sampled from 24,000 visited by the "
             "DEPLOYED policy (mode=hard) in MJX — not from the replay buffer, which "
             "is contaminated by warmup and by every earlier policy. "
             "Every parameter is compared against its own reproduced init (seed 0).",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.016, 1, 0.978))
    out = os.path.join(HERE, "c31_seed0_teardown.png")
    fig.savefig(out, dpi=155, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
