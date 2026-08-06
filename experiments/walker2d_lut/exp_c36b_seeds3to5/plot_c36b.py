"""exp_c36b — the anchor was not what it looked like.

Run in the SPIKY venv (matplotlib).

LEFT — every configuration in the bisect, at the largest n available for each. Once c36 is
sampled beyond its original three seeds, the "residual gap" that motivated c48-c53 is no
longer statistically distinguishable from zero.

RIGHT — what more seeds did to the two configurations that got them. Both regressed toward
the same place. c36's 3/3 was not extraordinary: under its own pooled takeoff rate of 4/6,
three-for-three has probability 0.30.

Usage:
  python plot_c36b.py
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
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#7b4fbd"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
TAKEOFF = 3000.0
C49 = [2722.6, 802.5, 3173.6]
C53 = [1042.4, 1084.7, 3890.9]
C50_POOL = [4447.2, 3719.6, 1156.3, 1962.4, 2618.0, 4186.8, 1174.3, 3917.1, 1118.8]
C36_OLD = [4527.5, 3933.2, 4277.6]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def stat(v):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    return m, sd, sum(1 for x in v if x >= TAKEOFF)


def main():
    new = [json.load(open(os.path.join(
        HERE, f"bucket_sac_c36_s{s}_cpueval.json")))["cpu_reference_mean"]
        for s in (3, 4, 5)]
    pool = C36_OLD + new

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: the whole bisect, at best available n ----------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.axhline(TAKEOFF, color=MUTED, linewidth=1.0, linestyle=":", zorder=3)
    ax.set_ylim(0, 5200)
    groups = [("c49\nclamped\nn=3", C49, BLUE),
              ("c53\ndetach_hard\nn=3", C53, RED),
              ("c50\nfloor removed\nn=9", C50_POOL, PURPLE),
              ("c36 ANCHOR\npooled\nn=6", pool, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m, sd, tk = stat(vals)
        jit = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 3 else [0] * len(vals)
        ax.scatter([i + j for j in jit], vals, s=88, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.4)
        ax.plot([i - 0.29, i + 0.29], [m, m], color=col, linewidth=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i - 0.29, m), xytext=(-21, -4),
                    textcoords="offset points", color=col, fontsize=9.2,
                    fontweight="bold", ha="center")
        ax.annotate(f"{tk}/{len(vals)}", xy=(i, 140), color=col, fontsize=9.0,
                    ha="center", fontweight="bold")
    ax.annotate("", xy=(3, stat(pool)[0]), xytext=(2, stat(C50_POOL)[0]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.annotate("+891   |t| 1.39\nNOT SIGNIFICANT", xy=(2.5, 3150), xytext=(0, -46),
                textcoords="offset points", color=INK, fontsize=8.8, ha="center",
                fontweight="bold")
    ax.annotate("takeoff 3000", xy=(3.45, 3060), color=MUTED, fontsize=7.6, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("With the anchor properly sampled, the gap the whole\n"
                 "c48–c53 bisect was chasing is no longer significant",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: what more seeds did --------------------------------------
    ax = axes[1]
    pairs = [("c36 anchor", [(3, stat(C36_OLD)), (6, stat(pool))], GREEN),
             ("c50 floor removed", [(3, stat(C50_POOL[:3])), (9, stat(C50_POOL))],
              PURPLE)]
    for lab, pts, col in pairs:
        ns = [p[0] for p in pts]
        ms = [p[1][0] for p in pts]
        ax.plot(ns, ms, "-o", color=col, linewidth=2.4, markersize=9,
                markeredgecolor="white", markeredgewidth=1.6, label=lab, zorder=5)
        for n, (m, sd, tk) in pts:
            ax.annotate(f"{m:.0f}\n{tk}/{n}", xy=(n, m), xytext=(0, 14),
                        textcoords="offset points", color=col, fontsize=8.8,
                        ha="center", fontweight="bold")
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=1.8, zorder=2)
    ax.annotate("c18 hyperplane baseline", xy=(9.3, BASE_M), color=ORANGE, fontsize=8.0,
                ha="right", va="bottom")
    ax.set_xlim(2, 10)
    ax.set_ylim(2000, 5000)
    ax.set_xlabel("seeds pooled", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean 100-episode CPU reference", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.annotate("both regressed toward the same place\nas soon as they were sampled "
                "properly", xy=(0.06, 0.14), xycoords="axes fraction", color=INK,
                fontsize=9.0, ha="left", fontweight="bold")
    ax.set_title("Every mean in this chapter moved DOWN with more seeds",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    m6, sd6, tk6 = stat(pool)
    fig.suptitle("exp_c36b — reproducing the anchor: its 3/3 was a lucky draw",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.068,
             f"New c36 seeds 3/4/5: {new[0]:.0f}, {new[1]:.0f}, {new[2]:.0f} — takeoff "
             f"1/3, against the original 3/3. POOLED c36 n=6: {m6:.0f} ± {sd6:.0f}, "
             f"takeoff {tk6}/6.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.048,
             "vs c50 pooled n=9 (2700 ± 1394): +891, Welch se 641, |t| 1.39. The gap read "
             "−1546 at |t| 3.12 when c50's n=9 was compared against c36's n=3.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.028,
             "Under c36's own pooled takeoff rate of 4/6 = 0.667, a 3/3 result has "
             "probability 0.667³ = 0.30. The original three seeds were not extraordinary "
             "— they were an ordinary run of luck read as a property.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Same code, same config, same protocol as the original: bucket_sac.py + "
             "jax_bucket_lif.py copied unmodified, parity 40/40, 239–242 min/seed.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    out = os.path.join(HERE, "c36b_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
