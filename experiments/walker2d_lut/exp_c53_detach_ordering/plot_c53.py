"""exp_c53 — the address surrogate was fixed, and the return did not follow.

Run in the SPIKY venv (matplotlib).

LEFT — returns. c53 against c50 on the same three seeds, c50 pooled to n=9, and the c36
anchor. The detached-hard crossing does not improve the return; the point estimate is
lower, and at n=3 that difference is not resolvable either way.

RIGHT — the mechanism that was supposed to help, measured on the TRAINED weights. The soft
partition is the entire address-gradient path, and under the soft crossing its mode
disagrees with the cell actually read 59% of the time. detach_hard fixes that almost
completely -- and the seed where it is MOST faithful is the one that failed.

Usage:
  python plot_c53.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#7b4fbd"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C36_S = [4527.5, 3933.2, 4277.6]
C49_S = [2722.6, 802.5, 3173.6]
C50_S = [4447.2, 3719.6, 1156.3]
C50_POOL = [4447.2, 3719.6, 1156.3, 1962.4, 2618.0, 4186.8, 1174.3, 3917.1, 1118.8]
TAKEOFF = 3000.0
# argmax(soft partition) == hard digit, measured on the trained checkpoints.
FAITH = [("c50 s0\nsoft crossing\n(took off, 4447)", 40.61, PURPLE),
         ("c53 s2\ndetach_hard\n(took off, 3891)", 95.38, RED),
         ("c53 s0\ndetach_hard\n(FAILED, 1042)", 97.24, RED)]


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
    c53 = [r["seeds"][k] for k in sorted(r["seeds"], key=int)]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: returns ----------------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.axhline(TAKEOFF, color=MUTED, linewidth=1.0, linestyle=":", zorder=3)
    ax.set_ylim(0, 5200)
    groups = [("c49\nclamped\nn=3", C49_S, BLUE),
              ("c50 soft\nsame seeds\nn=3", C50_S, PURPLE),
              ("c50 soft\nPOOLED\nn=9", C50_POOL, "#8e6bbf"),
              ("c53\ndetach_hard\nn=3", c53, RED),
              ("c36\nanchor\nn=3", C36_S, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        jit = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 3 else [0] * len(vals)
        ax.scatter([i + j for j in jit], vals, s=88, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.4)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.5, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i - 0.28, m), xytext=(-20, -4),
                    textcoords="offset points", color=col, fontsize=9.0,
                    fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= TAKEOFF)}/{len(vals)}", xy=(i, 140),
                    color=col, fontsize=8.8, ha="center", fontweight="bold")
    ax.annotate("takeoff 3000", xy=(4.45, 3060), color=MUTED, fontsize=7.6, ha="right")
    ax.set_xticks(range(5))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.9)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Detaching the crossing did not raise the return\n"
                 "(−1102 vs c50 same seeds, |t| 0.80 — n=3 cannot resolve it)",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: the surrogate was fixed, and it did not matter -------------
    ax = axes[1]
    xs = np.arange(len(FAITH))
    ax.bar(xs, [f[1] for f in FAITH], color=[f[2] for f in FAITH], alpha=0.85,
           width=0.6, edgecolor="white", linewidth=1.5)
    for i, (lab, v, col) in enumerate(FAITH):
        ax.annotate(f"{v:.1f}%", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                    color=col, fontsize=11, fontweight="bold", ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f[0] for f in FAITH], fontsize=8.2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("argmax(soft partition) == the cell actually read, %",
                  color=MUTED, fontsize=9.5)
    ax.annotate("the address gradient pointed at the\nwrong cell 59% of the time —\n"
                "and fixing that changed nothing",
                xy=(0.30, 0.52), xycoords="axes fraction", color=INK, fontsize=8.8,
                ha="left", fontweight="bold")
    ax.annotate("the MOST faithful seed\nis the one that FAILED",
                xy=(2, 97.24), xytext=(-6, -46), textcoords="offset points",
                color=INK, fontsize=8.4, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_title("The surrogate faithfulness was the point — it was fixed,\n"
                 "and the return did not follow",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c53 — the detached-hard crossing (t_soft := t_hard)",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.068,
             f"c53 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in sorted(c53, reverse=True))}), takeoff "
             f"{r['takeoff']}/3. vs c50 same seeds −1102 (|t| 0.80); vs c50 pooled n=9 "
             f"−694 (|t| 0.66); vs c36 −2240 (|t| 2.34). Parity 122/122.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.048,
             "COST: 2,432 parameters stop learning — w_raw (2,176) and tau_raw (128) "
             "reach the output only through the membrane V, which now merely picks a "
             "detached index; log_T_cross (128) goes unused. 36% of the front-end.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.028,
             "BENEFIT: 12.9% faster end to end — 32.9 min/seed against c50's 37.2, three "
             "seeds co-resident both times. The saving is the cumprod-survival VJP.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Everything else is c50 verbatim: 1 head × 128 tables × 1 detector × 16 "
             "buckets, per-table ladders, stock 0.1 table init, delay_init_std=0, delay "
             "clamp floor removed, SORT_FORM='rank', seeds 0/1/2.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    out = os.path.join(HERE, "c53_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
