"""exp_c50 — does removing the delay clamp's floor recover c36?

Run in the SPIKY venv (matplotlib).

LEFT — the verdict. c48 (frozen temps), c49 (unfrozen), c50 (floor removed), against c36's
original 4246. c48 and c49 between them exonerated the two init settings and the temperature
freeze; the delay clamp is the remaining structural difference with direct evidence behind
it, and this is the run that tests it.

RIGHT — the delays themselves, which is what the experiment actually manipulates. c49's
pile up on the clamp floor (dead in both value and gradient); c36's, unclamped, spread
symmetrically across [-10, +13]. Where c50's land is the mechanism, independent of whether
the return follows.

Usage:
  python plot_c50.py
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
C49 = os.path.join(D, "exp_c49_c48_unfrozen_temp")
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#7b4fbd"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C36_S = [4527.5, 3933.2, 4277.6]
C48_S = [3212.5, 1323.0, 3288.9]
C49_S = [2722.6, 802.5, 3173.6]
TAKEOFF = 3000.0


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
    c50 = [r["seeds"][k] for k in sorted(r["seeds"])]
    dd = r["delays"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: the verdict ------------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.set_ylim(0, 5200)
    groups = [("c48\ntemps frozen\ndelays CLAMPED", C48_S, RED),
              ("c49\ntemps unfrozen\ndelays CLAMPED", C49_S, BLUE),
              ("c50\ntemps unfrozen\nfloor REMOVED", c50, PURPLE),
              ("c36 ORIGINAL\nold module\nno clamp at all", C36_S, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=110, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i - 0.26, m), xytext=(-20, -4),
                    textcoords="offset points",
                    color=col, fontsize=9.2, fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= TAKEOFF)}/3", xy=(i, 160), color=col,
                    fontsize=9.0, ha="center", fontweight="bold")
    m49, m50 = sum(C49_S) / 3, sum(c50) / 3
    ax.annotate("", xy=(2, m50), xytext=(1, m49),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate(f"{m50 - m49:+.0f}", xy=(1.5, (m49 + m50) / 2), xytext=(0, -24),
                textcoords="offset points", color=INK, fontsize=9.0, ha="center",
                fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Removing the clamp's non-negativity floor —\n"
                 "the last structural difference from c36",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: the delay distributions -----------------------------------
    ax = axes[1]
    d36 = np.concatenate([np.load(os.path.join(
        C36, f"bucket_sac_c36_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    d49 = np.concatenate([np.load(os.path.join(
        C49, f"mhl_sac_c49_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    d50 = np.concatenate([np.load(os.path.join(
        HERE, f"mhl_sac_c50_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    lo = min(-14.0, float(d50.min()) - 1.0)
    hi = max(16.0, float(d50.max()) + 1.0)
    bins = np.linspace(lo, hi, 80)
    ax.hist(d36, bins=bins, color=GREEN, alpha=0.62, label="c36 — no clamp (old module)")
    ax.hist(d49, bins=bins, color=BLUE, alpha=0.62, label="c49 — clamped to [0, 32]")
    ax.hist(d50, bins=bins, color=PURPLE, alpha=0.62,
            label="c50 — floor removed, cap kept")
    ax.set_yscale("log")
    ax.axvline(0.0, color=INK, linewidth=1.6, linestyle="--", alpha=0.8)
    ax.annotate("the old floor", xy=(0.4, 2.2), color=INK, fontsize=8.4, ha="left",
                fontweight="bold")
    pn = 100 * (d50 < 0).mean()
    pd = 100 * (np.abs(d50) <= 1e-6).mean()
    ax.annotate(f"c50: {pn:.0f}% negative, {pd:.1f}% still\n"
                f"exactly at 0, span {d50.min():.1f} … {d50.max():.1f}",
                xy=(0.03, 0.72), xycoords="axes fraction", color=PURPLE, fontsize=8.4,
                ha="left", fontweight="bold")
    ax.annotate(f"c49: {100*(d49 <= 0).mean():.0f}% dead on the floor\n"
                f"c36: {100*(d36 < 0).mean():.0f}% negative, −10 … +13",
                xy=(0.03, 0.58), xycoords="axes fraction", color=MUTED, fontsize=8.4,
                ha="left", fontweight="bold")
    ax.set_xlabel("learned delay", color=MUTED, fontsize=9.5)
    ax.set_ylabel("count, log scale (3 seeds pooled, 6,528 delays)",
                  color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="upper right")
    ax.set_title("Where the delays actually went", color=INK, fontsize=10.5,
                 loc="left", pad=10)

    fig.suptitle("exp_c50 — the delay clamp's non-negativity floor removed",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.068,
             f"c50 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in sorted(c50, reverse=True))}), takeoff "
             f"{r['takeoff']}/3, against c49's 2233 ± 1259 (1/3) and c36's 4246 ± 298 "
             f"(3/3).",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.048,
             "Two of three seeds land within 5% of c36 (4447 vs 4527, 3720 vs 3933); the "
             "whole remaining gap is seed 2, which never took off at all.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.028,
             "Everything else is c49 verbatim: 1 head × 128 tables × 1 detector × 16 "
             "buckets, per-table ladders, stock 0.1 table init, delay_init_std=0, "
             "temperatures trainable, SORT_FORM='rank'. Parity 105/105.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Only the LOWER bound was dropped — clamp(delay, −inf, t_window). The upper "
             "cap is retained because it holds arrivals inside [·, 2·t_window] so "
             "exp(a/tau) stays float32-safe in the reference's cumsum membrane.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    out = os.path.join(HERE, "c50_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
