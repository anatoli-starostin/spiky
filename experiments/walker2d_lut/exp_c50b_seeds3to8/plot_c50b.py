"""exp_c50b — what six more seeds did to the exp_c50 reading.

Run in the SPIKY venv (matplotlib).

LEFT — the pooled verdict. c50 at n=3 read 3108 +/- 1729 with 2/3 takeoff, and the gap to
c36 was |t| 1.12, not distinguishable from zero. At n=9 it reads 2700 +/- 1394 with 4/9,
and the gap to c36 is |t| 3.12. The extra seeds did not confirm the recovery; they
established that a residual gap is real.

RIGHT — the delays, which is the part that did NOT change. All six new seeds land on c36's
distribution as precisely as the first three did: 39-42% negative, span ~-10..+12, nothing
dead. So the clamp mechanism is settled and it is simply not the whole story.

Usage:
  python plot_c50b.py
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
C50 = os.path.join(D, "exp_c50_no_delay_clamp")
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#7b4fbd"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C36_S = [4527.5, 3933.2, 4277.6]
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
    new = [r["seeds"][k] for k in sorted(r["seeds"], key=int)]
    prior = [r["prior_seeds"][k] for k in sorted(r["prior_seeds"], key=int)]
    pool = new + prior

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white")
    for ax in axes:
        style(ax)

    # ---- LEFT: n=3 against n=9 --------------------------------------------
    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    ax.set_ylim(0, 5200)
    groups = [("c49\nclamped\nn=3", C49_S, BLUE),
              ("c50\nfirst 3 seeds\nn=3", prior, PURPLE),
              ("c50 POOLED\nseeds 0–8\nn=9", pool, RED),
              ("c36 ORIGINAL\nold module\nn=3", C36_S, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        jit = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 3 else [0] * len(vals)
        ax.scatter([i + j for j in jit], vals, s=95, color=col, alpha=0.9, zorder=5,
                   edgecolor="white", linewidth=1.5)
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, linewidth=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i - 0.28, m), xytext=(-21, -4),
                    textcoords="offset points", color=col, fontsize=9.2,
                    fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= TAKEOFF)}/{len(vals)}", xy=(i, 150),
                    color=col, fontsize=9.0, ha="center", fontweight="bold")
    ax.axhline(TAKEOFF, color=MUTED, linewidth=1.0, linestyle=":", zorder=3)
    ax.annotate("takeoff 3000", xy=(3.45, 3050), color=MUTED, fontsize=7.8, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Six more seeds moved the mean DOWN and made the\n"
                 "gap to c36 significant: |t| 1.12 at n=3 → 3.12 at n=9",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- RIGHT: the delays, unchanged -------------------------------------
    ax = axes[1]
    d36 = np.concatenate([np.load(os.path.join(
        C36, f"bucket_sac_c36_s{s}_actor.npz"))["delay"].ravel() for s in (0, 1, 2)])
    d_new = np.concatenate([np.load(os.path.join(
        HERE, f"mhl_sac_c50_s{s}_actor.npz"))["delay"].ravel()
        for s in (3, 4, 5, 6, 7, 8)])
    bins = np.linspace(-13, 14, 80)
    ax.hist(d36, bins=bins, color=GREEN, alpha=0.62,
            label="c36 — no clamp (old module), 3 seeds", density=True)
    ax.hist(d_new, bins=bins, color=RED, alpha=0.62,
            label="c50 seeds 3–8 — floor removed, 6 seeds", density=True)
    ax.set_yscale("log")
    ax.axvline(0.0, color=INK, linewidth=1.5, linestyle="--", alpha=0.8)
    dd = r["delays"]
    neg = [dd[k]["pct_negative"] for k in sorted(dd, key=int)]
    ax.annotate(f"new seeds: {min(neg):.1f}–{max(neg):.1f}% negative,\n"
                f"0.0% dead, 0.00% on the retained cap\n"
                f"c36: {100*(d36 < 0).mean():.1f}% negative",
                xy=(0.03, 0.70), xycoords="axes fraction", color=INK, fontsize=8.6,
                ha="left", fontweight="bold")
    ax.set_xlabel("learned delay", color=MUTED, fontsize=9.5)
    ax.set_ylabel("density, log scale", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="upper right")
    ax.set_title("The delay fix held on every one of the six —\n"
                 "the mechanism is settled, it is just not the whole story",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c50b — six more seeds of the floor-removed clamp",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.068,
             f"new seeds 3–8: {r['mean']:.0f} ± {r['sd']:.0f}, takeoff "
             f"{r['takeoff']}/6 ({', '.join(f'{v:.0f}' for v in sorted(new, reverse=True))}). "
             f"POOLED n=9: {r['pooled_mean']:.0f} ± {r['pooled_sd']:.0f}, takeoff "
             f"{r['pooled_takeoff']}/9.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.048,
             "vs c36 (4246 ± 298, n=3): −1546, Welch se 496, |t| 3.12 — the residual gap "
             "is real. vs c49 (2233 ± 1259, n=3): +467, |t| 0.54 — the improvement over "
             "the clamped run is NOT established.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.028,
             "This is the c42b lesson repeating: c50 at n=3 showed 2/3 and 3108; the same "
             "configuration at n=9 shows 4/9 and 2700. A takeoff rate near one-half cannot "
             "be read from three seeds.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "c36's own n=3 is not decisive either: under c50's measured 4/9 takeoff rate, "
             "a 3/3 result has probability 0.44³ ≈ 8.8%.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    out = os.path.join(HERE, "c50b_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
