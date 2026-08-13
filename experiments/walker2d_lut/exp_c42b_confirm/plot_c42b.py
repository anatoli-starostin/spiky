"""exp_c42b — all nine seeds of the fan-in-corrected config, against stock exp_c39.

Run in the SPIKY venv (matplotlib).

LEFT — return curves for all 9 seeds (c42's 0-2 and c42b's 3-8), with stock exp_c39's three
behind them in grey. The question this figure answers is not "is the mean higher" but "does
the BIMODALITY go away" -- every earlier configuration in this line split into a group that
reached the baseline band and a group stranded near 1,000, and the whole point of the
confirmation run is whether that split survives more seeds.

RIGHT — the same nine as a strip against the baseline band, with the stock and the two
structured-init experiments for scale. Individual seeds, because a mean over a bimodal
sample is the one summary that hides what matters here.

Usage:
  python plot_c42b.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C42 = os.path.join(D, "exp_c42_table_init_std")
C39 = os.path.join(D, "exp_c39_mhl_3det_4bkt")

BLUE, ORANGE, GREEN, MUTED2 = "#2a78d6", "#eb6834", "#1f9e5a", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1

STRIP = [
    ("c39 stock\n1/3", [4217.3, 982.3, 890.8], MUTED2),
    ("c40 delay\n2/3", [4302.7, 3481.0, 1162.8], MUTED2),
    ("c41 boundary\n1/3", [4708.0, 1317.6, 959.2], MUTED2),
]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    r = json.load(open(os.path.join(HERE, "results.json")))
    h = {}
    for s in (0, 1, 2):
        h[f"c42 s{s}"] = json.load(
            open(os.path.join(C42, f"mhl_sac_c42_s{s}.json")))["history"]
    for s in (3, 4, 5, 6, 7, 8):
        h[f"c42b s{s}"] = json.load(
            open(os.path.join(HERE, f"mhl_sac_c42b_s{s}.json")))["history"]
    h39 = {s: json.load(open(os.path.join(C39, f"mhl_sac_c39_s{s}.json")))["history"]
           for s in (0, 1, 2)}

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5200)

    # ---- LEFT: all nine curves -------------------------------------------
    ax = axes[0]
    for s in (0, 1, 2):
        e = h39[s]
        ax.plot([p["iter"] for p in e], [p["mjx_return"] for p in e], color=MUTED2,
                linewidth=1.4, alpha=0.7, zorder=3,
                label="stock c39, table std 0.1 (3 seeds)" if s == 0 else None)
    for i, (lab, e) in enumerate(sorted(h.items())):
        ax.plot([p["iter"] for p in e], [p["mjx_return"] for p in e], color=BLUE,
                linewidth=1.8, alpha=0.85, zorder=4,
                label="fan-in corrected (9 seeds)" if i == 0 else None)
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("MJX return (20-ep proxy)", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="upper left")
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(9800, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="right", va="top", fontweight="bold")
    ax.set_title(f"All 9 seeds — takeoff {r['pooled_takeoff']}/{r['pooled_n']}, "
                 f"against stock c39's 1/3", color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: per-seed strip -------------------------------------------
    ax = axes[1]
    groups = STRIP + [(f"c42 + c42b\nfan-in corrected\n"
                       f"{r['pooled_takeoff']}/{r['pooled_n']}",
                       list(r["pooled"].values()), BLUE)]
    for i, (lab, vals, col) in enumerate(groups):
        jit = np.linspace(-0.17, 0.17, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter([i + j for j in jit], vals, s=92, color=col,
                   alpha=0.9, zorder=5, edgecolor="white", linewidth=1.5)
        m = float(np.mean(vals))
        ax.plot([i - 0.3, i + 0.3], [m, m], color=col, linewidth=2.2, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(0, 9), textcoords="offset points",
                    color=col, fontsize=8.8, fontweight="bold", ha="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.0)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Per-seed — the split narrows but does NOT close", color=INK, fontsize=11,
                 loc="left", pad=10)

    fig.suptitle("exp_c42b — confirmation of the fan-in-corrected table init, 9 seeds",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"Pooled {r['pooled_mean']:.0f} ± {r['pooled_sd']:.0f} over n={r['pooled_n']} "
             f"— takeoff {r['pooled_takeoff']}/{r['pooled_n']} "
             f"(Wilson 95% CI [{r['wilson95'][0]:.2f}, {r['wilson95'][1]:.2f}]), "
             f"{r['in_band']}/{r['pooled_n']} inside the baseline band, "
             f"{r['on_plateau']}/{r['pooled_n']} never learned to walk (< 1,500).",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             f"vs the exp_c18 hyperplane baseline: {r['baseline_delta']:+.0f}, unpaired "
             f"Welch se {r['welch_se']:.0f}, |t| "
             f"{abs(r['baseline_delta'])/r['welch_se']:.2f} — at 28,384 params, 101.3% of "
             f"its 28,032. An init change only: no parameter count, architecture or "
             f"hyperparameter differs from stock c39.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c42b_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
