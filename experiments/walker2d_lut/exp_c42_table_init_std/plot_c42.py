"""exp_c42 — fan-in-corrected table init against stock exp_c39.

Run in the SPIKY venv (matplotlib).

LEFT — per-seed return curves, stock c39 against c42. Identical in every respect except the
std of the random table draws, so the comparison is paired: the seed fixes both the init
and the RL stream, and each c42 seed can be read directly against its own c39 counterpart.

RIGHT — the init-only sweep that chose the std, measuring the INITIAL POLICY rather than
the tensor. Everything scales linearly with the std because tanh is near-linear in this
regime, so the interesting content is not the slope but where the stock constant sits on
it: 0.1 puts the initial policy at |mu| ~0.39 with visible tanh saturation, while the
fan-in-corrected value puts it at 0.081 with none.

Usage:
  python plot_c42.py
"""
import json
import math
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
FANIN = 0.1 / math.sqrt(32)


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
    h42 = {s: json.load(open(os.path.join(HERE, f"mhl_sac_c42_s{s}.json")))["history"]
           for s in (0, 1, 2)}
    h39 = {s: json.load(open(os.path.join(C39, f"mhl_sac_c39_s{s}.json")))["history"]
           for s in (0, 1, 2)}
    sweep = json.load(open(os.path.join(HERE, "table_std_sweep.json")))["stds"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax in axes:
        style(ax)

    ax = axes[0]
    ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
    ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
    for s in (0, 1, 2):
        h = h39[s]
        ax.plot([e["iter"] for e in h], [e["mjx_return"] for e in h], color=MUTED2,
                linewidth=1.6, alpha=0.75, zorder=3,
                label="stock c39 (table std 0.1)" if s == 0 else None)
    for s in (0, 1, 2):
        h = h42[s]
        ax.plot([e["iter"] for e in h], [e["mjx_return"] for e in h], color=BLUE,
                linewidth=2.1, alpha=0.95, zorder=4,
                label="c42 (table std 0.0177)" if s == 0 else None)
    ax.set_xlabel("training iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("MJX return (20-ep proxy)", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(300, BASE_M - BASE_SD),
                xytext=(0, -11), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", va="top", fontweight="bold")
    c42_pts = sorted([r["seeds"][k] for k in sorted(r["seeds"])], reverse=True)
    ax.set_title(f"Per-seed returns — stock "
                 f"{sum(v > 3000 for v in C39_FINAL.values())}/3 took off, "
                 f"fan-in-corrected {sum(v > 3000 for v in c42_pts)}/3",
                 color=INK, fontsize=11, loc="left", pad=10)

    ax = axes[1]
    stds = [d["table_init_std"] for d in sweep]
    ax.plot(stds, [d["abs_mu"] for d in sweep], "o-", color=BLUE, linewidth=2.2,
            markersize=5, label="|action| at init")
    ax.plot(stds, [d["smooth_time"] for d in sweep], "s-", color=GREEN, linewidth=2.0,
            markersize=5, label="|Δaction| between timesteps")
    ax.plot(stds, [d["smooth_addr"] for d in sweep], "d-", color=RED, linewidth=2.0,
            markersize=5, label="|Δaction| across neighbouring cells")
    ax.plot(stds, [d["sigma"] for d in sweep], "-", color=INK, linewidth=1.6, alpha=0.6,
            label="policy σ (unchanged — the log-σ bias is untouched)")
    ax.axvline(0.1, color=MUTED2, linewidth=1.2, linestyle=":", alpha=0.9)
    ax.annotate("stock 0.1", xy=(0.093, 0.44), color=MUTED2, fontsize=8.2, ha="right")
    ax.axvline(FANIN, color=BLUE, linewidth=1.2, linestyle=":", alpha=0.9)
    ax.annotate(f"chosen\n0.1/√32", xy=(FANIN * 1.15, 0.30), color=BLUE, fontsize=8.2,
                ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("table init std (log scale)", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK, loc="upper left")
    ax.set_title("Init-only sweep — measuring the POLICY, not the tensor",
                 color=INK, fontsize=11, loc="left", pad=10)

    fig.suptitle("exp_c42 — fan-in-corrected table init (0.1 → 0.1/√tph) vs stock exp_c39",
                 color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c42 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{v:.0f}' for v in c42_pts)}) vs stock c39 2030 ± 1895 "
             f"(4217, 982, 891). Params unchanged at 28,384 — an init change only. "
             f"Parity 84/84.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "A row is read one-hot then SUMMED over 32 tables, so the stock constant put "
             "the initial µ-head output at √32 × 0.1 = 0.57. The correction takes initial "
             "|action| 0.390 → 0.081 and makes it 4.6× smoother, with σ untouched at 0.363.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c42_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
