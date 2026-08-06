"""exp_c47 — the per-table control for c46, and a fan-in-corrected re-run of c36.

Run in the SPIKY venv (matplotlib).

LEFT — paired by seed against c46. Same shape, same everything, the only difference being
per-table vs one shared ladder. Lines connect the same seed, which is a genuine pairing
because the seed fixes both the init and the RL stream.

RIGHT — the three configurations that share this exact 128 × 1 × 16 shape, differing only
in the ladder and the table init. It shows both results at once: restoring per-table
ladders recovers most of the shared-ladder loss, and the fan-in table init did NOT reproduce
c36 -- it landed below it, opposite to the direction predicted for the configuration where
the stock init was most over-scaled.

Usage:
  python plot_c47.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, MUTED2 = "#2a78d6", "#eb6834", "#1f9e5a", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
BASE_M, BASE_SD = 4308.0, 500.1
C46 = {0: 681.2, 1: 585.8, 2: 1577.5}
C36 = [4528.0, 4181.0, 4029.3]          # exp_c36 per-seed, stock table init


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
    c47 = {int(k): v for k, v in r["seeds"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor="white")
    for ax in axes:
        style(ax)
        ax.axhspan(BASE_M - BASE_SD, BASE_M + BASE_SD, color=ORANGE, alpha=0.13, zorder=1)
        ax.axhline(BASE_M, color=ORANGE, linewidth=2.0, zorder=2)
        ax.set_ylim(0, 5000)

    # ---- LEFT: paired against c46 ---------------------------------------
    ax = axes[0]
    for s in (0, 1, 2):
        ax.plot([0, 1], [C46[s], c47[s]], color=MUTED2, linewidth=1.6, alpha=0.85,
                zorder=3)
        ax.annotate(f"s{s}", xy=(1.05, c47[s]), color=MUTED, fontsize=8.0,
                    va="center", ha="left")
    ax.scatter([0] * 3, list(C46.values()), s=115, color=RED, alpha=0.92, zorder=5,
               edgecolor="white", linewidth=1.7)
    ax.scatter([1] * 3, list(c47.values()), s=115, color=BLUE, alpha=0.92, zorder=5,
               edgecolor="white", linewidth=1.7)
    for i, (m, col) in enumerate(((948.2, RED), (r["mean"], BLUE))):
        ax.plot([i - 0.22, i + 0.22], [m, m], color=col, linewidth=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m),
                    xytext=(-40 if i == 0 else 0, -4 if i == 0 else 12),
                    textcoords="offset points", color=col, fontsize=9.4,
                    fontweight="bold", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["c46\nONE shared ladder\n29,328 par · 0/3",
                        "c47\nper-table ladders\n31,360 par · 2/3"], fontsize=8.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.annotate("exp_c18 baseline 4308 ± 500", xy=(-0.45, BASE_M + BASE_SD),
                xytext=(0, 6), textcoords="offset points", color=ORANGE,
                fontsize=8.2, ha="left", fontweight="bold")
    ax.set_title("Paired by seed — restoring per-table ladders lifted all three",
                 color=INK, fontsize=11, loc="left", pad=10)

    # ---- RIGHT: the three same-shape configs -----------------------------
    ax = axes[1]
    groups = [("c46\nSHARED ladder\nfan-in init", list(C46.values()), RED),
              ("c47\nper-table\nfan-in init", list(c47.values()), BLUE),
              ("c36\nper-table\nSTOCK init", C36, GREEN)]
    for i, (lab, vals, col) in enumerate(groups):
        m = sum(vals) / len(vals)
        ax.scatter([i] * len(vals), vals, s=110, color=col, alpha=0.92, zorder=5,
                   edgecolor="white", linewidth=1.7)
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, linewidth=2.4, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(36, -4), textcoords="offset points",
                    color=col, fontsize=9.2, fontweight="bold", ha="center")
        ax.annotate(f"{sum(1 for v in vals if v >= 3000)}/3", xy=(i, 150), color=col,
                    fontsize=9.0, ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.4)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("100-ep CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("All three are 128 tables × 1 detector × 16 buckets.\n"
                 "Only the ladder and the table init differ.",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    fig.suptitle("exp_c47 — 128 × 1 × 16 with per-table ladders and the fan-in table init",
                 color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             f"c47 {r['mean']:.0f} ± {r['sd']:.0f} "
             f"({', '.join(f'{c47[s]:.0f}' for s in (1, 2, 0))}), takeoff 2/3, 31,360 "
             f"params. (a) vs c46 +1835, |t| 1.74 — and PAIRED all three seeds rose "
             f"(+95, +3335, +2076), takeoff 0/3 → 2/3.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "(b) vs c36 −1463, |t| 1.43: the fan-in init did NOT reproduce c36 at "
             "tph=128, where the stock init was most over-scaled. Not resolvable at n=3, "
             "but the direction is opposite to the prediction.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c47_result.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
