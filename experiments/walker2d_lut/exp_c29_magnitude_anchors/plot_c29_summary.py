"""exp_c29 — the whole 2x2, in one figure.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

WHAT THE 2x2 IS. Four waves = two capacity partitions (nap6/tph64, nap5/tph128) crossed
with two anchor samplers (balanced, canonical_full_coverage). Every cell holds the same
49,152 learnable params, so nothing here is a capacity comparison -- only partition and
sampler move.

TWO PANELS, and the left one is the result. Left: the paired `grid - none` difference for
all twelve seeds, grouped by wave, against a zero line. Paired because the two arms share
a seed and the seed spread is several hundred; the difference is the only quantity with
any power at n=3. Grouping by wave is what exposes the interaction -- the sign is a
property of the PARTITION, not of the treatment.

Right: the same data unpaired, as absolute returns per arm. It answers the question the
left panel provokes: is the swing coming from both arms or one? `none` sits in a 167-point
band across all four waves while `grid` ranges over 1,116. The baseline is stable; every
bit of the volatility is on the treated arm.

Colors are categorical slots 1 and 2 of the validated default palette (blue #2a78d6,
orange #eb6834), matching plot_c29.py so the per-wave figures and this summary read as one
set. Sign convention also matches: orange = grid ahead, blue = none ahead.

Usage:
  python plot_c29_summary.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"
SEEDS = (0, 1, 2)
# (tag, two-line x label) in the order the waves ran.
WAVES = [("c29", "wave 1\nnap6/tph64\nbalanced"),
         ("c29c", "wave 2\nnap6/tph64\ncanonical"),
         ("c29m", "wave 3\nnap5/tph128\nbalanced"),
         ("c29mc", "wave 4\nnap5/tph128\ncanonical")]


def ev(tag, arm, s):
    p = os.path.join(HERE, f"lut_sac_{tag}_{arm}_s{s}_cpueval.json")
    return json.load(open(p))["cpu_reference_mean"] if os.path.exists(p) else None


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    return m, sd


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    data = {t: {a: [ev(t, a, s) for s in SEEDS] for a in ("none", "grid")}
            for t, _ in WAVES}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor="white")
    for ax in axes:
        style(ax)

    # --- left: the paired contrast, grouped by wave --------------------------
    ax = axes[0]
    ax.axhline(0, color=MUTED, linewidth=1.1, zorder=2)
    for i, (tag, _) in enumerate(WAVES):
        d = [g - n for g, n in zip(data[tag]["grid"], data[tag]["none"])
             if g is not None and n is not None]
        for j, v in enumerate(d):
            ax.scatter(i + (j - 1) * 0.13, v, s=95,
                       color=(ORANGE if v >= 0 else BLUE),
                       edgecolor="white", linewidth=1.8, zorder=4)
        m, _ = mean_sd(d)
        ax.plot([i - 0.30, i + 0.30], [m, m], color=INK, linewidth=2.4, zorder=5)
        # The wave mean is annotated because the eye cannot average three points when
        # one of them is an outlier -- which is exactly wave 3's situation.
        ax.annotate(f"{m:+.0f}", xy=(i + 0.34, m), fontsize=9.5, color=INK, va="center")
    # Wave 3 seed 2 is the single collapsed policy in the whole experiment; unlabelled it
    # reads as a plotting error rather than as a real, explainable run.
    ax.annotate("grid s2 collapsed\n(0/100 full-length)", xy=(2 + 0.13, -3257),
                xytext=(1.15, -2650), fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.0))
    # The direction labels live in a reserved gutter left of wave 1, not floating in the
    # plot area -- at wave 1 the data sits at +586 and the mean bar at +249, so anything
    # placed near the zero line there lands on top of them.
    ax.set_xlim(-0.95, 3.55)
    ax.set_xticks(range(len(WAVES)))
    ax.set_xticklabels([lb for _, lb in WAVES], fontsize=9)
    ax.set_ylabel("grid − none, paired by seed", color=MUTED, fontsize=9.5)
    # Descriptive, not causal. Wave MEANS are positive at nap6 and negative at nap5, but
    # the per-seed signs do not follow the partition -- wave 3 is 2/3 positive and wave 4
    # is 0/3, at the same partition. Claiming the partition sets the sign would read a
    # mechanism into four means backed by twelve very noisy seeds.
    ax.set_title("Wave means: positive at nap6/tph64, negative at nap5/tph128",
                 color=INK, fontsize=12, loc="left", pad=10)
    ax.annotate("grid\nahead", xy=(-0.90, 350), fontsize=8.5, color=ORANGE,
                va="center", linespacing=1.3)
    ax.annotate("none\nahead", xy=(-0.90, -350), fontsize=8.5, color=BLUE,
                va="center", linespacing=1.3)

    # --- right: absolute returns, per arm ------------------------------------
    ax = axes[1]
    for arm, col, off in (("none", BLUE, -0.11), ("grid", ORANGE, 0.11)):
        ms = []
        for i, (tag, _) in enumerate(WAVES):
            vals = [v for v in data[tag][arm] if v is not None]
            ax.scatter([i + off] * len(vals), vals, s=52, color=col, alpha=0.55,
                       edgecolor="white", linewidth=1.2, zorder=3)
            ms.append(sum(vals) / len(vals))
        ax.plot([i + off for i in range(len(WAVES))], ms, color=col, linewidth=2.2,
                marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.6,
                zorder=4, label=f"{arm}  (range {max(ms) - min(ms):.0f})")
    ax.set_xticks(range(len(WAVES)))
    ax.set_xticklabels([lb for _, lb in WAVES], fontsize=9)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_title("The baseline is flat; all the volatility is on grid",
                 color=INK, fontsize=12, loc="left", pad=10)

    fig.suptitle("exp_c29 — fixed thresholds on a magnitude-blind LUT walker2d: "
                 "four waves, 24 runs, param-matched at 49,152",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             "FastMHL frozen anchors, hard forward, 10k iters, determinism on. "
             "Every cell: tph x 2^nap x 12 = 49,152 learnable params.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.008, 0.008,
             "Checkpoint is the final actor, not the best — end-of-training "
             "instability is priced in at full weight, identically in all four waves.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    out = os.path.join(HERE, "c29_summary_2x2.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
