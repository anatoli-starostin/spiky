"""exp_c31 — the PureLIF (TTFS) actor against the three anchors of this line.

Run in the SPIKY venv (matplotlib lives there; the mjx venv has jax but not matplotlib).

LEFT: learning curves, with `temp_bit` on the twin axis where exp_c30's plot carried `eps`.
That swap IS the finding. In c30/c30b the dashed line was a schedule WE imposed, annealed
2.0 -> 0.3 on a horizon we picked, and every seed gave return back over its final stretch.
Here the dashed line is a PARAMETER the model moved on its own -- PureLIF ignores eps
entirely (parity-verified, 0.0 sensitivity) -- so there is nothing to sharpen at the end
and nothing to mis-match between training and eval.

RIGHT: the four-way comparison with PARAMETER COUNTS on the axis, because that is the
variable under test. Points are per-seed, the bar is the mean, the whisker is +/- 1 sd.
Drawn on one axis rather than four panels so the eye compares heights directly. The
per-1k-param figure is annotated because it is the only axis on which these four differ by
more than noise.

Colors: orange #eb6834 = hyperplane front-end, blue #2a78d6 = LIF front-end. The two
earlier LIF variants sit at lower alpha -- they are the same lineage on different budgets,
not different treatments. Matches plot_c29/c30/c30b so the chapter reads as one set.

Usage:
  python plot_c31.py
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


def foot():
    """The caption's statistics, read back from results.json so it cannot drift from the
    numbers collect.py actually computed."""
    r = json.load(open(os.path.join(HERE, "results.json")))
    parts = []
    for a in r["anchors"]:
        tag = a["name"].split()[0].replace("exp_", "")
        parts.append(f"vs {tag} {a['delta']:+.0f} (Welch se {a['welch_se']:.0f}, "
                     f"|t| {abs(a['delta'])/a['welch_se']:.2f})")
    return "Unpaired — no shared seeds.  " + ";  ".join(parts) + "."


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
    hist = {s: json.load(open(os.path.join(HERE, f"pure_lif_sac_c31_s{s}.json")))
            ["history"] for s in SEEDS}
    cpu = [json.load(open(os.path.join(HERE, f"pure_lif_sac_c31_s{s}_cpueval.json")))
           ["cpu_reference_mean"] for s in SEEDS]
    n_par = json.load(open(os.path.join(
        HERE, "pure_lif_sac_c31_s0_cpueval.json")))["params"]
    m = sum(cpu) / len(cpu)
    sd = (sum((v - m) ** 2 for v in cpu) / (len(cpu) - 1)) ** 0.5

    # (label, FRONT-END params, mean, sd, per-seed points or None, color, alpha)
    # Front-end, not total: all four carry the identical 24,576-entry table, so totals are
    # dominated by a component none of them changes. (The old version of this plot used
    # totals AND had the baseline wrong at 49,152 -- that is exp_c29's table-only figure
    # for nap6/tph64; exp_c18 is nap6/tph32, 28,032 total, 3,456 front-end.)
    groups = [("exp_c18\nhyperplane", 3456, 4308.0, 500.1, None, ORANGE, 1.0),
              ("exp_c30\nLIF, dense P", 62785, 3931.3, 585.8, None, BLUE, 0.34),
              ("exp_c30b\nLIF, factorised P", 23617, 4086.8, 991.2, None, BLUE, 0.34),
              ("exp_c31\nPureLIF (TTFS)", n_par - 24576, m, sd, cpu, BLUE, 1.0)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax in axes:
        style(ax)
    axes[0].grid(True, color=GRID, linewidth=0.8, alpha=0.9)

    # --- left: learning curves + the SELF-chosen sharpening ------------------
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        h = hist[s]
        ax.plot([r["iter"] for r in h], [r["mjx_return"] for r in h],
                color=BLUE, alpha=0.45 + 0.25 * i, linewidth=1.9, zorder=3,
                label=f"seed {s}")
        pk = max(h, key=lambda r: r["mjx_return"])
        ax.scatter([pk["iter"]], [pk["mjx_return"]], s=42, color=BLUE,
                   alpha=0.45 + 0.25 * i, edgecolor="white", linewidth=1.4, zorder=4)
    ax.set_xlabel("SAC iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("20-ep MJX proxy (= the deployed policy, always)", color=MUTED,
                  fontsize=9.5)
    # Upper left, not lower right: seed 2 runs flat at ~500 across the whole width and a
    # lower-right legend lands on top of it.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    ax.set_title("All 3 seeds END at their peak — the terminal dip is gone, 3/3",
                 color=INK, fontsize=12, loc="left", pad=10)

    ax2 = ax.twinx()
    for i, s in enumerate(SEEDS):
        ax2.plot([r["iter"] for r in hist[s]], [r["temp_bit"] for r in hist[s]],
                 color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)),
                 alpha=0.4 + 0.2 * i, zorder=2)
    # LOG scale: temp_bit falls 1.0 -> 0.004, so on a linear axis it is pinned to zero
    # after iteration ~1500 and the self-sharpening — the whole point of the panel — is
    # invisible.
    ax2.set_yscale("log")
    ax2.set_ylabel("temp_bit (log) — LEARNED, not scheduled", color=MUTED, fontsize=9.5)
    ax2.tick_params(colors=MUTED, labelsize=9, length=3)
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.annotate("dashed = temp_bit, 1.0 → 0.004.\n"
                 "c30/c30b had an imposed eps anneal here;\n"
                 "PureLIF ignores eps, so this is a parameter\n"
                 "the model sharpened on its own.",
                 xy=(0.26, 0.72), xycoords="axes fraction",
                 color=MUTED, fontsize=8.2, ha="left", va="top")

    # --- right: the four-way comparison at stated parameter budgets ----------
    ax = axes[1]
    for i, (lab, par, gm, gsd, pts, col, al) in enumerate(groups):
        ax.bar(i, gm, width=0.6, color=col, alpha=al * 0.30, zorder=2)
        ax.plot([i - 0.30, i + 0.30], [gm, gm], color=col, alpha=al, linewidth=2.8,
                zorder=4)
        ax.plot([i, i], [gm - gsd, gm + gsd], color=col, alpha=al, linewidth=1.6,
                zorder=3)
        if pts:
            # Jitter in x: seeds 0 and 1 are 190 apart and would otherwise overlap, and a
            # point sitting on the bar centre is hard to read against the whisker.
            xs = [i + 0.13 for _ in pts]
            ax.scatter(xs, pts, s=70, color=col, alpha=al,
                       edgecolor="white", linewidth=1.8, zorder=5)
        # Above whichever is higher, the whisker top or the highest seed point. A label
        # at the mean is struck through by the whisker; one at the whisker top collides
        # with any seed that sits above it.
        top = max([gm + gsd] + (list(pts) if pts else []))
        ax.annotate(f"{gm:.0f}", xy=(i, top), xytext=(0, 10),
                    textcoords="offset points", fontsize=9.5, color=col, alpha=al,
                    ha="center", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    # Front-end size goes in the tick label, not an in-axes annotation: exp_c31's stuck
    # seed sits at 518, right where a bottom-anchored annotation would be.
    ax.set_xticklabels([f"{g[0]}\n{g[1]:,} · {g[1]/3456:.1f}x" for g in groups],
                       fontsize=8.2)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_title("Two seeds on the band, one stuck", color=INK,
                 fontsize=12, loc="left", pad=10)

    fig.suptitle("exp_c31 — PureLIF time-to-first-spike detectors as the Walker2d SAC "
                 "actor's index front-end",
                 color=INK, fontsize=13.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.030,
             f"No ordered-pair P at all: order information enters through the arrival "
             f"dynamics instead. {n_par:,} total actor params ({n_par-24576:,} front-end "
             f"+ 24,576 table, a table all four share). Bars are means, whiskers ±1 sd.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.text(0.008, 0.008, foot(), color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.94))
    out = os.path.join(HERE, "c31_pure_lif_result.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
