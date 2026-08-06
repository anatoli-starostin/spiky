"""exp_c39 diagnosis — the two figures.

Run in the SPIKY venv (matplotlib).

FIGURE 1 `c39_diag_trajectories.png` — the negative result, which is the important one.
All six MHL seeds (exp_c38 and exp_c39 share the trainer, recipe and diagnostic
definitions, giving three takeoffs and three flats instead of one and two). Return separates
cleanly from iteration ~4,000. NONE of the three mechanical addressing diagnostics separates
at any point, and two of them separate the WRONG WAY inside c39: the seed that took off has
the LOWEST effective-cells and the LOWEST coverage of the three.

FIGURE 2 `c39_diag_forensics.png` — where the addressing-collapse story dies. Init is
regenerated exactly from PRNGKey(seed), so the starting state of each seed is measurable
rather than inferred. Left: at init the three seeds are indistinguishable on every measure.
Middle: at the end the WINNER is the most collapsed one — fewest live detectors, fewest
cells touched, lowest digit entropy. Right: the losers' tables are not gradient-starved;
they move MORE rows by MORE total displacement than the winner's.

Usage:
  python plot_diag.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C38 = os.path.join(D, "exp_c38_mhl_6det_2bkt")

GREEN, RED, MUTED2 = "#1f9e5a", "#c0392b", "#9a9890"
INK, MUTED, GRID = "#1c1c1a", "#6b6a63", "#e3e2dd"

# (label, dir, stem, took off)
SEEDS = [
    ("c38 s0", C38, "mhl_sac_c38_s0", False),
    ("c38 s1", C38, "mhl_sac_c38_s1", True),
    ("c38 s2", C38, "mhl_sac_c38_s2", True),
    ("c39 s0", HERE, "mhl_sac_c39_s0", False),
    ("c39 s1", HERE, "mhl_sac_c39_s1", False),
    ("c39 s2", HERE, "mhl_sac_c39_s2", True),
]
PANELS = [
    ("mjx_return", "MJX return (20-ep proxy)", None),
    ("eff_cells", "effective cells / table (of 64)", None),
    ("row_coverage", "table cell coverage (fraction)", None),
    ("nospike", "no-spike rate (detectors that never fire)", None),
]


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def fig_trajectories():
    hs = {lab: json.load(open(os.path.join(d, stem + ".json")))["history"]
          for lab, d, stem, _ in SEEDS}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), facecolor="white")
    for ax in axes.ravel():
        style(ax)

    for ax, (key, ylab, _) in zip(axes.ravel(), PANELS):
        for lab, d, stem, ok in SEEDS:
            h = hs[lab]
            xs = [e["iter"] for e in h]
            ys = [e.get(key) for e in h]
            if any(v is None for v in ys):
                continue
            ax.plot(xs, ys, color=(GREEN if ok else RED),
                    linestyle=("-" if "c39" in lab else "--"),
                    linewidth=(2.1 if "c39" in lab else 1.4),
                    alpha=(0.95 if "c39" in lab else 0.55), zorder=4)
        ax.set_ylabel(ylab, color=MUTED, fontsize=9)
        ax.set_xlabel("training iteration", color=MUTED, fontsize=9)

    axes[0, 0].axvline(4000, color=INK, linewidth=1.0, linestyle=":", alpha=0.6)
    axes[0, 0].annotate("return separates here", xy=(4200, 3600), color=INK,
                        fontsize=8.4, ha="left")
    for ax in axes.ravel()[1:]:
        ax.axvline(4000, color=INK, linewidth=1.0, linestyle=":", alpha=0.35)

    h1, = axes[0, 0].plot([], [], color=GREEN, linewidth=2.1, label="took off (3 seeds)")
    h2, = axes[0, 0].plot([], [], color=RED, linewidth=2.1, label="stayed flat (3 seeds)")
    h3, = axes[0, 0].plot([], [], color=MUTED2, linewidth=2.1, linestyle="-",
                          label="c39 (solid) · c38 (dashed)")
    axes[0, 0].legend(handles=[h1, h2, h3], frameon=False, fontsize=8.4,
                      labelcolor=INK, loc="upper left")

    fig.suptitle("exp_c39 diagnosis — the outcome separates; none of the mechanics do",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             "All six MHL seeds (c38 6det×2bkt and c39 3det×4bkt share the trainer, recipe "
             "and diagnostic definitions). Return separates cleanly from ~4,000. Effective "
             "cells, coverage and no-spike rate all OVERLAP at every probe —",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "and inside c39 two of them separate the WRONG WAY: the seed that took off has "
             "the lowest effective-cells and the lowest coverage of the three. Addressing "
             "collapse is not the failure mechanism.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c39_diag_trajectories.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def fig_forensics():
    f = json.load(open(os.path.join(HERE, "forensics.json")))
    seeds = ["0", "1", "2"]
    cols = [RED, RED, GREEN]
    labels = ["s0 flat\n891", "s1 flat\n982", "s2 WIN\n4217"]

    groups = [
        ("At INIT — indistinguishable", [
            ("no-spike\nrate", lambda s: f[s]["init"]["nospike"], 1.0),
            ("detector\nentropy (bits)", lambda s: f[s]["init"]["det_entropy_mean"], 1.0),
            ("eff cells\n/ table", lambda s: f[s]["init"]["eff_cells_mean"] / 10, 10.0),
            ("delay\nstd / 10", lambda s: f[s]["init"]["nospike"] * 0 +
             f[s]["delay_std"] / 10, 10.0),
        ]),
        ("At the END — the WINNER is the most collapsed", [
            ("live detectors\n/ 96", lambda s: f[s]["final"]["live_detectors"] / 96, 96),
            ("detector\nentropy (bits)", lambda s: f[s]["final"]["det_entropy_mean"], 1.0),
            ("eff cells\n/ 10", lambda s: f[s]["final"]["eff_cells_mean"] / 10, 10.0),
            ("cells touched\n/ 64", lambda s: f[s]["final"]["cells_touched_mean"] / 64,
             64.0),
        ]),
        ("Table learning — losers move MORE, not less", [
            ("mean row\ndisplacement", lambda s: f[s]["table_disp_mean"], 1.0),
            ("displacement on\naddressed rows", lambda s: f[s]["table_disp_used_mean"],
             1.0),
            ("displacement on\nUNused rows", lambda s: f[s]["table_disp_unused_mean"],
             1.0),
            ("rows moved\n>0.1  / 2048", lambda s: f[s]["rows_moved_gt_0p1"] / 2048,
             2048.0),
        ]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0), facecolor="white")
    for ax, (title, items) in zip(axes, groups):
        style(ax)
        w = 0.24
        xs = np.arange(len(items))
        for i, s in enumerate(seeds):
            vals = [fn(s) for _, fn, _ in items]
            ax.bar(xs + (i - 1) * w, vals, width=w, color=cols[i],
                   alpha=(0.92 if i == 2 else 0.62), zorder=4,
                   edgecolor="white", linewidth=1.1,
                   label=labels[i] if title.startswith("At INIT") else None)
            for x, v, (_, _, scale) in zip(xs + (i - 1) * w, vals, items):
                shown = v * scale if scale != 1.0 else v
                txt = f"{shown:.0f}" if scale != 1.0 else f"{v:.2f}"
                ax.annotate(txt, xy=(x, v), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            color=MUTED, fontsize=7.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _, _ in items], fontsize=8.0)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        ax.set_ylim(0, max(max(fn(s) for s in seeds) for _, fn, _ in items) * 1.28)
    axes[0].legend(frameon=False, fontsize=8.2, labelcolor=INK, loc="upper right",
                   ncol=3, columnspacing=0.8, handlelength=1.0)

    fig.suptitle("exp_c39 forensics — init regenerated exactly; the failure is not "
                 "addressing collapse and not a starved table",
                 color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.028,
             "Bars are annotated with the value in the units named on the axis label. "
             "Init: every seed within noise of every other, zero dead detectors in all "
             "three. End: the winner has the fewest live detectors and touches the fewest "
             "cells.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008,
             "Table: the losers are not gradient-starved — they move more rows further. "
             "What the winner has is CONCENTRATION: displacement on addressed rows over "
             "unaddressed rows is 1.96, against 1.58 and 1.51.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(HERE, "c39_diag_forensics.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_trajectories()
    fig_forensics()
