"""exp_c36 — the Bucket-LIF actor against the four anchors of this line.

Run in the SPIKY venv (matplotlib lives there).

LEFT: learning curves, with the MEAN BUCKET INDEX on the twin axis. That is the diagnostic
this model lives or dies by. Non-firing neurons fold into the last bucket, so every run
starts pinned near 15 and the escape is visible as the mean falling while the spread rises
-- an ordinary return curve cannot show whether a flat run is stuck at the wall or merely
learning slowly.

RIGHT: the five-way comparison, with each bar labelled by its FRONT-END size rather than
its total. Every other model in the chapter carries the same 24,576-entry table; this one
does not (16 rows, so 6,144), which is why the caption gives both numbers. Points are
per-seed, bar is the mean, whisker is ±1 sd.

Colors: orange #eb6834 = hyperplane, blue #2a78d6 = the LIF family, with the three earlier
LIF variants at lower alpha since they are the same lineage on different budgets. Matches
plot_c29/c30/c30b/c31 so the chapter reads as one set.

Usage:
  python plot_c32.py
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


def style(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def foot():
    r = json.load(open(os.path.join(HERE, "results.json")))
    parts = []
    for a in r["anchors"]:
        tag = a["name"].split()[0].replace("exp_", "")
        parts.append(f"vs {tag} {a['delta']:+.0f} (se {a['welch_se']:.0f}, "
                     f"|t| {abs(a['delta'])/a['welch_se']:.2f})")
    return "Unpaired — no shared seeds.  " + ";  ".join(parts) + "."


def main():
    hist = {s: json.load(open(os.path.join(HERE, f"bucket_sac_c36_s{s}.json")))
            ["history"] for s in SEEDS}
    cpu = [json.load(open(os.path.join(HERE, f"bucket_sac_c36_s{s}_cpueval.json")))
           ["cpu_reference_mean"] for s in SEEDS]
    r = json.load(open(os.path.join(HERE, "results.json")))
    n_par, n_front = r["actor_params"], r["frontend_params"]
    m = sum(cpu) / len(cpu)
    sd = (sum((v - m) ** 2 for v in cpu) / (len(cpu) - 1)) ** 0.5

    # (label, front-end params, mean, sd, per-seed points or None, color, alpha)
    # Six groups on one axis — labels must stay short or they collide.
    groups = [("c18\nhyper", 3456, 4308.0, 500.1, None, ORANGE, 1.0),
              ("c30\ndense P", 62785, 3931.3, 585.8, None, BLUE, 0.32),
              ("c30b\nfact. P", 23617, 4086.8, 991.2, None, BLUE, 0.32),
              ("c31\nTTFS bits", 6816, 2951.2, 2109.2, None, BLUE, 0.32),
              ("c32\nBROKEN", 1696, 1174.4, 517.2,
               [1739.9, 725.5, 1057.7], "#b03030", 0.55),
              ("c32b\nFIXED", n_front, m, sd, cpu, BLUE, 1.0)]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax in axes:
        style(ax)
    axes[0].grid(True, color=GRID, linewidth=0.8, alpha=0.9)

    # --- left: learning curves + the escape from the last-bucket wall --------
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        h = hist[s]
        ax.plot([q["iter"] for q in h], [q["mjx_return"] for q in h],
                color=BLUE, alpha=0.45 + 0.25 * i, linewidth=1.9, zorder=3,
                label=f"seed {s}")
        pk = max(h, key=lambda q: q["mjx_return"])
        ax.scatter([pk["iter"]], [pk["mjx_return"]], s=42, color=BLUE,
                   alpha=0.45 + 0.25 * i, edgecolor="white", linewidth=1.4, zorder=4)
    ax.set_xlabel("SAC iteration", color=MUTED, fontsize=9.5)
    ax.set_ylabel("20-ep MJX proxy (= the deployed policy, always)", color=MUTED,
                  fontsize=9.5)
    pass
    # NOT "no terminal dip". exp_c31 ended at peak 3/3 and it was tempting to read that as
    # a property of schedule-free LIF actors. Here seed 0 peaks at 9,000 and gives back
    # 510, seed 1 gives back 33, only seed 2 ends at its peak. Removing the anneal removed
    # the SYSTEMATIC dip (6/6 in c30/c30b); it does not prevent ordinary late decline.
    ax.set_title("Seed 1 walks: 84/100 full episodes, mean length 942",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    ax2 = ax.twinx()
    for i, s in enumerate(SEEDS):
        ax2.plot([q["iter"] for q in hist[s]], [q["bucket_mean"] for q in hist[s]],
                 color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)),
                 alpha=0.4 + 0.2 * i, zorder=2)
    ax2.set_ylabel("mean bucket index (dashed) — 15 = the no-spike wall",
                   color=MUTED, fontsize=9.5)
    ax2.set_ylim(0, 16)
    ax2.tick_params(colors=MUTED, labelsize=9, length=3)
    for sp in ("top", "left", "bottom"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    ax2.annotate("dashed = mean bucket index (right axis). Non-firing neurons fold\n"
                 "into the LAST bucket, so runs start near 15 and settle at 8–11 —\n"
                 "only 4.5 of 16 buckets end up effectively used (entropy 2.02/4.00).",
                 xy=(0.015, 0.985), xycoords="axes fraction", color=MUTED,
                 fontsize=8.2, ha="left", va="top")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left")

    # --- right: the five-way comparison, labelled by front-end size ----------
    ax = axes[1]
    for i, (lab, front, gm, gsd, pts, col, al) in enumerate(groups):
        ax.bar(i, gm, width=0.62, color=col, alpha=al * 0.30, zorder=2)
        ax.plot([i - 0.31, i + 0.31], [gm, gm], color=col, alpha=al, linewidth=2.8,
                zorder=4)
        ax.plot([i, i], [gm - gsd, gm + gsd], color=col, alpha=al, linewidth=1.6,
                zorder=3)
        if pts:
            ax.scatter([i + 0.15] * len(pts), pts, s=64, color=col, alpha=al,
                       edgecolor="white", linewidth=1.6, zorder=5)
        top = max([gm + gsd] + (list(pts) if pts else []))
        ax.annotate(f"{gm:.0f}", xy=(i, top), xytext=(0, 10),
                    textcoords="offset points", fontsize=9.5, color=col, alpha=al,
                    ha="center", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[0]}\n{g[1]:,}" for g in groups], fontsize=7.8)
    ax.set_ylabel("100-episode deterministic CPU reference", color=MUTED, fontsize=9.5)
    ax.set_xlabel("front-end params (all but the two c32 bars share a 24,576 table; theirs is 6,144)",
                  color=MUTED, fontsize=8.2)
    ax.set_title("0/300 full episodes -> 98/300; still short of the line",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    fig.suptitle("exp_c36 — bucket-addressed LIF: one neuron per table, row = which of 16 "
                 "time buckets its first spike lands in",
                 color=INK, fontsize=13, x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.030,
             f"No anchor pairs and no bit vector: the address is a monotone quantisation "
             f"of ONE scalar, with trainable sorted boundaries. {n_par:,} total params "
             f"({n_front:,} front-end + {n_par-n_front:,} table) — 28% of the baseline. "
             f"Bars are means, whiskers ±1 sd.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.text(0.006, 0.008, foot(), color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.062, 1, 0.945))
    out = os.path.join(HERE, "c36_tables_result.png")
    fig.savefig(out, dpi=165, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
