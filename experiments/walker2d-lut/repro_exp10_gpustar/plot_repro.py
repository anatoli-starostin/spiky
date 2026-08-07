"""exp10 reproduction on gpustar vs the committed nebius reference.

LEFT  — learning curves, all 6 runs (3 reference seeds + 3 reproduction seeds) on one
        return axis. The question the panel answers: do the two hosts trace the same
        curve, or only land on the same endpoint?
RIGHT — per-seed final return, paired by seed, with each arm's mean. Seeds are NOT
        expected to match one-for-one (different GPU, different nondeterministic kernel
        reductions); the arm means and the spread are what carry the claim.

Colors: slots 1-2 of the documented categorical palette, in fixed order (blue = the
committed reference, orange = this reproduction). Two series only, so the pair is the
documented adjacent pair. Identity is never color-alone: legend + line style + direct
labels, and all text is in ink tokens with a colored mark carrying the identity.
(`validate_palette.js` could not be run here — node is not installed on gpustar — so this
relies on the palette file's published guarantee for the adjacent slot-1/slot-2 pair.)

Usage:  python plot_repro.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "exp10_lut-anchor-pair-t32")
SEEDS = (0, 1, 2)

REPRO_C, REF_C = "#eb6834", "#2a78d6"        # categorical slots 2 and 1
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def load(folder):
    out = {}
    for s in SEEDS:
        j = json.load(open(os.path.join(folder, f"ppo_s{s}.json")))
        h = j["history"]
        out[s] = (np.array([r["env_steps"] for r in h], float) / 1e6,
                  np.array([r["ep_ret_mean"] for r in h], float))
    return out


def main():
    rep, ref = load(HERE), load(REF)
    sm = json.load(open(os.path.join(HERE, "summary.json")))
    c = sm["comparison"]
    rf, rb = sm["ppo_final_mean"], sm["ppo_final_std"]
    ff, fb = sm["reference"]["ppo_final_mean"], sm["reference"]["ppo_final_std"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    # ---- LEFT: learning curves ------------------------------------------
    ax = axes[0]
    for s in SEEDS:
        ax.plot(*ref[s], color=REF_C, lw=1.6, alpha=0.85, ls="--", zorder=4)
        ax.plot(*rep[s], color=REPRO_C, lw=1.6, alpha=0.85, ls="-", zorder=5)
    ax.plot([], [], color=REF_C, lw=1.8, ls="--", label="reference (nebius) — 3 seeds")
    ax.plot([], [], color=REPRO_C, lw=1.8, ls="-", label="reproduction (gpustar) — 3 seeds")
    ax.legend(frameon=False, fontsize=8.8, labelcolor=INK, loc="lower right")
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Same trajectory, not just the same endpoint",
                 color=INK, fontsize=10.8, loc="left", pad=10)

    # ---- RIGHT: per-seed finals -----------------------------------------
    ax = axes[1]
    x = np.arange(len(SEEDS))
    rep_f = [rep[s][1][-1] for s in SEEDS]
    ref_f = [ref[s][1][-1] for s in SEEDS]
    ax.scatter(x - 0.10, ref_f, s=104, color=REF_C, zorder=5, marker="s",
               edgecolor=SURFACE, linewidth=1.5,
               label=f"reference (nebius) — mean {ff:.0f}")
    ax.scatter(x + 0.10, rep_f, s=104, color=REPRO_C, zorder=5, marker="o",
               edgecolor=SURFACE, linewidth=1.5,
               label=f"reproduction (gpustar) — mean {rf:.0f}")
    ax.axhline(ff, color=REF_C, lw=1.6, ls="--", alpha=0.85, zorder=3)
    ax.axhline(rf, color=REPRO_C, lw=1.6, ls="-", alpha=0.85, zorder=3)
    # Legend carries the arm means: inline labels on the mean lines collided with the
    # seed-2 points, and the upper-left of this panel is empty.
    ax.legend(frameon=False, fontsize=8.8, labelcolor=INK, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=9, color=MUTED)
    ax.set_xlim(-0.45, 2.55)
    ax.set_ylabel("final mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Per-seed finals — arm means agree to "
                 f"{abs(c['final_delta']):.0f} points",
                 color=INK, fontsize=10.8, loc="left", pad=10)

    fig.suptitle("exp10_lut-anchor-pair-t32 reproduced on gpustar (RTX 5090)",
                 color=INK, fontsize=13.5, x=0.006, ha="left", y=0.985)
    lines = [
        f"Reproduction {rf:.0f} ± {rb:.0f} vs committed reference {ff:.0f} ± {fb:.0f} "
        f"— Δ {c['final_delta']:+.0f}, Welch se {c['welch_se']:.0f}, |t| "
        f"{c['welch_abs_t']:.2f}: no detectable difference. "
        f"{c['pct_of_reference']:.1f}% of the reference.",
        "Identical config (flags verbatim from exp10/config.json) and identical "
        "architecture (82,951 params, exact match); 3 seeds in parallel.",
        f"Collapse {len(sm['collapsed_seeds'])}/3 vs "
        f"{len(sm['reference']['collapsed_seeds'])}/3 — criterion final/best < 0.90, "
        f"calibrated to reproduce the committed exp02–05 labels.",
        f"Throughput {sm['throughput_env_per_s_mean']:,} env-steps/s vs "
        f"{sm['reference']['throughput_env_per_s_mean']:,} "
        f"({c['speedup_vs_reference']:.2f}× the reference host); "
        f"{sm['training_time_hours_mean'] * 60:.0f} min/seed vs "
        f"{sm['reference']['training_time_hours_mean'] * 60:.0f}.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.006, 0.086 - 0.021 * i, ln, color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.108, 1, 0.945))
    out = os.path.join(HERE, "repro_exp10.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
