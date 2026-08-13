"""exp16 — does a trainable exponential output transform hurt the anchor-pair actor?

LEFT   — learning curves. exp16's 3 seeds against gpustar's own exp10 reproduction (the
         same-host control) plus the committed exp10 mean as a reference line.
MIDDLE — what c learned, per seed, over training (init -1).
RIGHT  — what t learned, per seed, over training (init 1).

c and t are plotted in SEPARATE panels rather than on a shared twin axis: they have
different units and ranges, and a dual-scale axis would invite a false visual comparison.

Colors: slots 1-3 of the documented categorical palette in fixed order (blue = committed
exp10, orange = gpustar exp10 reproduction, aqua = exp16). Identity is carried by a legend
plus line style plus direct labels, never by color alone; aqua is a relief-rule color on a
light surface, so its series is always directly labelled. (`validate_palette.js` could not
be run — node is not installed on gpustar — so this relies on the palette file's published
guarantee that the first three slots clear the all-pairs floors in both modes.)

Usage:  python plot_exp16.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
REPRO = os.path.join(BASE, "repro_exp10_gpustar")
SEEDS = (0, 1, 2)

E10_C, RP_C, E16_C = "#2a78d6", "#eb6834", "#1baf7a"
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
        h = json.load(open(os.path.join(folder, f"ppo_s{s}.json")))["history"]
        out[s] = dict(x=np.array([r["env_steps"] for r in h], float) / 1e6,
                      y=np.array([r["ep_ret_mean"] for r in h], float),
                      c=np.array([r.get("c", np.nan) for r in h], float),
                      t=np.array([r.get("t", np.nan) for r in h], float))
    return out


def main():
    e16, rp = load(HERE), load(REPRO)
    sm = json.load(open(os.path.join(HERE, "summary.json")))
    vc, vr = sm["vs_exp10_committed"], sm["vs_exp10_gpustar_repro"]
    e10m = vc["reference_final"]

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.1), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    # ---- LEFT: learning curves -------------------------------------------
    ax = axes[0]
    for s in SEEDS:
        ax.plot(rp[s]["x"], rp[s]["y"], color=RP_C, lw=1.5, alpha=0.8, ls="-", zorder=4)
        ax.plot(e16[s]["x"], e16[s]["y"], color=E16_C, lw=1.7, alpha=0.95, ls="-", zorder=5)
    ax.axhline(e10m, color=E10_C, lw=1.7, ls="--", zorder=3)
    ax.plot([], [], color=E10_C, lw=1.7, ls="--", label=f"exp10 committed — mean {e10m:.0f}")
    ax.plot([], [], color=RP_C, lw=1.5,
            label=f"exp10 gpustar repro — mean {vr['reference_final']:.0f}")
    ax.plot([], [], color=E16_C, lw=1.9,
            label=f"exp16 exp-transform — mean {sm['ppo_final_mean']:.0f}")
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Learning curves — exp16 against the same-host exp10",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    # ---- MIDDLE / RIGHT: what c and t learned ----------------------------
    for ax, key, init, name in ((axes[1], "c", -1.0, "c"), (axes[2], "t", 1.0, "t")):
        finals = [e16[s][key][-1] for s in SEEDS]
        for s in SEEDS:
            ax.plot(e16[s]["x"], e16[s][key], color=E16_C, lw=1.6, alpha=0.9, zorder=5)
        # The three seeds land almost on top of each other, so one range annotation is
        # legible where three per-seed labels overlapped.
        fmt = "{:+.3f}" if key == "c" else "{:.3f}"
        ax.annotate(f"3 seeds converge together\nmean {fmt.format(np.mean(finals))}\n"
                    f"range {fmt.format(min(finals))} … {fmt.format(max(finals))}",
                    xy=(0.97, 0.70), xycoords="axes fraction", color=INK, fontsize=8.8,
                    ha="right", va="top", fontweight="bold", linespacing=1.5)
        ax.axhline(init, color=MUTED, lw=1.2, ls=":", zorder=3)
        ax.annotate(f"init {name} = {init:g}", xy=(0.02, init), xycoords=("axes fraction", "data"),
                    color=MUTED, fontsize=8.2, va="bottom")
        ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
        ax.set_ylabel(f"learned {name}", color=MUTED, fontsize=9.5)
        ax.set_title(f"What {name} learned (3 seeds)", color=INK, fontsize=10.6,
                     loc="left", pad=10)

    fig.suptitle("exp16 — trainable exponential output transform  mean → c + exp(mean / t)",
                 color=INK, fontsize=13.2, x=0.005, ha="left", y=0.985)
    lines = [
        f"exp16 {sm['ppo_final_mean']:.0f} ± {sm['ppo_final_std']:.0f}  vs  committed "
        f"exp10 {e10m:.0f} ± {vc['reference_std']:.0f}: Δ {vc['delta']:+.0f}, Welch se "
        f"{vc['welch_se']:.0f}, |t| {vc['welch_abs_t']:.2f} ({vc['pct_of_reference']:.1f}%)"
        f"   ·   vs gpustar's own exp10 {vr['reference_final']:.0f}: Δ {vr['delta']:+.0f}, "
        f"|t| {vr['welch_abs_t']:.2f} ({vr['pct_of_reference']:.1f}%).",
        f"Collapse {len(sm['collapsed_seeds'])}/3 (criterion final/best < 0.90). "
        f"Learned c mean {sm['learned_c_mean']:+.3f} (init −1), t mean "
        f"{sm['learned_t_mean']:.3f} (init 1). 82,953 params = exp10's 82,951 + c + t.",
        "Fork of exp10: only --arch changes (fastlut → fastlut_exp). c is a free trainable "
        "scalar, t is softplus-constrained > 0. Init c = −1, t = 1 makes the transform a "
        "first-order match to the identity,",
        "so exp16 starts behaviourally identical to exp10 (action means agree to 7.6e-5) "
        "and the experiment measures the transform's effect on learning, not on the "
        "starting point.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.005, 0.085 - 0.021 * i, ln, color=MUTED, fontsize=8.3, ha="left")
    fig.tight_layout(rect=(0, 0.108, 1, 0.945))
    out = os.path.join(HERE, "exp16_result.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
