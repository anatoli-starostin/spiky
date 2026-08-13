"""exp19 — does an exponential MLP-critic readout let the actor's tau reach the max regime?

LEFT   — learning curves: exp19 against its control exp17 (identical but for the critic's
         readout) and exp10.
MIDDLE — the actual question. tau_actor for exp17 vs exp19, and exp19's own tau_critic, all
         on one axis (they share a range, so no dual scale is needed).
RIGHT  — final return by arm.

Colors: blue = exp10, violet = exp17 (the control), orange = exp19. Identity via legend and
line style, never color alone.

Usage:  python plot_exp19.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
E17 = os.path.join(BASE, "exp17_lut-anchor-pair-t32-logsumexp")
E10 = os.path.join(BASE, "exp10_lut-anchor-pair-t32")
SEEDS = (0, 1, 2)

E10_C, E17_C, E19_C = "#2a78d6", "#4a3aa7", "#eb6834"
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
                      ta=np.array([r.get("tau_actor", r.get("tau", np.nan))
                                   for r in h], float),
                      tc=np.array([r.get("tau_critic", np.nan) for r in h], float))
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float(a.mean() - b.mean()), float(se), float(abs(a.mean() - b.mean()) / se)


def main():
    e19, e17, e10 = load(HERE), load(E17), load(E10)
    sm = json.load(open(os.path.join(HERE, "summary.json")))
    c = sm["comparisons"]
    ctl = c["exp17_plain_linear_critic_CONTROL"]

    # Is the small downward nudge in tau_actor even real?
    ta19 = [e19[s]["ta"][-1] for s in SEEDS]
    ta17 = [e17[s]["ta"][-1] for s in SEEDS]
    d_ta, se_ta, t_ta = welch(ta19, ta17)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    ax = axes[0]
    for lab, d, col, ls in (("exp10  plain actor + linear critic", e10, E10_C, "--"),
                            ("exp17  LSE actor + linear critic  (CONTROL)", e17, E17_C, "-"),
                            ("exp19  LSE actor + EXPONENTIAL critic", e19, E19_C, "-")):
        for s in SEEDS:
            ax.plot(d[s]["x"], d[s]["y"], color=col, ls=ls,
                    lw=2.0 if col == E19_C else 1.4, alpha=0.9,
                    zorder=6 if col == E19_C else 4)
        ax.plot([], [], color=col, ls=ls, lw=1.9, label=lab)
    ax.legend(fontsize=8.2, labelcolor=INK, loc="lower right", frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.94).set_zorder(10)
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("The exponential critic readout is harmless", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    # ---- MIDDLE: the tau question ---------------------------------------
    ax = axes[1]
    for lab, d, col, ls, key in (
            ("exp17 τ_actor (linear critic)", e17, E17_C, "-", "ta"),
            ("exp19 τ_actor (exponential critic)", e19, E19_C, "-", "ta"),
            ("exp19 τ_critic", e19, E19_C, "--", "tc")):
        for s in SEEDS:
            ax.plot(d[s]["x"], d[s][key], color=col, ls=ls, lw=1.8, alpha=0.9, zorder=5)
        ax.plot([], [], color=col, ls=ls, lw=1.9, label=lab)
    for init, lab in ((0.05, "init τ_actor = 0.05"), (0.25, "init τ_critic = 0.25")):
        ax.axhline(init, color=MUTED, lw=1.1, ls=":", zorder=3)
        ax.annotate(lab, xy=(0.02, init), xycoords=("axes fraction", "data"),
                    color=MUTED, fontsize=8.0, va="bottom")
    # The band between the two tau groups (y ~0.10-0.24) is empty across the whole width;
    # put the note at its left and the legend at its right so they cannot collide.
    ax.annotate("every τ moves UP\n= toward the\nplain sum / linear\n(down = max regime)",
                xy=(0.03, 0.62), xycoords="axes fraction", color=INK, fontsize=8.6,
                ha="left", va="top", fontweight="bold", linespacing=1.5)
    ax.legend(fontsize=8.1, labelcolor=INK, loc="center right", frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.94).set_zorder(10)
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("learned τ", color=MUTED, fontsize=9.5)
    ax.set_title("The question, answered: τ still goes the wrong way",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    ax = axes[2]
    arms = [("exp10\nplain+linear", e10, E10_C, "s"),
            ("exp17\nLSE+linear\n(control)", e17, E17_C, "o"),
            ("exp19\nLSE+exponential", e19, E19_C, "^")]
    for i, (lab, d, col, mk) in enumerate(arms):
        vals = [d[s]["y"][-1] for s in SEEDS]
        ax.scatter([i] * 3, vals, s=104, color=col, marker=mk, zorder=5,
                   edgecolor=SURFACE, linewidth=1.5)
        m = float(np.mean(vals))
        ax.plot([i - 0.26, i + 0.26], [m, m], color=col, lw=2.6, zorder=4)
        # Label beside the mean line, not above it: the seeds cluster tightly (exp17
        # spans only 80 points) so an overhead label lands on a data point.
        ax.annotate(f"{m:.0f}", xy=(i + 0.29, m), color=INK, fontsize=9.4,
                    ha="left", va="center", fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=8.4, color=MUTED)
    ax.set_xlim(-0.5, len(arms) - 0.15)
    ax.set_ylabel("final mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Final return by arm (3 seeds each)", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    fig.suptitle("exp19 — an exponential MLP-critic readout: harmless, but it still does "
                 "not pull τ toward max",
                 color=INK, fontsize=12.8, x=0.005, ha="left", y=0.985)
    lines = [
        f"exp19 {sm['ppo_final_mean']:.0f} ± {sm['ppo_final_std']:.0f}  vs its CONTROL "
        f"exp17 {ctl['reference_final']:.0f} ± {ctl['reference_std']:.0f} (identical but "
        f"for the critic's readout): Δ {ctl['delta']:+.0f}, Welch se {ctl['welch_se']:.0f}, "
        f"|t| {ctl['welch_abs_t']:.2f} — no detectable difference. vs exp10 "
        f"{c['exp10_plain_actor_mlp_critic']['reference_final']:.0f}: Δ "
        f"{c['exp10_plain_actor_mlp_critic']['delta']:+.0f}, |t| "
        f"{c['exp10_plain_actor_mlp_critic']['welch_abs_t']:.2f}. Collapse "
        f"{len(sm['collapsed_seeds'])}/3.",
        f"τ ANSWER: still no. τ_actor {sm['learned_tau_actor_mean']:.4f} with an "
        f"exponential critic vs {sm['exp17_tau_actor_mean']:.4f} with a linear one — both "
        f"UP from 0.05, and the difference between them is Δ {d_ta:+.4f}, se {se_ta:.4f}, "
        f"|t| {t_ta:.2f}: not significant. The critic's own τ_critic also rises, 0.25 → "
        f"{sm['learned_tau_critic_mean']:.4f}.",
        "This is now a properly controlled null: the strong MLP backbone is held fixed and "
        "bit-identical, so unlike exp18 nothing is confounded by an unstable LUT critic. "
        "Across exp17, exp18 and exp19 every τ on every head has moved toward the "
        "sum/linear limit.",
        "τ_critic init 0.25 was chosen by measurement, not taste: it keeps the value "
        "function 97.6% shape-correlated with exp17's while leaving the exponential live. "
        "τ_critic ≥ 1 is inert (corr 0.999, all 256 units uniform) and would have "
        "guaranteed this null by construction.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.005, 0.085 - 0.021 * i, ln, color=MUTED, fontsize=8.3, ha="left")
    fig.tight_layout(rect=(0, 0.108, 1, 0.945))
    out = os.path.join(HERE, "exp19_result.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"tau_actor exp19 vs exp17: delta {d_ta:+.5f} se {se_ta:.5f} |t| {t_ta:.2f}")


if __name__ == "__main__":
    main()
