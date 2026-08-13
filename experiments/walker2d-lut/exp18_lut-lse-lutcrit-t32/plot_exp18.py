"""exp18 — does an exponential (log-sum-exp) critic help, and does it push the actor's tau
toward the max-like regime?

LEFT   — learning curves for the 2x2: actor readout (plain sum / LSE-sum) x critic
         (MLP / anchor-pair LUT).
MIDDLE — tau trajectories. The question this experiment was built to answer: exp17's actor
         tau drifted UP (toward the plain sum) because the MLP critic's advantage signal is
         additive. Does an exponential critic pull it DOWN toward max?
RIGHT   — final return by arm, showing where the variance actually is.

Colors: blue = exp10 (plain actor, MLP critic), violet = exp17 (LSE actor, MLP critic),
aqua = exp13 (plain actor, LUT critic), orange = exp18 (LSE actor, exponential LUT critic),
red = exp18 control (LSE actor, plain LUT critic). Identity via legend + line style.

Usage:  python plot_exp18.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
CTRL = os.path.join(BASE, "exp18ctl_lut-lse-plaincrit-t32")
E10 = os.path.join(BASE, "exp10_lut-anchor-pair-t32")
E13 = os.path.join(BASE, "exp13_lut-anchor-pair-lutcrit-t32")
E17 = os.path.join(BASE, "exp17_lut-anchor-pair-t32-logsumexp")
SEEDS = (0, 1, 2)

E10_C, E17_C, E13_C, E18_C, CTL_C = "#2a78d6", "#4a3aa7", "#1baf7a", "#eb6834", "#e34948"
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


def main():
    e18, ctl, e10, e13, e17 = (load(HERE), load(CTRL), load(E10), load(E13), load(E17))
    sm = json.load(open(os.path.join(HERE, "summary.json")))
    c = sm["comparisons"]

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.2), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    series = [("exp10  plain actor + MLP critic", e10, E10_C, "--"),
              ("exp17  LSE actor + MLP critic", e17, E17_C, "-"),
              ("exp13  plain actor + LUT critic", e13, E13_C, ":"),
              ("exp18ctl  LSE actor + plain LUT critic", ctl, CTL_C, ":"),
              ("exp18  LSE actor + EXPONENTIAL LUT critic", e18, E18_C, "-")]

    ax = axes[0]
    for lab, d, col, ls in series:
        for s in SEEDS:
            ax.plot(d[s]["x"], d[s]["y"], color=col, ls=ls,
                    lw=2.0 if col == E18_C else 1.3, alpha=0.9, zorder=6 if col == E18_C else 4)
        ax.plot([], [], color=col, ls=ls, lw=1.9, label=lab)
    ax.legend(fontsize=7.9, labelcolor=INK, loc="upper left", frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.94).set_zorder(10)
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("The MLP-vs-LUT critic gap dominates everything",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    # ---- MIDDLE: tau ------------------------------------------------------
    ax = axes[1]
    for lab, d, col, ls, key in (
            ("exp17 actor τ (MLP critic)", e17, E17_C, "-", "ta"),
            ("exp18ctl actor τ (plain LUT critic)", ctl, CTL_C, ":", "ta"),
            ("exp18 actor τ (exponential critic)", e18, E18_C, "-", "ta"),
            ("exp18 CRITIC τ", e18, E18_C, "--", "tc")):
        for s in SEEDS:
            ax.plot(d[s]["x"], d[s][key], color=col, ls=ls, lw=1.7, alpha=0.9, zorder=5)
        ax.plot([], [], color=col, ls=ls, lw=1.9, label=lab)
    ax.axhline(sm["tau_init"], color=MUTED, lw=1.2, ls=":", zorder=3)
    ax.annotate(f"init τ = {sm['tau_init']:g}", xy=(0.02, sm["tau_init"]),
                xycoords=("axes fraction", "data"), color=MUTED, fontsize=8.2, va="bottom")
    # Top-left is the only region free of both the curves and the legend.
    ax.annotate("↑ more SUM-like  (τ→∞ = plain sum)\n↓ more MAX-like",
                xy=(0.03, 0.97), xycoords="axes fraction", color=INK, fontsize=8.6,
                ha="left", va="top", fontweight="bold", linespacing=1.6)
    ax.legend(fontsize=8.0, labelcolor=INK, loc="lower right", frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.94).set_zorder(10)
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("learned τ", color=MUTED, fontsize=9.5)
    ax.set_title("Every τ still drifts UP, toward the plain sum",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    # ---- RIGHT: finals ----------------------------------------------------
    ax = axes[2]
    arms = [("exp10\nplain+MLP", e10, E10_C, "s"), ("exp17\nLSE+MLP", e17, E17_C, "o"),
            ("exp13\nplain+LUT", e13, E13_C, "D"),
            ("exp18ctl\nLSE+plainLUT", ctl, CTL_C, "v"),
            ("exp18\nLSE+expLUT", e18, E18_C, "^")]
    for i, (lab, d, col, mk) in enumerate(arms):
        vals = [d[s]["y"][-1] for s in SEEDS]
        ax.scatter([i] * 3, vals, s=96, color=col, marker=mk, zorder=5,
                   edgecolor=SURFACE, linewidth=1.5)
        m = float(np.mean(vals))
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(0, 12), textcoords="offset points",
                    color=INK, fontsize=9.2, ha="center", fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=8.0, color=MUTED)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel("final mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Per-seed finals — the LUT-critic arms are wildly unstable",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    fig.suptitle("exp18 — an exponential critic does not rescue the LUT critic, and does "
                 "not push τ toward max",
                 color=INK, fontsize=12.8, x=0.005, ha="left", y=0.985)
    cc = c["control_plain_lut_critic"]
    lines = [
        f"exp18 {sm['ppo_final_mean']:.0f} ± {sm['ppo_final_std']:.0f}  vs its CONTROL "
        f"{cc['reference_final']:.0f} ± {cc['reference_std']:.0f} (identical but for the "
        f"critic's readout): Δ {cc['delta']:+.0f}, Welch se {cc['welch_se']:.0f}, |t| "
        f"{cc['welch_abs_t']:.2f} — NOT significant. The exponential critic is not shown "
        f"to help.",
        f"vs exp17 (same actor, MLP critic) {c['exp17_mlpcrit_lseactor']['reference_final']:.0f}: "
        f"Δ {c['exp17_mlpcrit_lseactor']['delta']:+.0f}, |t| "
        f"{c['exp17_mlpcrit_lseactor']['welch_abs_t']:.2f}, rank-separated   ·   vs exp13 "
        f"(plain actor, LUT critic) {c['exp13_lutcrit_plainactor']['reference_final']:.0f}: "
        f"Δ {c['exp13_lutcrit_plainactor']['delta']:+.0f}, |t| "
        f"{c['exp13_lutcrit_plainactor']['welch_abs_t']:.2f}.",
        f"τ ANSWER: no. Actor τ goes 0.05 → {sm['learned_tau_actor']['mean']:.4f} with the "
        f"exponential critic and → 0.0706 with the plain LUT critic — both UP, i.e. toward "
        f"the plain sum, exactly as in exp17 (0.0887). The critic's own τ also rises, to "
        f"{sm['learned_tau_critic']['mean']:.4f}.",
        "CAVEAT that governs the whole result: the LUT-critic arms have seed sd ~1050–1130 "
        "(finals span 1315–4070 and 1053–3396). At n=3 the Welch se is ~1088, so nothing "
        "smaller than ~2200 points is detectable here — this design cannot resolve the "
        "effect it was built to measure.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.005, 0.085 - 0.021 * i, ln, color=MUTED, fontsize=8.3, ha="left")
    fig.tight_layout(rect=(0, 0.108, 1, 0.945))
    out = os.path.join(HERE, "exp18_result.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
