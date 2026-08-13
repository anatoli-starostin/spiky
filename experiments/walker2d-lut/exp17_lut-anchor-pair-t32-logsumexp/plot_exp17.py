"""exp17 — the sum-scaled log-sum-exp readout: does it train like exp10?

LEFT   — learning curves: exp17 (sum-scaled) against exp16, gpustar's exp10 reproduction,
         and the abandoned plain-log-sum-exp attempt, with committed exp10 as a line.
MIDDLE — what tau learned (init 0.05), and the approx-KL trace, which is the direct read on
         "is the policy actually moving" — the thing the plain readout failed at.
RIGHT  — final return by arm.

Colors keep the entity->hue mapping used in the exp16 figure so they read together:
blue = committed exp10, orange = gpustar's exp10 reproduction, aqua = exp16,
violet = exp17 (sum-scaled), red = the abandoned plain log-sum-exp. Identity is carried by
legend + line style, never color alone. (`validate_palette.js` could not be run — node is
not installed on gpustar — so this relies on the palette file's documented slot order.)

Usage:  python plot_exp17.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
E10 = os.path.join(BASE, "exp10_lut-anchor-pair-t32")
RP = os.path.join(BASE, "repro_exp10_gpustar")
E16 = os.path.join(BASE, "exp16_lut-anchor-pair-t32-expout")
A1 = os.path.join(HERE, "attempt1_additive_init")
SEEDS = (0, 1, 2)

E10_C, RP_C, E16_C, E17_C = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
A1_C = "#e34948"
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
                      kl=np.array([r["kl"] for r in h], float),
                      tau=np.array([r.get("tau", np.nan) for r in h], float))
    return out


def main():
    e17, e16, rp, e10, a1 = load(HERE), load(E16), load(RP), load(E10), load(A1)
    sm = json.load(open(os.path.join(HERE, "summary.json")))
    c = sm["comparisons"]
    e10m = c["exp10_committed"]["reference_final"]
    a1m = c["plain_lse_attempt1"]["reference_final"]

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.2), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    # ---- LEFT: learning curves ------------------------------------------
    ax = axes[0]
    for s in SEEDS:
        ax.plot(rp[s]["x"], rp[s]["y"], color=RP_C, lw=1.3, alpha=0.6, zorder=4)
        ax.plot(e16[s]["x"], e16[s]["y"], color=E16_C, lw=1.3, alpha=0.6, zorder=4)
        ax.plot(a1[s]["x"], a1[s]["y"], color=A1_C, lw=1.4, alpha=0.8, ls=":", zorder=5)
        ax.plot(e17[s]["x"], e17[s]["y"], color=E17_C, lw=2.0, alpha=0.95, zorder=6)
    ax.axhline(e10m, color=E10_C, lw=1.7, ls="--", zorder=3)
    ax.plot([], [], color=E10_C, lw=1.7, ls="--", label=f"exp10 committed — {e10m:.0f}")
    ax.plot([], [], color=RP_C, lw=1.5,
            label=f"exp10 gpustar repro — {c['exp10_gpustar_repro']['reference_final']:.0f}")
    ax.plot([], [], color=E16_C, lw=1.5,
            label=f"exp16 c+exp(y/t) — {c['exp16_expout']['reference_final']:.0f}")
    ax.plot([], [], color=A1_C, lw=1.5, ls=":",
            label=f"plain log-sum-exp (abandoned) — {a1m:.0f}")
    ax.plot([], [], color=E17_C, lw=2.2,
            label=f"exp17 SUM-SCALED — {sm['ppo_final_mean']:.0f}")
    # Solid surface-coloured frame: the abandoned run's flat curve sits at ~500 and would
    # otherwise be drawn straight through the legend text.
    ax.legend(fontsize=8.2, labelcolor=INK, loc="lower right", frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.94).set_zorder(10)
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Learning curves", color=INK, fontsize=10.6, loc="left", pad=10)

    # ---- MIDDLE: KL — is the policy moving? -----------------------------
    ax = axes[1]
    for lab, d, col, ls in (("exp10 gpustar", rp, RP_C, "-"),
                            ("plain log-sum-exp", a1, A1_C, ":"),
                            ("exp17 sum-scaled", e17, E17_C, "-")):
        for s in SEEDS:
            ax.plot(d[s]["x"], d[s]["kl"], color=col, ls=ls, lw=1.6, alpha=0.9, zorder=5)
        ax.plot([], [], color=col, ls=ls, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="lower left")
    ax.set_xlabel("env-steps (millions)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("approx KL per update (log scale)", color=MUTED, fontsize=9.5)
    ax.set_title("Is the policy moving? — the plain readout's failure mode",
                 color=INK, fontsize=10.6, loc="left", pad=10)

    # ---- RIGHT: finals by arm -------------------------------------------
    ax = axes[2]
    arms = [("exp10\ncommitted", e10, E10_C, "s"), ("exp10\ngpustar", rp, RP_C, "^"),
            ("exp16\nc+exp(y/t)", e16, E16_C, "D"),
            ("plain LSE\n(abandoned)", a1, A1_C, "v"),
            ("exp17\nsum-scaled", e17, E17_C, "o")]
    for i, (lab, d, col, mk) in enumerate(arms):
        vals = [d[s]["y"][-1] for s in SEEDS]
        ax.scatter([i] * 3, vals, s=96, color=col, marker=mk, zorder=5,
                   edgecolor=SURFACE, linewidth=1.5)
        m = float(np.mean(vals))
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=2.6, zorder=4)
        ax.annotate(f"{m:.0f}", xy=(i, m), xytext=(0, 11), textcoords="offset points",
                    color=INK, fontsize=9.2, ha="center", fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=8.2, color=MUTED)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel("final mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Final return by arm (3 seeds each)", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    fig.suptitle("exp17 — sum-scaled log-sum-exp readout   "
                 "out = T · τ · log( (1/T) Σ_t exp(w_t / τ) )",
                 color=INK, fontsize=13.0, x=0.005, ha="left", y=0.985)
    wu = sm["warmup_updates_to"]
    wu1 = sm["warmup_reference"]["plain_lse_attempt1"]
    wu10 = sm["warmup_reference"]["exp10_gpustar_repro"]
    tau_f = [sm["learned_tau"][str(s)] for s in SEEDS]

    def wtxt(d, lv):
        v = d.get(str(lv))
        return "never" if v is None else f"{v:.0f}"

    lines = [
        f"exp17 {sm['ppo_final_mean']:.0f} ± {sm['ppo_final_std']:.0f}  vs  committed "
        f"exp10 {e10m:.0f}: Δ {c['exp10_committed']['delta']:+.0f}, |t| "
        f"{c['exp10_committed']['welch_abs_t']:.2f} "
        f"({c['exp10_committed']['pct_of_reference']:.1f}%)   ·   vs exp16 "
        f"{c['exp16_expout']['reference_final']:.0f}: Δ {c['exp16_expout']['delta']:+.0f}, "
        f"|t| {c['exp16_expout']['welch_abs_t']:.2f}   ·   vs the abandoned plain "
        f"log-sum-exp {a1m:.0f}: Δ {c['plain_lse_attempt1']['delta']:+.0f}.",
        f"WARMUP (mean updates of 768 to reach a level) — exp17: 1000 at "
        f"{wtxt(wu, 1000)}, 3000 at {wtxt(wu, 3000)}, 5000 at {wtxt(wu, 5000)}.  "
        f"exp10: {wtxt(wu10, 1000)} / {wtxt(wu10, 3000)} / {wtxt(wu10, 5000)}.  "
        f"plain log-sum-exp: {wtxt(wu1, 1000)} / {wtxt(wu1, 3000)} / {wtxt(wu1, 5000)}.",
        f"Collapse {len(sm['collapsed_seeds'])}/3 (final/best < 0.90). Learned τ "
        f"{', '.join(f'{v:.4f}' for v in tau_f)} (init {sm['tau_init']:g}). 82,952 params "
        f"= exp10's 82,951 + τ.",
        "Why this readout and not the plain τ·log Σ exp(w/τ): multiplying by T and "
        "subtracting τ·log T makes it generalise the SUM, not the mean — τ→∞ gives exactly "
        "Σw (exp10's readout), τ→0 gives T·max — and restores Σ_tables d(out)/d(w) = 32, "
        "against 1 for the plain form.",
        "That factor-32 loss of output sensitivity, not the initialisation, is why the "
        "plain readout plateaued: a log-space init that reproduced exp10's starting "
        "statistics exactly (mean +0.000256 vs +0.000259, std ratio 0.987) still stalled "
        "at ~350.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.005, 0.104 - 0.021 * i, ln, color=MUTED, fontsize=8.3, ha="left")
    fig.tight_layout(rect=(0, 0.128, 1, 0.945))
    out = os.path.join(HERE, "exp17_result.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
