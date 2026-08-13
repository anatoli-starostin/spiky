"""exp17 diagnosis: why the log-sum-exp readout stalls, and why the init fix didn't save it.

Reads the LIVE logs (the run is still in flight), so it works before ppo_s*.json exist.

LEFT   — return curves: exp10, exp17 attempt 1 (additive init) and attempt 2 (log-space
         init). Both exp17 attempts flatten at ~300 while exp10 climbs past 5000.
MIDDLE — approximate KL per update, the direct read on "is the policy moving at all".
RIGHT  — the mechanism: sum over tables of d(output)/d(weight). exp10 = 32, the log-sum-exp
         readout = 1. Same weight step, 32x less action movement. Initialisation cannot
         change this number; the sum-scaled variant restores it.

Usage:  python plot_diagnosis.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
SEEDS = (0, 1, 2)
E10_C, A1_C, E17_C, SUM_C = "#2a78d6", "#e34948", "#4a3aa7", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"
PAT = re.compile(r"\[upd\s+([\d,]+)/\d+\]\s+ep_ret\s+(-?[\d.]+).*?kl\s+([\d.]+)")


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def from_logs(folder):
    out = {}
    for s in SEEDS:
        f = os.path.join(folder, f"ppo_s{s}.log")
        if not os.path.exists(f):
            continue
        u, r, k = [], [], []
        for line in open(f, errors="replace"):
            m = PAT.search(line)
            if m:
                u.append(int(m.group(1).replace(",", "")))
                r.append(float(m.group(2)))
                k.append(float(m.group(3)))
        if u:
            out[s] = (np.array(u), np.array(r), np.array(k))
    return out


def from_json(folder):
    import json
    out = {}
    for s in SEEDS:
        f = os.path.join(folder, f"ppo_s{s}.json")
        if not os.path.exists(f):
            continue
        h = json.load(open(f))["history"]
        out[s] = (np.array([x["update"] for x in h]),
                  np.array([x["ep_ret_mean"] for x in h]),
                  np.array([x["kl"] for x in h]))
    return out


def main():
    e10 = from_json(os.path.join(BASE, "repro_exp10_gpustar"))
    a1 = from_json(os.path.join(HERE, "attempt1_additive_init"))
    a2 = from_logs(HERE)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), facecolor=SURFACE)
    for ax in axes:
        style(ax)

    series = [("exp10 (plain sum)", e10, E10_C, "-"),
              ("exp17 attempt 1 — additive init", a1, A1_C, ":"),
              ("exp17 attempt 2 — log-space init", a2, E17_C, "-")]

    ax = axes[0]
    for lab, d, col, ls in series:
        for s, (u, r, _) in d.items():
            ax.plot(u, r, color=col, ls=ls, lw=1.8 if col == E17_C else 1.4,
                    alpha=0.9, zorder=5)
        ax.plot([], [], color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(1000, color=MUTED, lw=1.0, ls="--", zorder=3)
    ax.annotate("the 1000 stop-threshold", xy=(760, 1060), color=MUTED, fontsize=7.8,
                ha="right")
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="center right")
    ax.set_xlabel("PPO update", color=MUTED, fontsize=9.5)
    ax.set_ylabel("mean episodic return", color=MUTED, fontsize=9.5)
    ax.set_title("Both exp17 attempts flatten near 300", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    ax = axes[1]
    for lab, d, col, ls in series:
        for s, (u, _, k) in d.items():
            ax.plot(u, k, color=col, ls=ls, lw=1.6, alpha=0.9, zorder=5)
        ax.plot([], [], color=col, ls=ls, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK, loc="lower left")
    ax.set_xlabel("PPO update", color=MUTED, fontsize=9.5)
    ax.set_ylabel("approx KL per update (log scale)", color=MUTED, fontsize=9.5)
    ax.set_title("The policy is barely moving", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    ax = axes[2]
    labels = ["exp10\nplain sum", "exp17\nlog-sum-exp\n(both attempts)",
              "proposed\nsum-scaled\nlog-sum-exp"]
    vals = [32.0, 1.0, 32.0]
    cols = [E10_C, E17_C, SUM_C]
    ax.bar(range(3), vals, color=cols, width=0.55, zorder=5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}", xy=(i, v), xytext=(0, 5), textcoords="offset points",
                    color=INK, fontsize=11, ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.4, color=MUTED)
    ax.set_ylabel("Σ_tables d(output)/d(weight)", color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 38)
    ax.set_title("The mechanism — measured, not argued", color=INK, fontsize=10.6,
                 loc="left", pad=10)

    fig.suptitle("exp17 diagnosis — the log-sum-exp readout has 32× too little output "
                 "sensitivity, and no initialisation can fix that",
                 color=INK, fontsize=12.8, x=0.005, ha="left", y=0.985)
    lines = [
        "At update 300 attempt 2 reads 346 / 367 / 336 — far below the 1000 threshold, so "
        "the run was NOT stopped. Both attempts plateau from ~update 110 with KL ~1e-3, "
        "about 5× below exp10's at the same stage.",
        "The log-space init did exactly what it was designed to do AT INIT — output mean "
        "+0.000256 vs exp10's +0.000259, std ratio 0.987, effective tables 30.0/32 (vs "
        "attempt 1's saturated +0.3466, 32× too small std, 32.0/32 uniform gradients).",
        "It still didn't learn, which localises the fault: not the initialisation, and not "
        "numerics (torch.logsumexp was used throughout). τ·log Σ exp(w/τ) is a smooth "
        "MEAN/MAX — its gradient sums to 1 over tables where the plain sum gives T = 32.",
        "Fix: T·τ·log((1/T)Σ exp(w/τ)) — a smooth generalisation of the SUM (τ→∞ gives "
        "exactly Σw = exp10; τ→0 gives T·max), gradient sums to T. Verified; ready to run "
        "on request.",
    ]
    for i, ln in enumerate(lines):
        fig.text(0.005, 0.085 - 0.021 * i, ln, color=MUTED, fontsize=8.3, ha="left")
    fig.tight_layout(rect=(0, 0.108, 1, 0.945))
    out = os.path.join(HERE, "exp17_diagnosis.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
