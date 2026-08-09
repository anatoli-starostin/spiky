"""Two figures for the inhibitory-sync experiment: error-vs-jitter, and the closed-phase law."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BG, INK, MUTED = "#faf9f7", "#2a2622", "#7d7368"
C_EXACT, C_LIF, C_REF = "#4a6fa5", "#c1666b", "#5c8d5a"
REQUIRED = 0.488          # t_release - min_ready, the closed duration the gate must cover


def closed_fraction(cfg):
    thr = (6.0 - cfg["theta0"]) / cfg["A"]
    if cfg["waveform"] == "square":
        return cfg["duty"]
    if cfg["waveform"] == "sawtooth":
        return 1.0 - min(1.0, max(0.0, thr))
    if thr <= 0:
        return 1.0
    if thr >= 1:
        return 0.0
    return math.acos(2.0 * thr - 1.0) / math.pi          # sine


def main():
    R = json.load(open(os.path.join(RES, "inhib_sync.json")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=BG)

    # ---- (a) error vs arrival jitter -------------------------------------------------
    ax1.set_facecolor(BG)
    for neuron, col, lab in ((("exact"), C_EXACT, "exact exp-kernel neuron"),
                             (("lif"), C_LIF, "hardware LIF (tau_s = tau_m/2)")):
        pts = [(x["jitter_std"], x["norm_mean"]) for x in R
               if x["cfg"]["neuron"] == neuron and x["miss_rate"] < 1e-9
               and np.isfinite(x["jitter_std"]) and x["jitter_std"] > 1e-6]
        if not pts:
            continue
        j, e = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        ax1.scatter(j, e, s=22, color=col, alpha=0.75, edgecolor="none", label=lab)
        sl, ic = np.polyfit(np.log(j), np.log(e), 1)
        xs = np.array([j.min(), j.max()])
        ax1.plot(xs, np.exp(ic) * xs ** sl, lw=1.5, color=col, alpha=0.55)
        ax1.annotate(f"slope {sl:.2f}", xy=(j.max(), np.exp(ic) * j.max() ** sl),
                     fontsize=8, color=col, ha="right", va="bottom")
    for x in R:
        if x["cfg"]["waveform"] == "NO-SYNC":
            ax1.scatter([x["jitter_std"]], [x["norm_mean"]], marker="X", s=110,
                        color=C_REF, zorder=5, edgecolor=BG, lw=1.2)
    ax1.axhline(1.145e-05, ls="--", lw=1.2, color=MUTED)
    ax1.annotate("ideal latch (zero jitter): 1.1e-05", xy=(2e-4, 1.6e-05),
                 fontsize=7.5, color=MUTED)
    ax1.annotate("NO-SYNC\n(cells fire when ready)", xy=(0.087, 1.14),
                 xytext=(0.011, 3.2), fontsize=8, color=C_REF,
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=1.1))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("arrival-time jitter across the 32 cells (std, units of T)",
                   color=MUTED, fontsize=9)
    ax1.set_ylabel("mean |a_spiking - a_LUT| / action std", color=MUTED, fontsize=9)
    ax1.set_title("Accuracy is set by arrival jitter", color=INK, fontsize=11)
    leg = ax1.legend(fontsize=8, framealpha=1.0, facecolor=BG, edgecolor="#ddd8d0",
                     loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- (b) the closed-phase law ------------------------------------------------------
    ax2.set_facecolor(BG)
    marks = {"square": "s", "sine": "o", "sawtooth": "^"}
    for wf, mk in marks.items():
        sub = [x for x in R if x["cfg"]["waveform"] == wf and x["cfg"]["memory"] == "latch"
               and x["cfg"]["neuron"] == "exact" and x["cfg"].get("aligned")]
        if not sub:
            continue
        cd = [closed_fraction(x["cfg"]) * x["cfg"]["P"] for x in sub]
        er = [max(x["norm_mean"], 1e-5) for x in sub]
        ax2.scatter(cd, er, marker=mk, s=40, alpha=0.8, edgecolor="none",
                    color=C_EXACT, label=wf)
    ax2.axvline(REQUIRED, ls="--", lw=1.4, color=C_LIF)
    ax2.annotate(f"required closed phase\n= t_release - min_ready\n= {REQUIRED:.3f} T",
                 xy=(REQUIRED * 1.06, 3e-4), fontsize=8, color=C_LIF)
    ax2.set_yscale("log")
    ax2.set_xlabel("duration the gate stays CLOSED per cycle  (units of T)",
                   color=MUTED, fontsize=9)
    ax2.set_ylabel("mean |a_spiking - a_LUT| / action std", color=MUTED, fontsize=9)
    ax2.set_title("One criterion predicts every pass and every failure",
                  color=INK, fontsize=11)
    leg2 = ax2.legend(fontsize=8, framealpha=1.0, facecolor=BG, edgecolor="#ddd8d0",
                      loc="center right", title="waveform")
    leg2.get_title().set_color(INK)
    leg2.get_title().set_fontsize(8)
    for t in leg2.get_texts():
        t.set_color(INK)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.18, lw=0.6)
        for s in ax.spines.values():
            s.set_color("#ddd8d0")
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Inhibitory-oscillator synchronisation of the LUT cell layer "
                 "(20,000 held-out observations)", color=INK, fontsize=12)
    fig.tight_layout()
    out = os.path.join(RES, "inhib_sync.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    print("wrote", out)


if __name__ == "__main__":
    main()
