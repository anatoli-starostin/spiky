"""exp012 pre-flight: is a 33-neuron directly-evolved net worth a full run?

  A  the operating point -- round-0 aliveness across w_max, and why 30 is the pick
  B  three 400-round smoke runs against the constant predictor, and against what exp009's
     800-excitatory STDP reservoir reached on this exact target
  C  where the MSE actually goes: over half of it is readout MISCALIBRATION, not error
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(D, "sanity")
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_T = "#B4453C"

PF = json.load(open(os.path.join(S, "preflight.json")))
F = {s: json.load(open(os.path.join(S, f"raw{s}_final.json"))) for s in SEEDS}
H = {s: json.load(open(os.path.join(S, f"raw{s}.json"))) for s in SEEDS}
DG = {s: json.load(open(os.path.join(S, f"diag_raw{s}.json"))) for s in SEEDS}
D0 = json.load(open(os.path.join(S, "diag_smoke0.json")))

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.72, bottom=0.115, wspace=0.30)

# ---------------------------------------------------------------- A  operating point
al = PF["alive"]
w = np.array([r["w_max"] for r in al])
ax.plot(w, [r["silent_frac"] for r in al], "-o", ms=5, lw=2.0, color=C_T,
        label="fraction of outputs silent")
ax.plot(w, [r["n_distinct_mean"] / 32 for r in al], "-o", ms=5, lw=2.0, color="#4E79A7",
        label="distinct offsets used / 32")
ax.plot(w, np.array([r["mse_std_across_pool"] for r in al]) / 32, "-o", ms=5, lw=2.0,
        color="#59A14F", label="spread of pool fitness (sd/32)")
ax.axvline(30, color=MUTE, ls="--", lw=1.3)
ax.annotate("w_max 30\nchosen", (30, 0.86), textcoords="offset points", xytext=(8, 0),
            fontsize=9, color=INK, fontweight="bold")
ax.set_xscale("log")
ax.set_xticks(w)
ax.set_xticklabels([f"{int(x)}" for x in w])
ax.set_xlabel("w_max (initial weight scale)", fontsize=10)
ax.set_ylabel("fraction", fontsize=10)
ax.set_title("A · Round 0 is alive, and the window is wide\n"
             "below 20 the net is silent, above 60 the pool goes flat;\n"
             "30 keeps 99 % of outputs firing AND the most fitness spread",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="center right")

# ---------------------------------------------------------------- B  it learns
for s in SEEDS:
    r = [h["rnd"] for h in H[s]]
    m = [h["mse_min"] for h in H[s]]
    bx.plot(r, m, "-", lw=1.9, color=C_SEED[s], alpha=0.9,
            label=f"seed {s} — held-out {F[s]['best']['heldout_mse']:.1f}, "
                  f"tau {F[s]['best']['heldout_tau']:+.2f}")
base = np.mean([F[s]["constant_baseline_val"] for s in SEEDS])
exp009 = 37.52 / 39.19 * base          # exp009's result as a ratio to ITS OWN chance level
bx.axhline(base, color=C_T, ls="--", lw=1.6, zorder=4)
bx.text(2, base + 1.2, f"constant predictor  {base:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
bx.axhline(exp009, color=MUTE, ls=":", lw=1.6, zorder=4)
bx.text(2, exp009 - 1.0, "exp009's 800-excitatory STDP reservoir,\n"
        "rescaled to this split (37.52 of its own 39.19)", fontsize=8.5, color=MUTE,
        ha="left", va="top")
bx.set_ylim(27.5, 62)
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("best MSE in the pool (batch)", fontsize=10)
bx.set_title("B · 400 rounds = 99 s, and two of three seeds clear chance\n"
             "33 neurons with NO plasticity reach the ratio-to-chance an 800-\n"
             "excitatory STDP reservoir needed 300 rounds for — still descending",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- C  where the MSE goes
lbl = ["bias²\n(wrong centre)", "scale error\n(too narrow)", "residual\n(real error)"]
g0 = D0["summary"]
g4 = DG[0]["summary"]
xs = np.arange(2)
bot = np.zeros(2)
cols = ["#B07AA1", "#F1A340", "#4E79A7"]
for k, key in enumerate(("bias2", "scale_err", "resid")):
    v = np.array([g0[key], g4[key]])
    cx.bar(xs, v, 0.52, bottom=bot, color=cols[k], label=lbl[k], zorder=3,
           edgecolor="white", linewidth=2)
    for i in range(2):
        if v[i] > 3:
            cx.text(xs[i], bot[i] + v[i] / 2, f"{v[i]:.1f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bot += v
cx.axhline(g4["constant_val"], color=C_T, ls="--", lw=1.6, zorder=5)
cx.text(1.34, g4["constant_val"] + 0.7, f"constant  {g4['constant_val']:.1f}", fontsize=9,
        color=C_T, fontweight="bold", ha="left")
cx.plot([1], [g4["mse_after_affine"]], "*", ms=22, color="#59A14F", mec="white", mew=1.0,
        zorder=7)
cx.annotate(f"{g4['mse_after_affine']:.1f} — same net,\nreadout recalibrated:\n"
            "35 % below chance",
            (1, g4["mse_after_affine"]), textcoords="offset points", xytext=(26, -36),
            fontsize=8.5, color=INK, ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
cx.set_xticks(xs)
cx.set_xticklabels(["60 rounds\nMSE 46.4", "400 rounds\nMSE 32.1"], fontsize=9.5)
cx.set_xlim(-0.55, 2.55)
cx.set_ylim(0, 52)
cx.set_ylabel("held-out MSE, decomposed", fontsize=10)
cx.set_title("C · Over half the MSE is a coding artefact\n"
             f"evolution fixes the centre (bias² {g0['bias2']:.1f}→{g4['bias2']:.1f});\n"
             "what it cannot fix is span — sd 2.1 against the target's 6.2",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 pre-flight — 17→8 exc + 2 inh→6, every synapse a gene, no STDP anywhere",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_preflight.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
