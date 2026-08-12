"""exp012 full run — one seed, pool 512, 1500 rounds, uniform [1,64], no plasticity.

  A  the trajectory against every reference this chapter has: chance, the pool-64 smoke best,
     the split-range best, and exp009's 800-excitatory STDP reservoir
  B  the three reporting-only metrics — mean |r|, silence, tau. None of them is in the
     fitness; the fitness is the raw offset MSE alone.
  C  where the final MSE goes, and how much of it a readout recalibration could remove
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, "full_run")
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
C_MAIN, C_ALT, C_3 = "#4E79A7", "#59A14F", "#B07AA1"

H = json.load(open(os.path.join(R, "s0.json")))
F = json.load(open(os.path.join(R, "s0_final.json")))
# The headline model is the FINAL EWMA LEADER -- the member selection settles on, chosen on
# training batches alone -- not the minimum of any held-out curve. See tiny_final_eval.py.
G = json.load(open(os.path.join(R, "final_leader.json")))
CONST = F["constant_baseline_val"]

rnd = np.array([h["rnd"] for h in H])
mmin = np.array([h["mse_min"] for h in H])
bh = np.array([h["best_heldout"] if h["best_heldout"] is not None else np.nan for h in H])
ev = [h for h in H if "ewma_leader_heldout_mse" in h]

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.72, bottom=0.115, wspace=0.31)

# ---------------------------------------------------------------- A  trajectory
# NOT a running minimum, and deliberately so. `best` is the genome that won the round's
# TRAINING batch; the curve shows that genome's held-out score, which can go UP when a new
# batch champion generalises worse. Taking a running min over it would be selecting on
# held-out through the back door. The EWMA-leader curve below is the honest one: the member
# selection has actually settled on, chosen on training batches alone.
ax.plot(rnd, mmin, "-", lw=1.1, color=C_MAIN, alpha=0.45, label="pool best (batch)")
ax.plot(rnd, bh, "-", lw=1.6, color=C_MAIN, alpha=0.55,
        label="held-out of the batch champion")
ax.plot([h["rnd"] for h in ev], [h["ewma_leader_heldout_mse"] for h in ev], "-", lw=2.6,
        color=C_3, label="held-out of the EWMA leader")
# The three references sit at 34.2 / 30.2 / 27.4 -- close enough on a log axis that inline
# labels collide with each other AND with the curves, which spend the second half of the run
# in exactly that band. They go in the legend instead.
for y, col, ls, lbl in ((CONST, C_T, "--", f"constant predictor  {CONST:.1f}"),
                        (30.2, "#E1A03C", ":", "pool-64 smoke best  30.2"),
                        (27.4, C_ALT, ":", "split-range best  27.4")):
    ax.axhline(y, color=col, ls=ls, lw=1.6, zorder=4, label=lbl)
ax.set_yscale("log")
ax.set_ylim(min(20, np.nanmin(bh) * 0.9), 200)
ax.set_yticks([20, 30, 40, 60, 100, 200])
ax.set_yticklabels(["20", "30", "40", "60", "100", "200"])
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("held-out MSE (offsets 0..31)", fontsize=10)
ax.set_title(f"A · One seed, pool 512, {len(H)} rounds\n"
             f"the model selection settles on scores {G['heldout_mse']:.2f} held-out against "
             f"a constant\npredictor of {CONST:.1f} — 8× the population, no plasticity anywhere",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- B  reporting-only metrics
bx2 = bx.twinx()
w = 25
def smooth(v):
    v = np.asarray(v, float)
    k = np.ones(w) / w
    return np.convolve(v, k, mode="valid")
xs = rnd[w - 1:]
bx.plot(xs, smooth([h["mean_abs_r"] for h in H]), "-", lw=2.2, color=C_MAIN,
        label="mean |r| (batch best)")
bx.plot(xs, smooth([h["tau_of_best"] for h in H]), "-", lw=2.0, color=C_3,
        label="Kendall tau (batch best)")
bx.axhline(0.32, color=MUTE, ls=":", lw=1.5)
bx.text(30, 0.335, "exp009's mean |r| ≈ 0.32", fontsize=8.5, color=MUTE, ha="left")
bx2.plot(xs, 100 * smooth([h["silent"] for h in H]), "-", lw=2.0, color=C_T,
         label="% silent (pool mean)")
bx2.set_ylabel("% of outputs silent", fontsize=10, color=C_T)
bx2.tick_params(axis="y", labelcolor=C_T, labelsize=9)
bx2.set_ylim(0, max(40, 100 * H[0]["silent"] * 1.15))
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("mean |r|  ·  Kendall tau", fontsize=10)
bx.set_title("B · The reporting-only metrics — none is in the fitness\n"
             "selection sees the raw offset MSE and nothing else; |r|, tau and\n"
             "the silent fraction are recorded to show what the MSE is buying",
             fontsize=10, loc="left", color=INK)
h1, l1 = bx.get_legend_handles_labels()
h2, l2 = bx2.get_legend_handles_labels()
bx.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center right")
for sp in ("top",):
    bx2.spines[sp].set_visible(False)

# ---------------------------------------------------------------- C  decomposition
lbl = ["bias²\n(wrong centre)", "scale error\n(too narrow)", "residual\n(real error)"]
cols = ["#B07AA1", "#F1A340", "#4E79A7"]
bot = 0.0
for k, key in enumerate(("bias2", "scale_err", "resid")):
    v = G[key]
    cx.bar(0, v, 0.46, bottom=bot, color=cols[k], zorder=3, edgecolor="white", linewidth=2,
           label=lbl[k])
    if v > 2:
        cx.text(0, bot + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")
    bot += v
REF = [("split-range\nbest (v2)", 27.4, C_ALT), ("gate\nbest (v1)", 32.1, "#7FA8C9"),
       ("exp009\n(rescaled)", 37.52 / 39.19 * CONST, MUTE)]
for k, (nm, v, col) in enumerate(REF):
    cx.bar(1.15 + k * 0.75, v, 0.46, color=col, alpha=0.85, zorder=3)
    cx.text(1.15 + k * 0.75, v + 0.9, f"{v:.1f}", ha="center", fontsize=9, color=INK)
cx.axhline(CONST, color=C_T, ls="--", lw=1.7, zorder=6)
cx.text(-0.42, CONST + 1.0, f"constant  {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
cx.plot([0], [G["affine_ceiling"]], "*", ms=22, color="#59A14F", mec="white", mew=1.0,
        zorder=7)
cx.annotate(f"{G['affine_ceiling']:.1f} — affine ceiling:\nsame net, readout recalibrated",
            (0, G["affine_ceiling"]), textcoords="offset points", xytext=(26, 20),
            fontsize=8.5, color=INK, ha="left", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
cx.set_xticks([0] + [1.15 + k * 0.75 for k in range(3)])
cx.set_xticklabels([f"this run\n{G['heldout_mse']:.1f}"] + [r[0] for r in REF], fontsize=9)
cx.set_xlim(-0.5, 3.2)
cx.set_ylim(0, 54)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.set_title(f"C · Where the {G['heldout_mse']:.1f} goes, and how it compares\n"
             f"mean |r| {G['mean_abs_r']:.2f} · silence {100 * G['silent']:.1f} % · "
             f"tau {G['tau']:+.2f}\n"
             "every bar is the same target and the same held-out split",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)
bx.spines["right"].set_visible(True)

fig.suptitle("exp012 full run — 33 neurons, every synapse a gene, no plasticity · "
             "pool 512 × 1500 rounds, uniform delays [1,64]",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_full_run.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
