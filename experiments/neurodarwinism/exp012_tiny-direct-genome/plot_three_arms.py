"""exp012: three winners, three different machines.

  A  final held-out MSE, decomposed. All three land close; the error moves between terms.
  B  ablations. The asexual net DIES without recurrence (+57.9); the crossover net does not
     notice it at all (-0.01) — they are not the same solution wearing different weights.
  C  why: the crossover net puts its long delays on the LAST hop, so it never needs recurrence
     to reach the readout window.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
ARMS = [("mutation only", "full_run", "s0", "#4E79A7"),
        ("+ crossover", "full_run_crossover", "x0", "#59A14F"),
        ("+ crossover\n+ lateral inhib", "full_run_crossover_latinhib", "L0", "#E1A03C")]

LEAD = {a[0]: json.load(open(os.path.join(D, a[1], "final_leader.json"))) for a in ARMS}
ABL2 = json.load(open(os.path.join(D, "full_run_crossover_latinhib", "ablations_both.json")))
A0 = json.load(open(os.path.join(D, "full_run", "analysis_leader.json")))["ablations"]
ABL = {"mutation only": {r["ablation"]: r["heldout_mse"] for r in A0["results"]},
       "+ crossover": {k: v for k, v in ABL2["crossover"].items() if k != "base"},
       "+ crossover\n+ lateral inhib": {k: v for k, v in ABL2["lat-inhib"].items()
                                        if k != "base"}}
CONST = 34.152

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.055, right=0.988, top=0.70, bottom=0.155, wspace=0.30)

# ---------------------------------------------------------------- A decomposition
lbl3 = ["bias²", "scale error", "residual"]
cols = ["#B07AA1", "#F1A340", "#4E79A7"]
xs = np.arange(3)
for k, (name, _p, _t, _c) in enumerate(ARMS):
    G = LEAD[name]
    bot = 0.0
    for j, key in enumerate(("bias2", "scale_err", "resid")):
        v = G[key]
        ax.bar(xs[k], v, 0.55, bottom=bot, color=cols[j], zorder=3, edgecolor="white",
               linewidth=2, label=lbl3[j] if k == 0 else None)
        ax.text(xs[k], bot + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        bot += v
    ax.text(xs[k], bot + 0.6, f"{G['heldout_mse']:.2f}", ha="center", fontsize=10,
            color=INK, fontweight="bold")
    ax.plot([xs[k]], [G["affine_ceiling"]], "*", ms=17, color="#59A14F", mec="white",
            mew=1.0, zorder=7)
ax.axhline(CONST, color=C_T, ls="--", lw=1.6, zorder=6)
ax.text(-0.42, CONST + 0.5, f"constant  {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
ax.set_xticks(xs)
ax.set_xticklabels([a[0] for a in ARMS], fontsize=9)
ax.set_ylim(0, 44)
ax.set_ylabel("held-out MSE", fontsize=10)
ax.set_title("A · All three land within 1 MSE point of each other\n"
             "★ = affine ceiling — 16.7 / 16.9 for the two crossover arms, i.e.\n"
             "after recalibration they are the same quality",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- B ablations
KEYS = [("no_recurrence", "cut recurrence"), ("prune50", "prune weakest 50 %"),
        ("no_inhibition", "zero all inhibition")]
w = 0.26
xs = np.arange(len(KEYS))
for k, (name, _p, _t, col) in enumerate(ARMS):
    d = [ABL[name][key] - LEAD[name]["heldout_mse"] for key, _ in KEYS]
    bxs = xs + (k - 1) * w
    bx.bar(bxs, d, w * 0.88, color=col, zorder=3, label=name.replace("\n", " "))
    for x, v in zip(bxs, d):
        bx.text(x, v + (1.6 if v >= 0 else -2.6), f"{v:+.1f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
bx.axhline(0, color=INK, lw=1.0)
bx.set_yscale("symlog", linthresh=1.0)
bx.set_yticks([0, 1, 10, 60])
bx.set_yticklabels(["0", "+1", "+10", "+60"])
bx.set_xticks(xs)
bx.set_xticklabels([k[1] for k in KEYS], fontsize=9)
bx.set_ylim(-4, 200)
bx.set_ylabel("Δ held-out MSE when ablated (symlog)", fontsize=10)
bx.set_title("B · They are NOT the same solution\n"
             "mutation-only dies without recurrence (+57.9); the crossover net\n"
             "loses 44 recurrent synapses and does not notice (−0.01)",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, columnspacing=1.1)

# ---------------------------------------------------------------- C output delays
import sys                                                        # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(D), "src"))
import tiny_snn as T                                              # noqa: E402
from tiny_evolve import load_ckpt                                 # noqa: E402
for k, (name, path, tag, col) in enumerate(ARMS):
    pool, ewma, *_ = load_ckpt(os.path.join(D, path, f"ck_{tag}.npz"))
    fin = np.where(np.isfinite(ewma))[0]
    g = pool[int(fin[np.argmin(ewma[fin])])]
    eo = g["mask"][:, T.COL_OUT]
    d = g["delay"][:, T.COL_OUT][eo]
    cx.hist(d, bins=np.arange(0, 70, 6), histtype="step", lw=2.4, color=col,
            label=f"{name.replace(chr(10), ' ')} — median {np.median(d):.0f}")
cx.axvspan(49, 64, color="#59A14F", alpha=0.09, zorder=0)
cx.text(23, 5.4, "the shaded band is long\nenough to reach [64,96)\nin ONE hop",
        fontsize=8.5, color="#3d7a35", ha="center", fontweight="bold")
cx.set_xlabel("delay on the final hop (hidden → output)", fontsize=10)
cx.set_ylabel("synapses", fontsize=10)
cx.set_title("C · …and this is why\n"
             "the crossover nets load the LAST hop with long delays, so\n"
             "timing decouples from computation; mutation-only spreads it thin",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.2, loc="upper left")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — three winners, three different machines · 33 neurons, every synapse a "
             "gene, no plasticity", fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_three_arms.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
