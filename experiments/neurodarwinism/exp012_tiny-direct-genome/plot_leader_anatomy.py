"""exp012: what the best evolved network actually is.

  A  what each output reads — permutation sensitivity, 17 inputs x 6 outputs. The solution is
     not distributed: each output is driven by one or two observation dimensions, and 7 of the
     17 inputs are dead weight.
  B  ablations. Recurrence is the whole machine; inhibition is very nearly decorative.
  C  the outputs use a narrower band than the target does — the scale-error term, made visible.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, "full_run", "analysis_leader.json")))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.05, right=0.985, top=0.70, bottom=0.13, wspace=0.42)

# ---------------------------------------------------------------- A sensitivity
S = np.array(R["function"]["sensitivity_mean_abs_delta_offset"])       # [17, 6]
# sequential single-hue ramp: this is a magnitude, so one hue light->dark, never a rainbow
cmap = LinearSegmentedColormap.from_list("seq", ["#f7f9fb", "#c6d9e8", "#7FA8C9", "#4E79A7",
                                                 "#2b4a6b"])
im = ax.imshow(S.T, aspect="auto", cmap=cmap, vmin=0)
ax.set_xticks(range(17))
ax.set_xticklabels([f"{i}" for i in range(17)], fontsize=8)
ax.set_yticks(range(6))
ax.set_yticklabels([f"OUT{o}" for o in range(6)], fontsize=9)
ax.set_xlabel("observation dimension", fontsize=10)
dead = [j for j, v in R["function"]["input_total_influence"] if v < 0.01]
for j in dead:
    ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 6, fill=False, ec=C_T, lw=1.4, ls=":"))
for o in range(6):
    j = int(np.argmax(S[:, o]))
    ax.text(j, o, f"{S[j, o]:.1f}", ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.015)
cb.set_label("mean |Δ offset| when shuffled", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.set_title("A · Each output reads one or two inputs, not all of them\n"
             f"the 7 dotted columns are observation dims with NO influence at all;\n"
             "OUT1 is driven by dim 13 alone — 130× its next-strongest input",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B ablations
base = R["ablations"]["baseline_mse"]
chance = R["ablations"]["constant_baseline"]
NAMES = {"no_recurrence": "remove hidden→hidden\nrecurrence",
         "prune50": "prune the weakest\n50 % of weights",
         "no_inhibition": "zero ALL\ninhibition",
         "ko_inh0": "knock out I0", "ko_inh1": "knock out I1"}
res = sorted(R["ablations"]["results"], key=lambda x: -x["delta"])
ys = np.arange(len(res))
cols = [C_T if r["heldout_mse"] > chance else ("#E1A03C" if r["delta"] > 1 else "#59A14F")
        for r in res]
bx.barh(ys, [r["heldout_mse"] for r in res], 0.6, color=cols, zorder=3)
for y, r in zip(ys, res):
    bx.text(r["heldout_mse"] + 1.2, y, f"{r['heldout_mse']:.1f}   ({r['delta']:+.2f})",
            va="center", fontsize=9, color=INK)
bx.axvline(base, color="#4E79A7", ls="-", lw=2.0, zorder=5)
bx.axvline(chance, color=C_T, ls="--", lw=1.6, zorder=5)
bx.text(base - 1.5, -0.62, f"intact  {base:.1f}", fontsize=9, color="#4E79A7",
        fontweight="bold", ha="right", va="center")
bx.text(chance + 1.5, -0.62, f"chance  {chance:.1f}", fontsize=9, color=C_T,
        fontweight="bold", va="center")
bx.set_yticks(ys)
bx.set_yticklabels([NAMES[r["ablation"]] for r in res], fontsize=9)
bx.set_xlim(0, 100)
bx.set_ylim(len(res) - 0.4, -1.0)
bx.set_xlabel("held-out MSE after the ablation", fontsize=10)
bx.set_title("B · Recurrence is the machine; inhibition is decorative\n"
             "cutting recurrence costs +57.9 MSE and lands far past chance;\n"
             "deleting ALL 8 inhibitory synapses costs +0.18",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C offset ranges
per = R["function"]["per_output"]
xs = np.arange(6)
tgt_sd = 6.16
for o, p in enumerate(per):
    cx.vlines(o - 0.14, p["offset_min"], p["offset_max"], color="#7FA8C9", lw=7,
              alpha=0.55, zorder=3)
    cx.vlines(o - 0.14, p["offset_mean"] - p["offset_sd"], p["offset_mean"] + p["offset_sd"],
              color="#4E79A7", lw=7, zorder=4)
    cx.plot([o - 0.14], [p["offset_mean"]], "_", ms=14, color="white", mew=2.4, zorder=6)
    cx.vlines(o + 0.14, 0, 31, color="#e7d3e3", lw=7, alpha=0.9, zorder=3)
    cx.vlines(o + 0.14, 14.21 - tgt_sd, 14.21 + tgt_sd, color="#B07AA1", lw=7, zorder=4)
cx.plot([], [], "-", lw=7, color="#4E79A7", label="network ±1 sd (pale = full range)")
cx.plot([], [], "-", lw=7, color="#B07AA1", label="target ±1 sd (pale = full range)")
for o, p in enumerate(per):
    cx.text(o, 32.6, f"r {p['r']:.2f}", ha="center", fontsize=8.5, color=INK,
            fontweight="bold")
cx.set_xticks(xs)
cx.set_xticklabels([f"OUT{o}" for o in range(6)], fontsize=9)
cx.set_ylim(-1, 44)
cx.set_ylabel("first-spike offset (0..31)", fontsize=10)
cx.set_title("C · The outputs cannot span the target's range\n"
             "network sd 2.6–5.2 against the target's 6.2, on every\n"
             "dimension — the scale-error term, and it is structural",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)

for p in (bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)
for p in (ax, bx, cx):
    p.tick_params(labelsize=9)

fig.suptitle("exp012 — anatomy of the best evolved network: 33 neurons, 140 synapses, "
             "no plasticity, held-out MSE 25.2 vs chance 34.2",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_leader_anatomy.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
