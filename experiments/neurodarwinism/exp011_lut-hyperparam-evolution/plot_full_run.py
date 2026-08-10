"""exp011 full run: 3 seeds x 60 rounds x K=16. Why nothing dominated the teacher.

  A  params vs fit -- evolution DOES find teacher-matching nets with fewer parameters
  B  throughput vs fit -- and every one of them costs ~3x the teacher to run
  C  the reason: fitness penalises params only, so the search rides tables_per_head
     straight into its cap and throughput follows it up
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(D, "full_run")
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_TEACH = "#B4453C"
TPH_CAP = 128

fin = {s: json.load(open(os.path.join(RUN, f"seed{s}", f"lut_evolve_s{s}_final.json")))
       for s in SEEDS}
hist = {s: json.load(open(os.path.join(RUN, f"seed{s}", f"lut_evolve_s{s}.json")))
        for s in SEEDS}
T = fin[0]["teacher"]
t_mse = float(np.mean([fin[s]["teacher"]["mse"] for s in SEEDS]))

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(15.4, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.0]),
    layout="constrained")

for panel, key, xlab, title in (
        (ax, "params", "parameters",
         "A · Evolution DOES beat the teacher on parameters\n"
         "every seed finds teacher-matching nets at ~9k params\n"
         "against the teacher's 12,288"),
        (bx, "throughput", "throughput (weights read per forward)",
         "B · …and NONE of them is cheaper to RUN\n"
         "78 candidates matched the teacher's fit with fewer params;\n"
         "0 did so with lower throughput, out of 2,880")):
    for s in SEEDS:
        pts = fin[s]["pareto_joint"]
        panel.plot([p[key] for p in pts], [p["mse"] for p in pts], "o", ms=4.5,
                   color=C_SEED[s], alpha=0.75, label=f"seed {s}", zorder=3)
    panel.axhline(t_mse, color=C_TEACH, ls="--", lw=1.3, zorder=2)
    panel.plot([T[key]], [t_mse], "*", ms=20, color=C_TEACH, zorder=5,
               markeredgecolor="white", markeredgewidth=0.8)
    panel.annotate("TEACHER", (T[key], t_mse), textcoords="offset points",
                   xytext=(8, 10), fontsize=9, color=C_TEACH, fontweight="bold")
    panel.set_xscale("log")
    panel.set_yscale("log")
    panel.set_xlabel(xlab, fontsize=10)
    panel.set_ylabel("held-out MSE", fontsize=10)
    panel.set_title(title, fontsize=10, loc="left", color=INK)
    panel.legend(frameon=False, fontsize=8.5, loc="upper right")

# the "dominates" quadrant on panel A: cheaper AND no worse fit
ax.axvspan(1, T["params"], color=C_TEACH, alpha=0.06, zorder=0)
ax.text(700, t_mse * 0.62, "cheaper than the teacher\n(and plenty of hits)",
        fontsize=8.5, color=C_TEACH)
bx.axvspan(1, T["throughput"], color=C_TEACH, alpha=0.06, zorder=0)
# The region is NOT empty -- 38 of 2,880 candidates landed in it. None reached the teacher's
# fit: the best got to 0.0289 against 0.0253. Say that, rather than "empty".
bx.text(3.5, 0.055, "cheaper to run than the teacher:\n38 candidates visited,\n"
                    "best fit 0.0289 — none matched", fontsize=8.5, color=C_TEACH)

# ---------------------------------------------------------------- C: the cap
for s in SEEDS:
    r = [h["rnd"] for h in hist[s]]
    cx.plot(r, [np.median(h["tph_vec"]) for h in hist[s]], "-", lw=2.0,
            color=C_SEED[s], label=f"seed {s}", zorder=3)
cx.axhline(TPH_CAP, color=INK, ls=":", lw=1.6, zorder=4)
cx.text(28, TPH_CAP * 0.90, f"TPH_RANGE cap = {TPH_CAP}", fontsize=9, color=INK, va="top")
cx.set_yscale("log")
cx.set_ylim(45, 190)
cx.set_xlabel("round", fontsize=10)
cx.set_ylabel("pool median tables_per_head", fontsize=10)
cx.set_title("C · The search rides tables into the cap\n"
             "47 of 48 final members sit at tph = 128, and throughput\n"
             "= tph × 6 rides up with it. Fitness never charged for it.",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="lower right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp011 full run — 3 seeds × 60 rounds × K=16, 2,880 candidates trained · "
             "0 Pareto-dominate the teacher",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp011_full_run.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
