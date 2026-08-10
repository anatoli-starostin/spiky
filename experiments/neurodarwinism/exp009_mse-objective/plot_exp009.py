"""exp009 figure: MSE training trajectory, and both metrics for both objectives."""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
E8 = os.path.join(os.path.dirname(D), "exp008_output-delay-gate")
SEEDS = (0, 1, 2)
COLOR = {"mse": "#59A14F", "tau": "#4E79A7"}
BASELINE = 39.185          # per-seed mean of the pool-mean constant predictor

# measured in exp009_diag / exp009_baselines (see README)
RESULTS = {
    "mse": dict(mse=[37.387, 36.286, 38.892], tau=[0.2965, 0.2981, 0.2047]),
    "tau": dict(mse=[49.048, 49.486, 45.543], tau=[0.3098, 0.2867, 0.3887]),
}

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(14.5, 4.4), gridspec_kw=dict(width_ratios=[2.0, 1, 1]),
    layout="constrained")

for s in SEEDS:
    p = os.path.join(D, f"mse_seed{s}", f"steady_state_mse_s{s}.json")
    h = json.load(open(p))
    ax.plot([r["rnd"] for r in h], [-r["best"] for r in h],
            color=COLOR["mse"], lw=1.0, alpha=0.5)
h0 = json.load(open(os.path.join(D, "mse_seed0", "steady_state_mse_s0.json")))
n = len(h0)
m = np.mean([[-r["best"] for r in json.load(open(
    os.path.join(D, f"mse_seed{s}", f"steady_state_mse_s{s}.json")))][:n] for s in SEEDS], 0)
ax.plot([r["rnd"] for r in h0], m, color=COLOR["mse"], lw=2.2, label="MSE-trained (mean of 3)")
ax.axhline(BASELINE, color="0.35", ls="--", lw=1.2,
           label=f"constant predictor ({BASELINE:.1f})")
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("best in-pool MSE (training batch)", fontsize=10)
ax.set_title("Training-batch MSE: drops in ~15 rounds, then flat for 285 more\n"
             "(dashed line is the held-out constant-predictor level, for scale)",
             fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=9)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
ax.set_ylim(30, 110)

for panel, key, lab, better in ((bx, "mse", "held-out MSE (lower better)", "low"),
                                (cx, "tau", "held-out corrected tau-b (higher better)", "high")):
    for i, obj in enumerate(("tau", "mse")):
        v = np.array(RESULTS[obj][key])
        panel.scatter(np.full(len(v), i), v, s=62, color=COLOR[obj], zorder=3)
        panel.plot([i - 0.18, i + 0.18], [v.mean()] * 2, color="0.2", lw=2.4, zorder=4)
    for a, b in zip(RESULTS["tau"][key], RESULTS["mse"][key]):
        panel.plot([0, 1], [a, b], color="0.8", lw=1.0, zorder=1)
    if key == "mse":
        panel.axhline(BASELINE, color="0.35", ls="--", lw=1.2)
        panel.text(1.42, BASELINE, "constant\npredictor", fontsize=8, color="0.35",
                   va="center")
    panel.set_xticks([0, 1])
    panel.set_xticklabels(["tau-trained\n(exp008)", "MSE-trained\n(exp009)"], fontsize=9)
    panel.set_xlim(-0.4, 1.45)
    panel.set_title(lab, fontsize=10.5, loc="left")
    panel.grid(axis="y", color="0.92", lw=0.6)
    panel.set_axisbelow(True)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)

fig.suptitle("exp009 — MSE on a quantised centred target vs tau-b, both on the gated "
             "[64,80] readout · K=32, 300 rounds, 3 seeds",
             fontsize=11, x=0.005, ha="left")
out = os.path.join(D, "exp009_mse_vs_tau.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
