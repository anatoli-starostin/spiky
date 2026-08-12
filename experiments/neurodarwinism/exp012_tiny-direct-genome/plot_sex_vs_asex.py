"""exp012: does uniform crossover buy anything over mutation alone?

Identical config on both arms — 1 seed, pool 512, 1500 rounds, batch 256, w_max 60, uniform
delays [1,64], no plasticity, selection on raw offset MSE alone. The ONLY difference is
whether an offspring gets one parent or two.

  A  the two trajectories, against every reference this chapter has
  B  mean |r| and the silent fraction — what the MSE is actually buying, on both arms
  C  the final models decomposed, side by side with the earlier substrates
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
ARMS = [("mutation only", os.path.join(D, "full_run"), "s0", "#4E79A7"),
        ("+ uniform crossover", os.path.join(D, "full_run_crossover"), "x0", "#59A14F")]

A = {}
for lbl, path, tag, col in ARMS:
    A[lbl] = dict(
        hist=json.load(open(os.path.join(path, f"{tag}.json"))),
        final=json.load(open(os.path.join(path, f"{tag}_final.json"))),
        lead=json.load(open(os.path.join(path, "final_leader.json"))),
        col=col)
CONST = A["mutation only"]["final"]["constant_baseline_val"]

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.72, bottom=0.115, wspace=0.31)

# ---------------------------------------------------------------- A trajectories
for lbl, path, tag, col in ARMS:
    H = A[lbl]["hist"]
    ev = [h for h in H if "ewma_leader_heldout_mse" in h]
    ax.plot([h["rnd"] for h in H], [h["mse_min"] for h in H], "-", lw=0.9, color=col,
            alpha=0.28)
    ax.plot([h["rnd"] for h in ev], [h["ewma_leader_heldout_mse"] for h in ev], "-", lw=2.4,
            color=col, label=f"{lbl} — final {A[lbl]['lead']['heldout_mse']:.2f}")
for y, col, lbl in ((CONST, C_T, f"constant predictor  {CONST:.1f}"),
                    (27.4, "#8C6BB1", "split-range best  27.4")):
    ax.axhline(y, color=col, ls="--", lw=1.5, zorder=4)
    ax.text(A["mutation only"]["hist"][-1]["rnd"], y + 0.6, lbl, fontsize=8.5, color=col,
            ha="right", fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(20, 160)
ax.set_yticks([20, 30, 40, 60, 100, 150])
ax.set_yticklabels(["20", "30", "40", "60", "100", "150"])
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("held-out MSE of the EWMA leader", fontsize=10)
ax.set_title("A · One parent or two — everything else identical\n"
             "pool 512 × 1500 rounds, uniform delays [1,64], no plasticity;\n"
             "faint lines are the pool best on the training batch",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- B what it buys
bx2 = bx.twinx()
w = 25
for lbl, path, tag, col in ARMS:
    H = A[lbl]["hist"]
    k = np.ones(w) / w
    xs = np.array([h["rnd"] for h in H])[w - 1:]
    bx.plot(xs, np.convolve([h["mean_abs_r"] for h in H], k, "valid"), "-", lw=2.2, color=col,
            label=f"mean |r| — {lbl}")
    bx2.plot(xs, 100 * np.convolve([h["silent"] for h in H], k, "valid"), ":", lw=1.8,
             color=col)
bx.axhline(0.32, color=MUTE, ls="--", lw=1.4)
bx.text(A["mutation only"]["hist"][-1]["rnd"], 0.328, "exp009's mean |r| ≈ 0.32", fontsize=8.5,
        color=MUTE, ha="right")
bx2.set_ylabel("% silent (dotted)", fontsize=10, color=MUTE)
bx2.tick_params(axis="y", labelcolor=MUTE, labelsize=9)
bx2.set_ylim(0, 30)
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("mean |r|  (solid)", fontsize=10)
bx.set_title("B · What the MSE is buying — neither is in the fitness\n"
             "selection sees the raw offset MSE alone; |r| and the silent\n"
             "fraction are recorded to show what it drags along with it",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="lower right")
bx2.spines["top"].set_visible(False)

# ---------------------------------------------------------------- C decomposition
lbl3 = ["bias²", "scale error", "residual"]
cols = ["#B07AA1", "#F1A340", "#4E79A7"]
xs = [0, 0.85]
for k, (lbl, path, tag, col) in enumerate(ARMS):
    G = A[lbl]["lead"]
    bot = 0.0
    for j, key in enumerate(("bias2", "scale_err", "resid")):
        v = G[key]
        cx.bar(xs[k], v, 0.5, bottom=bot, color=cols[j], zorder=3, edgecolor="white",
               linewidth=2, label=lbl3[j] if k == 0 else None)
        if v > 2:
            cx.text(xs[k], bot + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        bot += v
    cx.plot([xs[k]], [G["affine_ceiling"]], "*", ms=18, color="#59A14F", mec="white", mew=1.0,
            zorder=7)
REF = [("split-range\nbest (v2)", 27.4, "#8C6BB1"), ("gate\nbest (v1)", 32.1, "#7FA8C9"),
       ("exp009\n(rescaled)", 37.52 / 39.19 * CONST, MUTE)]
for k, (nm, v, col) in enumerate(REF):
    cx.bar(2.0 + k * 0.7, v, 0.5, color=col, alpha=0.85, zorder=3)
    cx.text(2.0 + k * 0.7, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, color=INK)
cx.axhline(CONST, color=C_T, ls="--", lw=1.7, zorder=6)
cx.text(3.7, CONST + 0.8, f"constant  {CONST:.1f}", fontsize=9, color=C_T, fontweight="bold",
        ha="right")
cx.text(0.42, 32.2, "★ = affine ceiling (same net, readout recalibrated)", fontsize=8.5,
        color=INK, ha="center")
cx.set_xticks(xs + [2.0 + k * 0.7 for k in range(3)])
cx.set_xticklabels([f"mutation\nonly {A['mutation only']['lead']['heldout_mse']:.1f}",
                    f"+ crossover\n{A['+ uniform crossover']['lead']['heldout_mse']:.1f}"]
                   + [r[0] for r in REF], fontsize=8.5)
cx.set_xlim(-0.45, 3.75)
cx.set_ylim(0, 44)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.set_title("C · The final models, decomposed and placed\n"
             "every bar is the same target, the same held-out split and the\n"
             "same properly-selected model (final EWMA leader)",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)
bx.spines["right"].set_visible(True)

fig.suptitle("exp012 — mutation alone vs uniform crossover · 33 neurons, every synapse a "
             "gene, no plasticity anywhere", fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_sex_vs_asex.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
