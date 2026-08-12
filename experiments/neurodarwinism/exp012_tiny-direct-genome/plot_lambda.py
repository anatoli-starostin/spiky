"""exp012 growable nets: what the size penalty buys, at lambda 0.35 vs 0.05.

  A  the two size trajectories. At 0.35 the pool collapses to 4 by round 400 and locks; at
     0.05 it drifts gently to 8 and is still descending at 1500.
  B  the inhibitory pool. At 0.35 it is extinct by round 250; at 0.05 one unit survives.
  C  the leaders. Raw MSE and the affine ceiling move in OPPOSITE directions -- the bigger net
     carries more information and calibrates it worse.
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
RUNS = [("λ = 0.35", "full_run_grow", "g0", "#B4453C"),
        ("λ = 0.05", "full_run_grow_norm", "n0", "#4E79A7")]
H = {n: json.load(open(os.path.join(D, p, f"{t}.json"))) for n, p, t, _ in RUNS}
L = {n: json.load(open(os.path.join(D, p, "final_leader.json"))) for n, p, _, _ in RUNS}
CONST = 34.152

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.135, wspace=0.30)

# ---------------------------------------------------------------- A size
for name, path, tag, col in RUNS:
    h = H[name]
    ax.plot([x["rnd"] for x in h], [x["n_active_mean"] for x in h], "-", lw=2.4, color=col,
            label=f"{name} — ends at {L[name]['size']['n_active']} neurons, "
                  f"{L[name]['size']['n_syn']} syn")
ax.axhline(10, color=MUTE, ls=":", lw=1.5)
ax.text(1490, 10.35, "seeded at 10", fontsize=8.5, color=MUTE, ha="right")
ax.annotate("collapses by round 400,\nthen locked for 1,100 rounds", (450, 4.2),
            textcoords="offset points", xytext=(30, 30), fontsize=8.5, color=C_T,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_T, lw=1.1))
ax.annotate("still descending at 1,500", (1400, 8.3), textcoords="offset points",
            xytext=(-20, 34), fontsize=8.5, color="#4E79A7", ha="right", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4E79A7", lw=1.1))
ax.set_ylim(2.5, 11)
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("active hidden neurons (pool mean)", fontsize=10)
ax.set_title("A · A 7× lighter size penalty, a completely different regime\n"
             "same substrate, same seed, same 1,500 rounds — only λ differs;\n"
             "0.35 collapses to 4 and locks, 0.05 drifts to 8 and is still moving",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

# ---------------------------------------------------------------- B inhibition
for name, path, tag, col in RUNS:
    h = H[name]
    bx.plot([x["rnd"] for x in h], [x["n_inh_mean"] for x in h], "-", lw=2.4, color=col,
            label=f"{name} — {L[name]['size']['n_inh']} inhibitory in the final leader")
bx.axhline(0, color=INK, lw=1.0)
bx.annotate("extinct: 0.02 by round 600", (700, 0.02), textcoords="offset points",
            xytext=(10, 40), fontsize=8.5, color=C_T, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_T, lw=1.1))
bx.annotate("holds at ~1.1", (1300, 1.13), textcoords="offset points", xytext=(-16, 30),
            fontsize=8.5, color="#4E79A7", ha="right", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4E79A7", lw=1.1))
bx.set_ylim(-0.15, 2.3)
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("active inhibitory neurons (pool mean)", fontsize=10)
bx.set_title("B · Inhibition survives at 0.05 and dies at 0.35\n"
             "seeded with 2; at the heavier penalty both are gone by round ~250,\n"
             "at the lighter one exactly one unit pays its way to the end",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- C leaders
NAMES = ["λ = 0.35\n4 neurons\n56 syn", "λ = 0.05\n8 neurons\n108 syn"]
xs = np.arange(2)
w = 0.34
mse = [L[n]["heldout_mse"] for n, *_ in RUNS]
ceil = [L[n]["affine_ceiling"] for n, *_ in RUNS]
cx.bar(xs - w / 2, mse, w, color=[r[3] for r in RUNS], zorder=3, label="held-out MSE")
cx.bar(xs + w / 2, ceil, w, color=[r[3] for r in RUNS], alpha=0.42, zorder=3,
       label="affine ceiling")
for x, v in zip(xs - w / 2, mse):
    cx.text(x, v + 0.35, f"{v:.2f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
for x, v in zip(xs + w / 2, ceil):
    cx.text(x, v + 0.35, f"{v:.2f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
for y, lbl, col in ((23.306, "best fixed-size (crossover) 23.31", MUTE),
                    (24.271, "the seed (lat-inhib) 24.27", MUTE)):
    cx.axhline(y, color=col, ls=":", lw=1.4, zorder=2)
cx.text(1.72, 24.55, "the seed  24.27", fontsize=8.5, color=MUTE, ha="right")
cx.text(1.72, 22.35, "best fixed-size  23.31", fontsize=8.5, color=MUTE, ha="right")
cx.annotate("MSE worse,\nceiling BETTER", (1 + w / 2, 16.22), textcoords="offset points",
            xytext=(28, 26), fontsize=8.5, color=INK, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
cx.set_xticks(xs)
cx.set_xticklabels(NAMES, fontsize=9)
cx.set_xlim(-0.55, 1.75)
cx.set_ylim(0, 30)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.set_title("C · The two measures disagree, and that is the finding\n"
             "the bigger net fits WORSE raw (23.71 vs 23.45) but its affine\n"
             "ceiling is the chapter's best (16.22) — more signal, worse scaled",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 growable nets — normalised weights, constant gain 200 · "
             "the size penalty is the only difference between these two runs",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_lambda.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
