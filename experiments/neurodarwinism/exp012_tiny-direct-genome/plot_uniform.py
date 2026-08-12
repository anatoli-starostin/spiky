"""exp012 with ONE uniform delay range [1,32] and no output gate — why it fails.

  A  the arithmetic: which inputs can even reach the readout window on two hops. Under a
     uniform range this is 0.1 % for the earliest input and 51.6 % for the latest — a 528x
     bias against exactly the observations the latency code puts first.
  B  evolution DOES solve "fire in the window at all" unaided: 90 % silent -> 4 %. It just
     spends most of the run doing it.
  C  what survives: the signal the net carries collapses, and every seed lands above chance.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(D, "sanity_v1_gate64-80")
V2 = os.path.join(D, "sanity_v2_split1-32_32-64")
V3 = os.path.join(D, "sanity")
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_T = "#B4453C"
CONST = 34.152                      # every diagnostic runs on seed 0's held-out split

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.72, bottom=0.115, wspace=0.31)

# ---------------------------------------------------------------- A  the arithmetic
d = np.arange(1, 33)
D1, D2 = np.meshgrid(d, d)
uni = np.array([(((t + D1 + D2) >= 64) & ((t + D1 + D2) < 96)).mean() for t in range(32)])
d2s = np.arange(32, 65)
D1b, D2b = np.meshgrid(d, d2s)
spl = np.array([(((t + D1b + D2b) >= 64) & ((t + D1b + D2b) < 96)).mean() for t in range(32)])
ax.plot(range(32), 100 * uni, "-", lw=2.6, color=C_T, label="uniform [1,32] — this run")
ax.plot(range(32), 100 * spl, "--", lw=2.0, color="#4E79A7",
        label="split [1,32] / [32,64] — previous")
ax.plot([0], [100 * uni[0]], "o", ms=9, color=C_T, mec="white", mew=1.4, zorder=5)
ax.annotate("largest observations\nfire at tick 0 — and only\n1 of 1024 delay pairs\n"
            "reaches the window", (0, 100 * uni[0]), textcoords="offset points",
            xytext=(16, 46), fontsize=8.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.annotate("smallest\nobservations:\n51.6 %", (31, 100 * uni[31]), textcoords="offset points",
            xytext=(-14, 6), fontsize=8.5, color=C_T, ha="right")
ax.set_xlabel("input spike tick  (early = LARGE observation)", fontsize=10)
ax.set_ylabel("% of (d₁,d₂) pairs landing in [64,96)", fontsize=10)
ax.set_title("A · The uniform range is not neutral — it is a 528× bias\n"
             "a two-hop path from the earliest input reaches the readout\n"
             "window 1 time in 1024; from the latest, 1 time in 2",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

# ---------------------------------------------------------------- B  it learns to fire
for s in SEEDS:
    H = json.load(open(os.path.join(V3, f"w30_{s}.json")))
    bx.plot([h["rnd"] for h in H], [100 * h["silent"] for h in H], "-", lw=2.0,
            color=C_SEED[s], label=f"seed {s}")
bx.axhline(5, color=MUTE, ls=":", lw=1.4)
bx.text(4, 7.5, "5 % silent", fontsize=8.5, color=MUTE, ha="left")
bx.annotate("90 % of outputs never fire\nin the window at round 0", (4, 90),
            textcoords="offset points", xytext=(24, -6), fontsize=8.5, color=INK,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
bx.annotate("…and 4 % by round 400,\nwith no architectural help", (399, 4.3),
            textcoords="offset points", xytext=(-16, 46), fontsize=8.5, color=INK, ha="right",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("% of outputs silent (pool mean)", fontsize=10)
bx.set_ylim(-3, 100)
bx.set_title("B · Evolution DOES discover how to fire in the window\n"
             "unaided, via recurrence — but it takes 284–367 rounds to get\n"
             "under 5 %, i.e. most of the budget goes on this, not the task",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="center right")

# ---------------------------------------------------------------- C  what survives
ARMS = [("gate\n[1,20]/[64,80]", V1, "raw{}", "#4E79A7"),
        ("split\n[1,32]/[32,64]", V2, "raw{}", "#59A14F"),
        ("uniform [1,32]\nw_max 30", V3, "w30_{}", C_T),
        ("uniform [1,32]\nw_max 90", V3, "w90_{}", "#E1A03C")]
xs = np.arange(len(ARMS))
for k, (lbl, path, pat, col) in enumerate(ARMS):
    S = [json.load(open(os.path.join(path, f"diag_{pat.format(s)}.json")))["summary"]
         for s in SEEDS]
    mse = [x["mse"] for x in S]
    cx.bar(xs[k], np.mean(mse), 0.56, color=col, zorder=3, alpha=0.9)
    cx.plot([xs[k]] * 3, mse, "o", ms=6, mfc="white", mec=INK, mew=1.3, zorder=5)
    cx.text(xs[k], 2.2, f"mean |r|\n{np.mean([x['mean_abs_r'] for x in S]):.2f}",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold", zorder=6)
cx.axhline(CONST, color=C_T, ls="--", lw=1.7, zorder=6)
cx.text(-0.45, CONST + 1.4, f"constant predictor {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
cx.text(-0.45, 68, "exp009's 800-excitatory STDP\nreservoir: mean |r| ≈ 0.32", fontsize=8.5,
        color=MUTE, ha="left", va="top")
cx.set_xticks(xs)
cx.set_xticklabels([a[0] for a in ARMS], fontsize=9)
cx.set_ylabel("held-out MSE (bar = mean, dots = seeds)", fontsize=10)
cx.set_ylim(0, 72)
cx.set_title("C · Removing the gate costs the signal, not the speed\n"
             "mean |r| collapses 0.52 → 0.19; every seed of both\n"
             "uniform arms lands ABOVE chance, at either w_max",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — one uniform delay range [1,32], no output gate, 3 seeds × 400 rounds",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_uniform.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
