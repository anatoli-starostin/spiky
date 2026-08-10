"""exp011 constrained run: throughput hard-capped at the teacher's 192 weights/forward.

  A  the whole population against the teacher, with the tolerance band that decides what
     counts as "matching its fit"
  B  what the search converged to -- essentially the teacher's own shape
  C  what the budget actually buys: fit is available inside 192, but only by spending params
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(D, "capped_tput192")
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_T = "#B4453C"
TOL = 0.002

F = {s: json.load(open(os.path.join(CAP, f"seed{s}", f"lut_evolve_cap{s}_final.json")))
     for s in SEEDS}
H = {s: json.load(open(os.path.join(CAP, f"seed{s}", f"lut_evolve_cap{s}.json")))
     for s in SEEDS}
t_mse = float(np.mean([F[s]["teacher"]["mse"] for s in SEEDS]))
T_P, T_T = F[0]["teacher"]["params"], F[0]["teacher"]["throughput"]

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15.2, 4.7), layout="constrained")

# ---------------------------------------------------------------- A
win = []
for s in SEEDS:
    p = np.array([x["params"] for x in F[s]["seen"]])
    m = np.array([x["mse"] for x in F[s]["seen"]])
    tp = np.array([x["throughput"] for x in F[s]["seen"]])
    ax.plot(p, m, ".", ms=3.2, color=C_SEED[s], alpha=0.35, zorder=2, label=f"seed {s}")
    k = (m <= F[s]["teacher"]["mse"] + TOL) & (p < T_P) & (tp < T_T)
    win += list(zip(p[k], m[k]))
if win:
    ax.plot([w[0] for w in win], [w[1] for w in win], "o", ms=9, mfc="none",
            mec="#E1575A", mew=2.0, zorder=5,
            label=f"cheaper on BOTH axes ({len(win)})")
ax.axhspan(t_mse, t_mse + TOL, color=C_T, alpha=0.13, zorder=0)
ax.axhline(t_mse, color=C_T, ls="--", lw=1.3, zorder=3)
ax.axvline(T_P, color=C_T, ls=":", lw=1.2, zorder=3)
ax.plot([T_P], [t_mse], "*", ms=20, color=C_T, mec="white", mew=0.8, zorder=6)
ax.annotate("TEACHER", (T_P, t_mse), textcoords="offset points", xytext=(6, 8),
            fontsize=9, color=C_T, fontweight="bold")
ax.text(8600, t_mse + TOL * 0.30, "tolerance band, +0.002", fontsize=8, color=C_T)
ax.text(8600, t_mse - 0.0016, "strictly better fit than the teacher\nAND cheaper: 0 of 2,880",
        fontsize=8.5, color=INK)
# ZOOMED on the only region that can answer the question. The full log-log view buries the
# eight winners under the dense scatter around the teacher.
ax.set_xlim(8500, 17000)
ax.set_ylim(t_mse - 0.005, t_mse + 0.005)
ax.set_xlabel("parameters", fontsize=10)
ax.set_ylabel("held-out MSE", fontsize=10)
ax.set_title("A · Inside the teacher's budget, zoomed on the decision\n"
             "8 of 2,880 are cheaper on BOTH axes at tolerance-matched fit;\n"
             "ZERO also beat it on fit. Every winner is a 3–9 % trim.",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right", markerscale=2)

# ---------------------------------------------------------------- B
for s in SEEDS:
    bx.plot([h["rnd"] for h in H[s]], [np.median(h["nap_vec"]) for h in H[s]],
            "-", lw=2.0, color=C_SEED[s], label=f"seed {s} NAP")
    bx.plot([h["rnd"] for h in H[s]], [np.median(h["tph_vec"]) for h in H[s]],
            "--", lw=1.6, color=C_SEED[s], alpha=0.7)
bx.axhline(6, color=C_T, ls=":", lw=1.4)
bx.axhline(32, color=C_T, ls=":", lw=1.4)
bx.text(31, 6.6, "teacher NAP 6", fontsize=8.5, color=C_T)
bx.text(31, 33, "teacher tph 32 (= the cap)", fontsize=8.5, color=C_T)
bx.set_yscale("log")
bx.set_xlabel("round", fontsize=10)
bx.set_ylabel("pool median (solid NAP, dashed tph)", fontsize=10)
bx.set_title("B · It converges on the teacher's own shape\n"
             "all three seeds settle at NAP 6 with tph at or just under 32 —\n"
             "the winners are NAP 6 × tph 29–31, marginal trims",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="center right")

# ---------------------------------------------------------------- C
for s in SEEDS:
    pts = F[s]["pareto_params"]
    cx.plot([p["params"] for p in pts], [p["mse"] for p in pts], "-o", ms=4,
            color=C_SEED[s], lw=1.6, label=f"seed {s}", zorder=3)
cx.axhline(t_mse, color=C_T, ls="--", lw=1.3)
cx.plot([T_P], [t_mse], "*", ms=18, color=C_T, mec="white", mew=0.8, zorder=5)
cx.annotate(f"best inside budget:\n0.01247 at 98,304 params\n(2× the teacher's fit,\n"
            f"8× its parameters)", (98304, 0.01247), textcoords="offset points",
            xytext=(-135, -8), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.0))
cx.set_xscale("log"); cx.set_yscale("log")
cx.set_xlabel("parameters", fontsize=10)
cx.set_ylabel("held-out MSE", fontsize=10)
cx.set_title("C · Fit IS available inside the budget…\n"
             "…but only by spending parameters. Throughput 192 does not\n"
             "cap quality; it caps quality-per-parameter.",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="lower left")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp011 constrained — throughput ≤ 192 (the teacher's own budget) · "
             "3 seeds × 60 rounds × K=16 · cap held exactly, max observed 192",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp011_capped.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
