"""exp011, corrected substrate: LSE readout + the teacher's real anchors, throughput <= 192.

  A  the whole population against the real teacher -- a 151x gap at best
  B  the isolation: 159 candidates had the teacher's EXACT architecture and differed ONLY in
     their anchor pairs. Best of them: 286x worse.
  C  60 rounds of anchor search move it essentially not at all
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(D, "capped_lse_tput192")
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_T = "#B4453C"

F = {s: json.load(open(os.path.join(CAP, f"seed{s}", f"lut_evolve_lse{s}_final.json")))
     for s in SEEDS}
H = {s: json.load(open(os.path.join(CAP, f"seed{s}", f"lut_evolve_lse{s}.json")))
     for s in SEEDS}
T = F[0]["teacher"]

# constrained_layout collapses here -- three 3-line left-aligned titles plus a suptitle leave
# it no room to solve. Explicit spacing instead.
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.115, wspace=0.30)

# ---------------------------------------------------------------- A
for s in SEEDS:
    p = np.array([x["params"] for x in F[s]["seen"]])
    m = np.array([x["mse"] for x in F[s]["seen"]])
    ax.plot(p, m, ".", ms=3.2, color=C_SEED[s], alpha=0.35, label=f"seed {s}", zorder=2)
ax.axhline(T["mse"], color=C_T, ls="--", lw=1.4, zorder=3)
ax.plot([T["params"]], [T["mse"]], "*", ms=22, color=C_T, mec="white", mew=0.9, zorder=6)
ax.annotate(f"REAL TEACHER\n{T['mse']:.5f} @ {T['params']:,} params",
            (T["params"], T["mse"]), textcoords="offset points", xytext=(0, 26),
            fontsize=9, color=C_T, fontweight="bold", ha="center")
ax.annotate("best in budget\n0.01268 @ 98,305\n(151× the teacher)", (98305, 0.01268),
            textcoords="offset points", xytext=(-30, 34), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.0))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("parameters", fontsize=10)
ax.set_ylabel("held-out MSE", fontsize=10)
ax.set_title("A · Against the REAL teacher, nothing is close\n"
             "0 of 2,880 matched its fit; the best inside the budget is\n"
             "151× worse and costs 8× the parameters",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="center left", markerscale=3)

# ---------------------------------------------------------------- B: the isolation
same = [x["mse"] for s in SEEDS for x in F[s]["seen"]
        if x["genome"]["n_anchor_pairs"] == 6 and x["genome"]["tables_per_head"] == 32]
bx.hist(same, bins=28, color="#4E79A7", alpha=0.85, zorder=3)
bx.axvline(T["mse"], color=C_T, ls="--", lw=1.8, zorder=4)
bx.annotate(f"the teacher\n{T['mse']:.5f}", (T["mse"], len(same) * 0.11),
            textcoords="offset points", xytext=(34, 0), fontsize=9, color=C_T,
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_T, lw=1.2))
bx.axvline(min(same), color=INK, ls=":", lw=1.4, zorder=4)
bx.text(min(same) * 0.055, len(same) * 0.30, f"best of the {len(same)}: {min(same):.5f}\n"
        f"= {min(same)/T['mse']:.0f}× the teacher", fontsize=9, color=INK)
bx.set_xscale("log")
bx.set_xlabel("held-out MSE", fontsize=10)
bx.set_ylabel("candidates", fontsize=10)
bx.set_title("B · Same architecture, only the ANCHORS differ\n"
             f"{len(same)} candidates had NAP 6 × tph 32 — the teacher's exact\n"
             "shape, params and throughput. Best was 286× worse.",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C
for s in SEEDS:
    cx.plot([h["rnd"] for h in H[s]], [h["min_mse"] for h in H[s]], "-", lw=2.0,
            color=C_SEED[s], label=f"seed {s}", zorder=3)
cx.axhline(T["mse"], color=C_T, ls="--", lw=1.4, zorder=4)
cx.text(2, T["mse"] * 1.35, "the real teacher", fontsize=9, color=C_T, fontweight="bold")
cx.set_yscale("log")
cx.set_ylim(T["mse"] * 0.5, 0.1)
cx.set_xlabel("round", fontsize=10)
cx.set_ylabel("best held-out MSE in the pool", fontsize=10)
cx.set_title("C · 60 rounds of anchor search barely move it\n"
             "the gap to the teacher stays over two orders of magnitude\n"
             "for the whole run, on every seed",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp011 corrected — LSE readout + the teacher's real anchors, throughput ≤ 192",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp011_capped_lse.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
