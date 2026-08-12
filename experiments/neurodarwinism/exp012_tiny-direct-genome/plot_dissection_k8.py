"""exp012: what the K=8 diagonal-LS leader actually is.

  A  the decisive test. Freeze wiring + delays, redraw every weight on the {-1,0,+1} grid:
     the net collapses to chance. Redraw the wiring too and it barely moves further. The
     skeleton alone is worth ~10% of the distance from chance -- the weights carry the rest.
  B  what the net needs. The 340 zero-weight edges are literally inert; inhibition is dead
     weight; recurrence is the one thing whose removal is fatal.
  C  no critical neuron. 50 knockouts, none worth more than 3.5 of the 8.4-MSE gap, and a
     third of them HELP. The function is smeared across the excitatory pool.
  D  and it is not one solution but six. Per target dimension the same network ranges from
     r=0.85 to r=0.16 -- target 1 is essentially a constant predictor.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(D, "analysis", "leader_diagls_k8")
R = json.load(open(os.path.join(A, "dissection.json")))

INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"          # chance / the thing to beat
C_W = "#4E79A7"          # weights redrawn
C_S = "#E1A03C"          # wiring redrawn
C_G = "#59A14F"          # the leader itself
BASE, CH = R["baseline_mse"], R["chance"]

fig, AX = plt.subplots(2, 2, figsize=(15.6, 10.0))
fig.subplots_adjust(left=0.055, right=0.985, top=0.845, bottom=0.065, wspace=0.24, hspace=0.42)
ax, bx, cx, dx = AX[0, 0], AX[0, 1], AX[1, 0], AX[1, 1]

# ------------------------------------------------------------------ A weights vs skeleton
WS = R["weights_vs_skeleton"]
wr = np.array(WS["weight_redraw_all"])
sr = np.array(WS["wiring_redraw_all"])
bins = np.linspace(24, 40, 33)
ax.hist(wr, bins, color=C_W, alpha=0.72, label=f"weights redrawn, skeleton kept  (n={wr.size})",
        zorder=3)
ax.hist(sr, bins, color=C_S, alpha=0.62, label=f"wiring redrawn too  (n={sr.size})", zorder=3)
for v, c, lab in ((BASE, C_G, f"the leader  {BASE:.1f}"), (CH, C_T, f"chance  {CH:.1f}")):
    ax.axvline(v, color=c, lw=2.2, zorder=5)
    ax.text(v - 0.16, 0.35, lab, rotation=90, va="bottom", ha="right", fontsize=9, color=c,
            fontweight="bold")
ax.annotate("", xy=(wr.mean(), 8.6), xytext=(sr.mean(), 8.6),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(sr.mean() + 0.4, 8.6, f"the skeleton is worth only {sr.mean() - wr.mean():.1f}",
        ha="left", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.annotate("", xy=(BASE, 10.4), xytext=(wr.mean(), 10.4),
            arrowprops=dict(arrowstyle="<->", color=C_W, lw=1.6))
ax.text((BASE + wr.mean()) / 2, 10.8, f"the WEIGHTS are worth {wr.mean() - BASE:.1f}",
        ha="center", fontsize=9.5, color=C_W, fontweight="bold")
ax.set_xlabel("held-out MSE", fontsize=10)
ax.set_ylabel("draws", fontsize=10)
ax.set_ylim(0, 12.0)
ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.52))
ax.set_title("A · The hypothesis is CONTRADICTED — the weights carry it, not the skeleton\n"
             "freeze wiring and delays, redraw all 761 weights on the grid: every one of 40\n"
             "draws is worse than the leader, and the mean sits AT chance",
             fontsize=10, loc="left", color=INK)

# ------------------------------------------------------------------ B ablations
AB = R["ablations"]
ROWS = [("the leader\n761 syn", BASE, C_G),
        ("drop the 340\nzero-weight edges", AB["prune_weakest_25pct"]["mse"], "#9db8d1"),
        ("drop ALL 87 inhibitory\nsynapses", AB["no_inhibition"]["mse"], "#7FA8C9"),
        ("drop hidden→hidden\n(recurrence)", AB["no_recurrence_exc"]["mse"], C_T)]
ys = np.arange(len(ROWS))[::-1]
bx.barh(ys, [r[1] for r in ROWS], 0.58, color=[r[2] for r in ROWS], zorder=3)
for y, (nm, v, c) in zip(ys, ROWS):
    bx.text(v + 0.4, y, f"{v:.2f}", va="center", fontsize=10, color=INK, fontweight="bold")
bx.axvline(CH, color=C_T, ls="--", lw=1.8, zorder=5)
bx.text(CH + 0.3, 3.45, f"chance {CH:.1f}", fontsize=9, color=C_T, fontweight="bold")
bx.text(37.0, ys[1], "exactly 0.000 change —\nthe grid's zeros are truly inert",
        va="center", fontsize=8.5, color=MUTE, ha="left")
bx.text(37.0, ys[2], "removing inhibition\nIMPROVES the net by 0.41", va="center",
        fontsize=8.5, color=MUTE, ha="left")
bx.set_yticks(ys)
bx.set_yticklabels([r[0] for r in ROWS], fontsize=9)
bx.set_xlim(0, 54)
bx.set_xlabel("held-out MSE", fontsize=10)
bx.set_title("B · Only recurrence is load-bearing\n"
             "half the genome is inert, the inhibitory half-network is dead weight, and\n"
             "cutting hidden→hidden drops the net below chance",
             fontsize=10, loc="left", color=INK)

# ------------------------------------------------------------------ C per-neuron knockout
KO = sorted(R["knockout"], key=lambda k: -k["delta"])
dv = np.array([k["delta"] for k in KO])
is_i = np.array([k["unit"].startswith("I") for k in KO])
cols = np.where(is_i, C_S, C_W)
xs = np.arange(len(KO))
cx.bar(xs, dv, 0.78, color=cols, zorder=3)
cx.axhline(0, color=INK, lw=1.0, zorder=4)
for i in range(2):
    cx.text(xs[i], dv[i] + 0.12, KO[i]["unit"], ha="center", fontsize=8, color=INK,
            fontweight="bold")
cx.text(len(KO) - 1, dv[-1] - 0.18, KO[-1]["unit"], ha="center", va="top", fontsize=8,
        color=INK, fontweight="bold")
cx.text(30, 3.1, f"the whole net is only {CH - BASE:.1f} MSE better than chance;\n"
                 f"the single most critical neuron is worth {dv[0]:.1f},\n"
                 f"and {int((dv < 0).sum())} of 50 neurons HELP when removed",
        fontsize=9, color=INK, ha="left", va="top")
cx.scatter([], [], color=C_W, s=44, label="excitatory (40)")
cx.scatter([], [], color=C_S, s=44, label="inhibitory (10)")
cx.legend(frameon=False, fontsize=8.5, loc="lower left")
cx.set_xlabel("hidden neuron, sorted by how much its removal hurts", fontsize=10)
cx.set_ylabel("Δ held-out MSE when knocked out", fontsize=10)
cx.set_title("C · No critical subnetwork — the function is smeared\n"
             "every one of the 50 hidden neurons individually knocked out. Mean |r| to any\n"
             "target is 0.42 and not one neuron is specialised to a single dimension",
             fontsize=10, loc="left", color=INK)

# ------------------------------------------------------------------ D per-target
PT = R["per_target"]
xs = np.arange(len(PT))
w = 0.26
mse = [p["mse"] for p in PT]
b2 = [p["bias2"] for p in PT]
se = [p["scale_err"] for p in PT]
res = [m - a - b for m, a, b in zip(mse, b2, se)]
dx.bar(xs, res, 0.6, color="#9db8d1", label="residual (unexplainable by an affine)", zorder=3)
dx.bar(xs, se, 0.6, bottom=res, color=C_W, label="scale error", zorder=3)
dx.bar(xs, b2, 0.6, bottom=np.array(res) + np.array(se), color=C_T, label="bias²", zorder=3)
dx.axhline(CH, color=C_T, ls="--", lw=1.6, zorder=5)
dx.text(5.45, CH + 0.8, f"chance {CH:.1f}", fontsize=9, color=C_T, fontweight="bold",
        ha="right")
for x, p in zip(xs, PT):
    dx.text(x, p["mse"] + 0.7, f"r={p['r']:.2f}", ha="center", fontsize=9,
            color=INK if p["r"] > 0.3 else C_T, fontweight="bold")
dx.set_xticks(xs)
dx.set_xticklabels([f"target {i}" for i in range(6)], fontsize=9)
dx.set_ylim(0, 46)
dx.set_ylabel("held-out MSE", fontsize=10)
dx.legend(frameon=False, fontsize=8.5, loc="upper left")
dx.set_title("D · It is six unequal solutions, not one\n"
             "target 5 is genuinely predicted (r=0.85); target 1 is a near-constant output\n"
             "(pred sd 0.88 vs target sd 4.95) and sits well ABOVE chance",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx, cx, dx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — dissecting the K=8 diagonal-LS leader (25.74 held-out vs 34.15 chance): "
             "a distributed, weight-borne, recurrence-dependent solution",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_dissection_k8.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
