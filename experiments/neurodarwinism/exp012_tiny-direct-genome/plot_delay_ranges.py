"""exp012 — four delay layouts, and what each one costs.

  A  two-hop reachability of the readout window as a function of input tick. [1,64] flattens
     the 528x bias that [1,32] imposed, down to 1.18x.
  B  silence at round 0 vs round 400: evolution learns to fire in-window under every layout,
     and a wider range makes it cheaper.
  C  what the net ends up carrying. The signal comes back with [1,64] — 0.19 -> 0.42 — but
     the split ranges still hold the best MSE.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_SEED = {0: "#4E79A7", 1: "#59A14F", 2: "#B07AA1"}
C_T = "#B4453C"
CONST = 34.152                      # every diagnostic runs on seed 0's held-out split

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.2, 5.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.72, bottom=0.115, wspace=0.31)

# ---------------------------------------------------------------- A  reachability
def reach(lo, hi, lo2=None, hi2=None):
    d1 = np.arange(lo, hi + 1)
    d2 = np.arange(lo if lo2 is None else lo2, (hi if hi2 is None else hi2) + 1)
    A, B = np.meshgrid(d1, d2)
    return np.array([(((t + A + B) >= 64) & ((t + A + B) < 96)).mean() for t in range(32)])

for lbl, args, col, ls in (("uniform [1,64] — this run", (1, 64), "#59A14F", "-"),
                           ("split [1,32] / [32,64]", (1, 32, 32, 64), "#4E79A7", "--"),
                           ("uniform [1,32]", (1, 32), C_T, "-")):
    r = reach(*args)
    ax.plot(range(32), 100 * r, ls, lw=2.4 if ls == "-" else 2.0, color=col, label=lbl)
ax.set_xlabel("input spike tick  (early = LARGE observation)", fontsize=10)
ax.set_ylabel("% of (d₁,d₂) pairs landing in [64,96)", fontsize=10)
ax.set_ylim(-3, 82)
ax.annotate("1 pair in 1024", (0, 0.1), textcoords="offset points", xytext=(12, 26),
            fontsize=8.5, color=C_T, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_T, lw=1.1))
ax.annotate("flat to 1.18× across\nthe whole input phase", (16, 43.7),
            textcoords="offset points", xytext=(-4, -44), fontsize=8.5, color="#3d7a35",
            ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#3d7a35", lw=1.1))
ax.set_title("A · [1,64] removes the bias [1,32] introduced\n"
             "under [1,32] the earliest input reached the window 528× less\n"
             "often than the latest; under [1,64] the ratio is 1.18×",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# ---------------------------------------------------------------- B  silence
BARS = [("uniform [1,32]\nw_max 30", "sanity_v3_uniform1-32", "w30_{}", C_T),
        ("uniform [1,32]\nw_max 90", "sanity_v3_uniform1-32", "w90_{}", "#E1A03C"),
        ("uniform [1,64]\nw_max 30", "sanity", "w30_{}", "#7FA8C9"),
        ("uniform [1,64]\nw_max 60", "sanity", "w60_{}", "#59A14F")]
xs = np.arange(len(BARS))
for k, (lbl, path, pat, col) in enumerate(BARS):
    H = [json.load(open(os.path.join(D, path, f"{pat.format(s)}.json"))) for s in SEEDS]
    r0 = 100 * np.mean([h[0]["silent"] for h in H])
    rE = 100 * np.mean([h[-1]["silent"] for h in H])
    bx.bar(xs[k] - 0.19, r0, 0.36, color=col, alpha=0.35, zorder=3, edgecolor=col, lw=1.6)
    bx.bar(xs[k] + 0.19, rE, 0.36, color=col, zorder=3)
    bx.text(xs[k] - 0.19, r0 + 2.0, f"{r0:.0f}", ha="center", fontsize=9, color=INK)
    bx.text(xs[k] + 0.19, rE + 2.0, f"{rE:.1f}", ha="center", fontsize=9, color=INK,
            fontweight="bold")
bx.set_xticks(xs)
bx.set_xticklabels([b[0] for b in BARS], fontsize=8.5)
bx.set_ylabel("% of outputs silent (pool mean)", fontsize=10)
bx.set_ylim(0, 104)
bx.text(0.02, 0.955, "pale = round 0     solid = round 400", transform=bx.transAxes,
        fontsize=9, color=MUTE)
bx.set_title("B · Evolution learns to fire in the window under every layout\n"
             "with no architectural help at all — and a wider delay range\n"
             "makes the starting point far less hostile",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- C  what survives
ARMS = [("gate\n[1,20]/[64,80]", "sanity_v1_gate64-80", "raw{}", "#4E79A7"),
        ("split\n[1,32]/[32,64]", "sanity_v2_split1-32_32-64", "raw{}", "#59A14F"),
        ("uniform [1,32]\nw_max 90", "sanity_v3_uniform1-32", "w90_{}", C_T),
        ("uniform [1,64]\nw_max 60", "sanity", "w60_{}", "#E1A03C")]
xs = np.arange(len(ARMS))
for k, (lbl, path, pat, col) in enumerate(ARMS):
    S = [json.load(open(os.path.join(D, path, f"diag_{pat.format(s)}.json")))["summary"]
         for s in SEEDS]
    mse = [x["mse"] for x in S]
    cx.bar(xs[k], np.mean(mse), 0.56, color=col, zorder=3, alpha=0.9)
    cx.plot([xs[k]] * 3, mse, "o", ms=6, mfc="white", mec=INK, mew=1.3, zorder=5)
    cx.text(xs[k], 1.8, f"mean |r|\n{np.mean([x['mean_abs_r'] for x in S]):.2f}",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold", zorder=6)
cx.axhline(CONST, color=C_T, ls="--", lw=1.7, zorder=6)
cx.text(-0.45, CONST + 1.2, f"constant predictor {CONST:.1f}", fontsize=9, color=C_T,
        fontweight="bold", ha="left")
cx.text(-0.45, 63, "exp009's 800-excitatory STDP reservoir:\nMSE 37.52 of its own 39.19 "
        "chance, mean |r| ≈ 0.32", fontsize=8.5, color=MUTE, ha="left", va="top")
cx.set_xticks(xs)
cx.set_xticklabels([a[0] for a in ARMS], fontsize=9)
cx.set_ylabel("held-out MSE (bar = mean, dots = seeds)", fontsize=10)
cx.set_ylim(0, 66)
cx.set_title("C · [1,64] brings the signal back — 0.19 → 0.42 mean |r|\n"
             "but the split ranges still hold the best single seed (27.4).\n"
             "Handing the readout a dedicated delay band is worth real MSE.",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — four delay layouts, 33 neurons, no plasticity, 3 seeds × 400 rounds",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_delay_ranges.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
