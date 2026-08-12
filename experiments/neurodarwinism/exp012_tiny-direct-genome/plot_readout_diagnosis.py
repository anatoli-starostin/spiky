"""exp012: why the 40/10 quantized nets never beat chance, and the fix.

  A  the ladder. The same network, read out four ways. The substrate carries the target at
     8.9 MSE; the readout the run uses delivers 48.2, WORSE than chance.
  B  the regime sweep. Across gain 50-800 and inhibition 0.1-2.0, the DIAGONAL readout never
     meaningfully beats chance while the linear one is far below it everywhere -- so the
     dynamical regime is not the problem, the readout FORM is.
  C  the fix in evolution: same substrate, same seed, same everything except the readout.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(D, "analysis")
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
R = json.load(open(os.path.join(A, "readout_z0.json")))
SW = json.load(open(os.path.join(A, "regime_sweep_z0.json")))
CH = R["chance"]

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(16.4, 5.6))
fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.155, wspace=0.28)

# ---------------------------------------------------------------- A the ladder
LAD = [("evolved\ndiagonal", R["evolved_diagonal"], C_T),
       ("BEST\ndiagonal", R["LS_diagonal_on_out_win"], "#E1A03C"),
       ("6×6\nlinear", R["LS_full6x6_on_out_win"], "#59A14F"),
       ("96-tick\nlinear", R["LS_full_on_out_full_96tick"], "#7FA8C9"),
       ("HIDDEN\nlinear", R["LS_on_hidden_first"], "#4E79A7")]
xs = np.arange(len(LAD))
ax.bar(xs, [x[1] for x in LAD], 0.6, color=[x[2] for x in LAD], zorder=3)
for x, (nm, v, c) in zip(xs, LAD):
    ax.text(x, v + 1.0, f"{v:.1f}", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.axhline(CH, color=C_T, ls="--", lw=1.8, zorder=5)
ax.text(4.42, CH + 1.2, f"chance {CH:.1f}", fontsize=9.5, color=C_T, fontweight="bold",
        ha="right")
ax.annotate("", xy=(0, 44), xytext=(1, 44),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(0.5, 45.5, "search\n−13.5", ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.annotate("", xy=(1, 30), xytext=(2, 30),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(1.5, 22.5, "readout FORM\n−13.0", ha="center", fontsize=8.5, color=INK,
        fontweight="bold")
ax.annotate("", xy=(3, 14), xytext=(4, 14),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(3.5, 3.0, "output\nbottleneck −9.3", ha="center", fontsize=8.5, color=INK,
        fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([x[0] for x in LAD], fontsize=9)
ax.set_ylim(0, 56)
ax.set_ylabel("held-out MSE (fitted on training)", fontsize=10)
ax.set_title("A · The substrate is fine; the READOUT is the bottleneck\n"
             "the identical network, read out four ways. Its hidden layer\n"
             "carries the target at 8.9 — the run's own readout gives 48.2",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B regime sweep
rows = SW["rows"]
gains = sorted({r["gain"] for r in rows})
for key, col, lab in (("mse_diag", C_T, "diagonal readout"),
                      ("mse_linear", "#59A14F", "full linear readout"),
                      ("mse_hidden", "#4E79A7", "linear on hidden")):
    m = [np.mean([r[key] for r in rows if r["gain"] == gg]) for gg in gains]
    lo = [np.min([r[key] for r in rows if r["gain"] == gg]) for gg in gains]
    hi = [np.max([r[key] for r in rows if r["gain"] == gg]) for gg in gains]
    bx.plot(gains, m, "-o", ms=5, lw=2.2, color=col, label=lab)
    bx.fill_between(gains, lo, hi, color=col, alpha=0.15)
bx.axhline(SW["chance"], color=C_T, ls="--", lw=1.6)
bx.text(52, SW["chance"] + 1.0, f"chance {SW['chance']:.1f}", fontsize=9, color=C_T,
        fontweight="bold")
bx.set_xscale("log")
bx.set_xticks(gains)
bx.set_xticklabels([f"{int(g)}" for g in gains])
bx.set_xlabel("gain  (band = range over inh_coeff 0.1 … 2.0)", fontsize=10)
bx.set_ylabel("held-out MSE floor", fontsize=10)
bx.set_ylim(0, 40)
bx.set_title("B · No regime rescues the diagonal readout\n"
             "20 (gain, inhibition) settings on ONE fixed wiring: the diagonal floor\n"
             "never clears chance by much; the linear floor is far below it everywhere",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="center left")

# ---------------------------------------------------------------- C the fix, in evolution
def hist(tag):
    p = os.path.join(A, "ab_readout", f"ab_{tag}.json")
    if os.path.exists(p):
        return json.load(open(p))
    # mid-run: parse the log
    # mid-run fallback: the log line is
    #   r <n>  fit <f>  mse <m>  neurons <a> (min <b> max <c>)  syn <s>  held-out best <h> [t]
    # so parse by the 'best' keyword rather than by column index
    out = []
    for line in open(os.path.join(A, f"ab_readout_{tag}.log")):
        if line.startswith("r ") and "held-out best" in line:
            f = line.split()
            out.append(dict(rnd=int(f[1]), best=float(f[f.index("best") + 1])))
    return out


for tag, col, lab in (("evolved", C_T, "evolved diagonal readout"),
                      ("linear", "#59A14F", "full linear readout")):
    h = hist(tag)
    r = [x["rnd"] for x in h]
    v = [x.get("best_heldout", x.get("best")) for x in h]
    cx.plot(r, v, "-o", ms=4, lw=2.4, color=col, label=lab)
cx.axhline(CH, color=C_T, ls="--", lw=1.6)
cx.text(5, CH + 2, f"chance {CH:.1f}", fontsize=9, color=C_T, fontweight="bold")
cx.axhline(43.07, color=MUTE, ls=":", lw=1.5)
cx.text(5, 44.5, "best of the FULL 1700-round run  43.1", fontsize=8.5, color=MUTE)
cx.set_xlabel("round", fontsize=10)
cx.set_ylabel("champion held-out MSE", fontsize=10)
cx.set_title("C · The fix, in evolution — identical seed and substrate\n"
             "only the readout differs. The linear arm beats the whole 1700-\n"
             "round run's best inside 25 rounds, and halves chance.",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=8.5, loc="upper right")

for p in (ax, bx, cx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — why the 40/10 quantized nets never beat chance: the readout form, "
             "not the substrate, the dynamics, or the search",
             fontsize=12, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_readout_diagnosis.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
