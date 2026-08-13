"""exp23 N=22 full run — the detail diagrams: LR schedule, per-dim raw vs effective
output, the raw-readout/delay-span finding, and per-dim R2 of the readout.

Plots only what is already on disk (probe/*.json, probe/*_raw.npy, qat_s*.json). Nothing is
retrained or re-evaluated.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe")
AN = os.path.join(HERE, "analysis")
INK, MUTE = "#2b2b2b", "#6b6b6b"
CS = ["#4E79A7", "#59A14F", "#B4453C"]
C_PAR, C_RAW, C_EFF, C_GRID = "#8b8b8b", "#b0b7c3", "#4E79A7", "#7d8794"
LEVELS = np.linspace(-1, 1, 22)

H = {s: json.load(open(os.path.join(HERE, f"qat_s{s}.json")))["history"] for s in (0, 1, 2)}
par = json.load(open(os.path.join(PROBE, "parent.json")))
qat = {s: json.load(open(os.path.join(PROBE, f"qat_s{s}.json"))) for s in (0, 1, 2)}
raw_p = np.load(os.path.join(PROBE, "parent_raw.npy")).astype(np.float64)
raw_q = np.load(os.path.join(PROBE, "qat_s0_raw.npy")).astype(np.float64)


def effective(x):
    """what the actor actually emits: clip to [-1,1], then snap to the 22-level grid"""
    c = np.clip(x, -1.0, 1.0)
    step = 2.0 / 21.0
    return np.clip(np.round((c + 1.0) / step) * step - 1.0, -1.0, 1.0)


def r2(y, yhat):
    ss = ((y - yhat) ** 2).sum()
    return 1.0 - ss / ((y - y.mean()) ** 2).sum()


# =============================================================== FIGURE: per-dim raw vs eff
fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.4))
fig.subplots_adjust(left=0.045, right=0.99, top=0.83, bottom=0.08, wspace=0.14, hspace=0.34)
LO, HI = -4.3, 4.3
r2_raw, r2_clip = [], []
for o, ax in enumerate(axes.ravel()):
    v = raw_q[:, o]
    eff = effective(v)
    r2_raw.append(r2(v, eff))
    r2_clip.append(r2(np.clip(v, -1, 1), eff))
    ax.axvspan(-1, 1, color="#eef2f7", zorder=0)
    n_, _, _ = ax.hist(v, bins=190, range=(LO, HI), color=C_RAW, zorder=3,
                       label="RAW pre-clip")
    ym = n_.max()
    # the effective output is a set of 22 spikes; draw it as stems on the same axes
    vals, cnts = np.unique(eff, return_counts=True)
    ax.bar(vals, cnts / cnts.max() * ym * 0.92, width=0.05, color=C_EFF, zorder=7,
           label="EFFECTIVE (clip+quantize)")
    for L in LEVELS:
        ax.plot([L, L], [0, ym * 0.30], color=C_GRID, lw=0.6, alpha=0.55, zorder=2)
    for sgn in (-1.0, 1.0):
        ax.plot([sgn, sgn], [0, ym * 1.06], color="#B4453C", lw=1.1, ls="--",
                alpha=0.75, zorder=1)
    d = qat[0]["per_dim"][o]
    ax.annotate(f"{d['pct_outside']:.1f}% out-of-band\nR2 vs raw {r2_raw[o]:.3f} · "
                f"vs clipped {r2_clip[o]:.3f}",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
                color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.93))
    ax.set_title(f"dim {o}   raw mean {d['mean']:+.2f}  sd {d['std']:.2f}",
                 fontsize=9.5, loc="left", color=INK)
    ax.set_xlim(LO, HI)
    ax.set_ylim(0, ym * 1.16)
    ax.set_yticks([])
    ax.set_xlabel("action value", fontsize=8.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)
    if o == 0:
        ax.legend(frameon=False, fontsize=8, loc="center right")

fig.suptitle("exp23 N=22 · per-dim output after fine-tuning — RAW pre-clip (grey) against "
             "what the actor can EMIT (blue, 22 levels)\n"
             "the emitted mass piles onto the two rails because half the raw output is "
             "outside the band the readout can represent",
             fontsize=12, x=0.004, ha="left", color=INK)
P1 = os.path.join(AN, "exp23_n22_perdim_output.png")
fig.savefig(P1, dpi=135)
print("wrote", P1)

# =============================================================== FIGURE: curves + finding
fig2, ax2 = plt.subplots(1, 3, figsize=(17.0, 5.0))
fig2.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.13, wspace=0.26)
a, b, c = ax2

for s in (0, 1, 2):
    u = [r["update"] for r in H[s]]
    a.plot(u, [r["lr"] for r in H[s]], lw=1.9, color=CS[s], label=f"seed {s}")
a.set_yscale("log")
a.set_xlabel("update", fontsize=9.5)
a.set_ylabel("learning rate (log)", fontsize=9.5)
a.legend(frameon=False, fontsize=9)
a.annotate("full cosine 3e-4 -> 3e-5\n(--init-lr-mode cosine)", xy=(0.35, 0.75),
           xycoords="axes fraction", fontsize=9, color=INK, fontweight="bold")
a.set_title("A · LR schedule as actually run", fontsize=10, loc="left", color=INK)

labs = ["parent", "s0", "s1", "s2"]
oob = [par["pct_outside_all"]] + [qat[s]["pct_outside_all"] for s in (0, 1, 2)]
xs = np.arange(4)
b.bar(xs, oob, width=0.6, color=[C_PAR] + CS, zorder=3)
b.axhline(par["pct_outside_all"], color=C_PAR, ls="--", lw=1.4, zorder=4)
for x, v in zip(xs, oob):
    b.annotate(f"{v:.1f}%", xy=(x, v), xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=10, fontweight="bold", color=INK)
b.set_xticks(xs); b.set_xticklabels(labs, fontsize=9)
b.set_ylim(0, 68)
b.set_ylabel("% of raw output outside [-1,1]", fontsize=9.5)
b.set_title("B · It did NOT come in-band\n51.6% before, 51.1% after — flat",
            fontsize=10, loc="left", color=INK)

dmx = [par["dmax"]] + [qat[s]["dmax"] for s in (0, 1, 2)]
tks = [par["n_ticks_est"]] + [qat[s]["n_ticks_est"] for s in (0, 1, 2)]
w = 0.38
c.bar(xs - w / 2, dmx, w, color=[C_PAR] + CS, zorder=3, label="dmax (ticks)")
c.bar(xs + w / 2, np.array(tks) / 4.0, w, color=[C_PAR] + CS, alpha=0.45, zorder=3,
      label="episode ticks / 4")
for x, v in zip(xs, dmx):
    c.annotate(f"{v}", xy=(x - w / 2, v), xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=9, fontweight="bold", color=INK)
for x, v in zip(xs, tks):
    c.annotate(f"{v}", xy=(x + w / 2, v / 4.0), xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=9, color=MUTE)
c.set_xticks(xs); c.set_xticklabels(labs, fontsize=9)
c.set_ylabel("ticks", fontsize=9.5)
c.legend(frameon=False, fontsize=8.5, loc="lower right")
c.set_title("C · ...and the spiking cost went UP\ndmax 84 -> 96, episode 302 -> 314",
            fontsize=10, loc="left", color=INK)

for p in (a, b, c):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)
    p.tick_params(labelsize=8.5)

fig2.suptitle("exp23 N=22 · LR schedule, and the raw-readout finding that the return "
              "number cannot show",
              fontsize=12, x=0.004, ha="left", color=INK)
P2 = os.path.join(AN, "exp23_n22_lr_and_finding.png")
fig2.savefig(P2, dpi=135)
print("wrote", P2)

json.dump(dict(r2_vs_raw=r2_raw, r2_vs_clipped=r2_clip),
          open(os.path.join(AN, "readout_r2.json"), "w"), indent=1)
print("per-dim R2 vs RAW    :", [round(v, 4) for v in r2_raw])
print("per-dim R2 vs CLIPPED:", [round(v, 4) for v in r2_clip])
