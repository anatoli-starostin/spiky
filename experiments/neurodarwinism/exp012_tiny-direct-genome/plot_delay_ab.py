"""exp012: the fine-delay A/B -- does halving the delay step buy anything?

  A  the two runs' champion trajectories. Identical config, identical seed; only the delay
     grid differs (32 odd ticks vs all 64).
  B  the evolved delay histograms against their own uniform prior, with the null band a
     sample of that size would produce anyway. If the finer grid let selection sculpt delays,
     the fine arm's histogram should sit further outside its band than the odd arm's.
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
AB = json.load(open(os.path.join(D, "analysis", "delay_ab.json")))
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_T = "#B4453C"
COL = {"odd": "#4E79A7", "fine": "#E1A03C"}
LAB = {"odd": "odd ticks 1,3..63  (32 levels, 2-tick hop)",
       "fine": "every tick 1..64  (64 levels, 1-tick hop)"}
RUN = {"odd": "run_diagls_k8", "fine": "run_diagls_k8_finedelay"}
CH = AB["chance"]


def curve(tag):
    """champion held-out per logged round, straight from the runner's own log.

    The round field is right-padded to width 4, so it is 'r  375' but 'r1000' -- splitting on
    whitespace and taking field 1 silently drops every round from 1000 on. Parse the number
    off the front instead.
    """
    out = []
    p = os.path.join(D, RUN[tag], "P0.log")
    for line in open(p):
        m = re.match(r"^r\s*(\d+)\s", line)
        if m and "held-out best" in line:
            f = line.split()
            out.append((int(m.group(1)), float(f[f.index("best") + 1])))
    return np.array(out)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.4, 5.9))
fig.subplots_adjust(left=0.055, right=0.985, top=0.775, bottom=0.115, wspace=0.22)

# ---------------------------------------------------------------- A trajectories
for tag in ("odd", "fine"):
    c = curve(tag)
    ax.plot(c[:, 0], c[:, 1], lw=1.6, color=COL[tag], alpha=0.85, label=LAB[tag])
    m = AB["arms"][tag]["heldout_mse"]
    ax.axhline(m, color=COL[tag], ls=":", lw=1.6)
    ax.text(1700, m - 0.45, f"final EWMA leader {m:.2f}", ha="right", fontsize=9,
            color=COL[tag], fontweight="bold")
ax.axhline(CH, color=C_T, ls="--", lw=1.8)
ax.text(20, CH + 0.5, f"chance {CH:.1f}", fontsize=9.5, color=C_T, fontweight="bold")
d = AB["delta_fine_minus_odd"]
ax.text(860, 21.4, f"Δ (fine − odd) = {d:+.2f} MSE  —  the finer grid is WORSE",
        ha="center", fontsize=11, color=INK, fontweight="bold")
ax.set_xlabel("round", fontsize=10)
ax.set_ylabel("champion held-out MSE", fontsize=10)
ax.set_ylim(20, 38)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("A · Halving the delay step changes nothing\n"
             "same seed, same substrate, same 1700 rounds — the only difference is\n"
             "whether a delay lives on 32 or 64 levels",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B prior-likeness
w = 0.4
xs = np.arange(2)
tv = [AB["arms"][t]["delay_prior_gap"]["tv_from_uniform"] for t in ("odd", "fine")]
nl = [AB["arms"][t]["delay_prior_gap"]["tv_null_mean"] for t in ("odd", "fine")]
p95 = [AB["arms"][t]["delay_prior_gap"]["tv_null_p95"] for t in ("odd", "fine")]
bx.bar(xs - w / 2, nl, w, color="0.82", zorder=3,
       label="a sample from the exact prior (mean)")
bx.bar(xs + w / 2, tv, w, color=[COL["odd"], COL["fine"]], zorder=3,
       label="the EVOLVED delays")
for x, v in zip(xs, p95):
    bx.plot([x - w, x + w], [v, v], color=C_T, lw=1.8, ls="--", zorder=6)
bx.text(1 - w - 0.03, p95[1] + 0.003, "95th pct of the null", fontsize=8.5, color=C_T,
        ha="left", va="bottom", fontweight="bold")
for x, t in zip(xs, ("odd", "fine")):
    g = AB["arms"][t]["delay_prior_gap"]
    bx.text(x + w / 2, tv[x] + 0.004, f"{tv[x]:.3f}\n(pct {g['tv_percentile_in_null']:.0f})",
            ha="center", fontsize=9, color=INK, fontweight="bold")
    bx.text(x - w / 2, nl[x] + 0.004, f"{nl[x]:.3f}", ha="center", fontsize=9, color=MUTE)
bx.set_xticks(xs)
bx.set_xticklabels([f"odd grid\n{AB['arms']['odd']['delay_prior_gap']['n']} unpinned delays, "
                    "32 levels",
                    f"fine grid\n{AB['arms']['fine']['delay_prior_gap']['n']} unpinned delays, "
                    "64 levels"], fontsize=9)
bx.set_ylabel("total-variation distance from the uniform prior", fontsize=10)
bx.set_ylim(0, max(max(tv), max(p95)) * 1.45)
bx.legend(frameon=False, fontsize=9, loc="upper left")
bx.set_title("B · The evolved delays are still the prior — on either grid\n"
             "TV from uniform, against the band a same-size sample of the prior gives\n"
             "anyway. Neither arm's delays are distinguishable from random draws",
             fontsize=10, loc="left", color=INK)

for p in (ax, bx):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.tick_params(labelsize=9)
    p.grid(color="0.93", lw=0.6)
    p.set_axisbelow(True)

fig.suptitle("exp012 — the delay dimension is inert: doubling the delay resolution neither "
             "improves the net nor makes the evolved delays any less random",
             fontsize=12.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp012_delay_ab.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
