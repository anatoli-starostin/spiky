"""exp010 follow-up figure: scoped + homeostatic STDP, and where the signal actually goes.

  A  20 scoped/homeostatic configurations, paired teacher minus shuffled teacher. Every one
     straddles zero: the pairing is not being written.
  B  the homeostasis knob is demonstrably live — weight_scaling_cf moves the readout firing
     rate monotonically — so A is a null result about learning, not about a flag left off.
  C  the reason, and it is not STDP. An UNTRAINED frozen reservoir already carries +0.317 in
     the very six first-spike times the readout reads; the rank-order metric recovers +0.060.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(D, "sanity")
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_FROZEN, C_UNFROZEN, C_HI = "#4E79A7", "#9C9C9C", "#59A14F"
CEILING = 0.5399

rows = json.load(open(os.path.join(S, "scoped_sweep.json")))
probe = json.load(open(os.path.join(S, "linear_probe_fanout2.0.json")))

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(15.4, 4.9), gridspec_kw=dict(width_ratios=[1.35, 0.95, 1.3]),
    layout="constrained")

# ---------------------------------------------------------------- A: every gap is noise
order = sorted(range(len(rows)), key=lambda i: rows[i]["gap"])
ys = np.arange(len(order))
for y, i in zip(ys, order):
    r = rows[i]
    col = C_FROZEN if r["freeze"] else C_UNFROZEN
    ax.errorbar(r["gap"], y, xerr=r["sd"], fmt="o", ms=5.5, color=col,
                ecolor=col, elinewidth=1.4, capsize=2.5, zorder=3)
ax.axvline(0, color=INK, lw=1.1, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels([f"÷{rows[i]['fanout_scale']:g}  lr {rows[i]['stdp_lr']:g}  "
                    f"wsc {rows[i]['wsc']:g}{'' if rows[i]['freeze'] else '  (unfrozen)'}"
                    for i in order], fontsize=7.2)
ax.set_xlabel("held-out tau, paired teacher − shuffled teacher", fontsize=10)
ax.set_title("A · 20 configurations, no association written\n"
             "each bar is ±1 member-to-member sd; every one crosses zero,\n"
             "and the sign of the gap is a coin flip (10 up, 9 down, 1 flat)",
             fontsize=10, loc="left", color=INK)
ax.plot([], [], "o", color=C_FROZEN, label="frozen reservoir (scoped)")
ax.plot([], [], "o", color=C_UNFROZEN, label="unfrozen (pre-flight regime)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")

# ---------------------------------------------------------------- B: the knob is live
for fs, mark in ((2.0, "o"), (3.0, "s")):
    for lr, ls in ((0.001, "-"), (0.01, "--")):
        sel = [r for r in rows if r["freeze"] and r["fanout_scale"] == fs
               and r["stdp_lr"] == lr]
        sel.sort(key=lambda r: r["wsc"])
        bx.plot([max(r["wsc"], 3e-4) for r in sel],
                [r["paired"]["spikes_per_cell"] for r in sel],
                ls + mark, color=C_HI if fs == 2.0 else C_FROZEN, lw=1.8, ms=5.5,
                label=f"÷{fs:g}, lr {lr:g}", zorder=3)
bx.axhspan(0.7, 1.6, color=MUTE, alpha=0.12, zorder=0)
bx.text(3.4e-4, 1.1, "TTFS band", fontsize=8.5, color=MUTE, va="center")
bx.set_xscale("log")
bx.set_xticks([3e-4, 1e-2, 3e-2, 1e-1])
bx.set_xticklabels(["0", "0.01", "0.03", "0.1"])
bx.set_yscale("symlog", linthresh=1.0)
bx.set_yticks([0, 1, 2, 5, 10])
bx.set_yticklabels(["0", "1", "2", "5", "10"])
bx.set_xlabel("weight_scaling_cf (homeostatic drift)", fontsize=10)
bx.set_ylabel("spikes per readout cell per episode", fontsize=10)
bx.set_title("B · Homeostasis works — mechanically\n"
             "the drift term does rescue weights from 0 and lifts\n"
             "firing monotonically. It just buys no tau.",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=8.5, loc="upper left")

# ---------------------------------------------------------------- C: where the signal goes
res = probe["results"]
bars = [("TTFS rank order\nTHE METRIC", probe["ttfs_rank_order"]["corrected"], "#B4453C"),
        ("linear decode\nSAME 6 ticks",
         res["READOUT CELLS only (6), FIRST SPIKE only"]["heldout"], C_HI),
        ("linear decode\n238 afferents",
         res["AFFERENTS of readout (238 cells)"]["heldout"], C_FROZEN),
        ("linear decode\nall 800 cells", res["ALL EXC (800 cells)"]["heldout"], C_FROZEN)]
for i, (lab, v, col) in enumerate(bars):
    cx.bar(i, v, width=0.62, color=col, zorder=3)
    cx.text(i, v + 0.014, f"{v:+.3f}", ha="center", fontsize=9.5, color=INK, zorder=5)
cx.axhline(CEILING, color=INK, ls="--", lw=1.3, zorder=4)
cx.text(3.42, CEILING + 0.012, f"design ceiling {CEILING:+.3f}", fontsize=8.5,
        color=INK, ha="right")
cx.axhline(0, color=INK, lw=1.0)
cx.set_xticks(range(len(bars)))
cx.set_xticklabels([b[0] for b in bars], fontsize=8.0)
cx.set_ylim(0, 0.62)
cx.set_ylabel("held-out corrected Kendall tau-b", fontsize=10)
cx.set_title("C · The signal is already there, UNTRAINED — the\n"
             "readout code throws it away. Same six numbers,\n"
             "5× more tau when decoded instead of ranked.",
             fontsize=10, loc="left", color=INK)
cx.grid(axis="y", color="0.92", lw=0.6)
cx.set_axisbelow(True)

for p in (ax, bx, cx):
    for s in ("top", "right"):
        p.spines[s].set_visible(False)
    p.tick_params(labelsize=9)
ax.grid(axis="x", color="0.94", lw=0.6); ax.set_axisbelow(True)
bx.grid(color="0.94", lw=0.6); bx.set_axisbelow(True)

fig.suptitle("exp010 follow-up — scoped, homeostatic teacher-clamped STDP · "
             "400 batches, K=4, 256–512 held-out states, frozen reservoir at fan-out ÷2",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp010_scoped.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
