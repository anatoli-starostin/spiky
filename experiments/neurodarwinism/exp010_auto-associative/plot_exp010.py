"""exp010 pre-flight figure: why the auto-associative readout is not ready for a 300-round run.

Three panels, in the order the failures appear:
  A  at the chapter's default gain, teacher-clamped STDP kills the network outright
  B  a healthy operating point exists, but it is a narrow one
  C  at that healthy point the readout path is provably correct (the teacher-on control
     reaches the design's ceiling) and STDP still writes no association
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
C_DEAD, C_LIVE, C_SHUF = "#B4453C", "#4E79A7", "#9C9C9C"
CEILING = 0.5399                       # corrected tau of the teacher ticks themselves

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(15.0, 4.5), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.3]),
    layout="constrained")

# ---------------------------------------------------------------- A: the collapse
rep = json.load(open(os.path.join(S, "default_offset64.json")))
xs = [0] + [t["batches"] for t in rep["trace"]] + [rep["config"]["train_batches"]]
ys = ([rep["cold"]["spikes_per_cell"]] + [t["spikes_per_cell"] for t in rep["trace"]]
      + [rep["trained"]["spikes_per_cell"]])
ax.plot(np.maximum(xs, 0.5), ys, "-o", color=C_DEAD, lw=2.0, ms=6, zorder=3)
ax.axhline(1.0, color=MUTE, ls="--", lw=1.2)
ax.text(3.0, 1.9, "1 spike/cell — what a\nfirst-spike code needs", fontsize=8.5, color=MUTE)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=0.1)
ax.set_xticks([0.5, 1, 4, 16, 64, 200])
ax.set_xticklabels(["0", "1", "4", "16", "64", "200"])
ax.set_yticks([0, 0.1, 1, 10])
ax.set_yticklabels(["0", "0.1", "1", "10"])
ax.set_xlabel("teacher-clamped STDP batches", fontsize=10)
ax.set_ylabel("spikes per readout cell per episode", fontsize=10)
ax.set_title("A · At the chapter's default gain the net dies\n"
             "10 spikes/cell → silent by batch 64; 80 % of excitatory\n"
             "weights pinned at 0, the rest at the 45 ceiling",
             fontsize=10, loc="left", color=INK)

# ---------------------------------------------------------------- B: the operating point
rows = (json.load(open(os.path.join(S, "probe_gain_coarse.json")))
        + json.load(open(os.path.join(S, "probe_gain_fine.json"))))
# ONE stdp_lr across the whole curve (1e-3, the only rate that both moves weights and leaves a
# net standing). Mixing rates here would make the trained curve a comparison of two things.
best = {r["fanout_scale"]: r for r in rows if abs(r["stdp_lr"] - 1e-3) < 1e-9}
gains = sorted(best)
cold = [best[g]["cold"]["spikes_per_cell"] for g in gains]
warm = [best[g]["trained"]["spikes_per_cell"] for g in gains]
bx.plot(gains, cold, "-o", color=MUTE, lw=2.0, ms=6, label="before training", zorder=3)
bx.plot(gains, warm, "-o", color=C_LIVE, lw=2.0, ms=6,
        label="after 64 clamped batches (lr 1e-3)", zorder=3)
bx.axhspan(0.7, 1.6, color=C_LIVE, alpha=0.10, zorder=0)
bx.text(11.5, 1.1, "TTFS-codeable\nband", fontsize=8.5, color=C_LIVE, ha="right", va="center")
bx.axvline(3.0, color=INK, ls=":", lw=1.4)
bx.text(3.25, 6.5, "chosen: fan-out ÷ 3", fontsize=8.5, color=INK)
bx.set_xscale("log")
bx.set_yscale("symlog", linthresh=0.1)
bx.set_xticks(gains)
bx.set_xticklabels([f"{g:g}" for g in gains])
bx.set_yticks([0, 0.1, 1, 10])
bx.set_yticklabels(["0", "0.1", "1", "10"])
bx.set_xlabel("fan-out scale (reservoir gain knob, ÷)", fontsize=10)
bx.set_ylabel("spikes per readout cell per episode", fontsize=10)
bx.set_title("B · A healthy point exists, and it is narrow\n"
             "÷1 starts at 10 spikes/cell (no first-spike code) and ÷10\n"
             "is dead cold; only ÷2–÷3 starts near one spike and stays",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=9, loc="lower left")

# ---------------------------------------------------------------- C: ceiling vs reality
lr = json.load(open(os.path.join(S, "healthy_offset32_lr0.001.json")))
bars = [("teacher ticks\ndesign ceiling", CEILING, INK),
        ("clamp left on\nteacher-ON ctrl", lr["teacher_on_control"]["corrected_window_mean"],
         INK),
        ("cold\nno STDP", lr["cold"]["corrected_window_mean"], MUTE),
        ("TRAINED\npaired teacher", lr["trained"]["corrected_window_mean"], C_LIVE),
        ("control\nshuffled teacher", lr["trained_shuffled"]["corrected_window_mean"], C_SHUF)]
pos = np.arange(len(bars))
sds = [0, lr["teacher_on_control"]["corrected_sd"], lr["cold"]["corrected_sd"],
       lr["trained"]["corrected_sd"], lr["trained_shuffled"]["corrected_sd"]]
for i, ((lab, v, col), sd) in enumerate(zip(bars, sds)):
    cx.bar(i, v, width=0.62, color=col, zorder=3)
    if sd:
        cx.errorbar(i, v, yerr=sd, color=INK, lw=1.2, capsize=3, zorder=4)
    cx.text(i, v + 0.022 + sd, f"{v:+.3f}", ha="center", fontsize=9, color=INK, zorder=5)
cx.axhline(0, color=INK, lw=1.0)
cx.set_xticks(pos)
cx.set_xticklabels([b[0] for b in bars], fontsize=8.0)
cx.set_ylim(-0.06, 0.66)
cx.set_ylabel("held-out corrected Kendall tau-b", fontsize=10)
cx.set_title("C · The readout is wired right — and learns nothing\n"
             "the clamp reaches the ceiling, so the path and metric are sound;\n"
             "trained is inside one sd of a randomly-permuted teacher",
             fontsize=10, loc="left", color=INK)
cx.grid(axis="y", color="0.92", lw=0.6)
cx.set_axisbelow(True)

for p in (ax, bx, cx):
    for s in ("top", "right"):
        p.spines[s].set_visible(False)
    p.tick_params(labelsize=9)
ax.grid(color="0.92", lw=0.6); ax.set_axisbelow(True)
bx.grid(color="0.92", lw=0.6); bx.set_axisbelow(True)

fig.suptitle("exp010 pre-flight — auto-associative readout (6 reservoir cells, teacher-clamped "
             "STDP) · K=4, 256 held-out states",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp010_preflight.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
