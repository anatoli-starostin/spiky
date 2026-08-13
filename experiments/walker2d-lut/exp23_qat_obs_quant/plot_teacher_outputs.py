"""Output distribution of the real fastlut_lse (exp19) LUT teacher, all 6 action dims.

Values are the CONTINUOUS, PRE-CLIP action means the logsumexp pooling produces -- exactly
what feeds Stage 3 of the spiking actor. They are read from `y_action_mean_f64` in
distill_exp19_100k.npz rather than recomputed: that array was produced by the very teacher
this figure is about (verified -- the npz's `weights` are bit-identical to the actor's), so a
forward pass would reproduce it and could only introduce drift.

The quantization grid drawn on each panel is the SHIPPED decode of the spiking readout,
mu = affine[o,0]*T + affine[o,1] for integer ticks T -- i.e. the actual action values the
deployed actor can emit, not a nominal step.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = ("/home/astarostin/projects/spiky/landing/walker2d-viz/server/models/"
       "spiking_lut_actor.npz")
DIS = ("/home/astarostin/projects/spiky/experiments/neurodarwinism/data/"
       "distill_exp19_100k.npz")
OUT = os.path.join(HERE, "analysis", "teacher_output_distribution.png")

INK, MUTE = "#2b2b2b", "#6b6b6b"
C_HIST, C_CLIP, C_GRID = "#4E79A7", "#B4453C", "#9aa5b1"

Q = np.load(NPZ)
D = np.load(DIS)
assert np.array_equal(Q["weights"], D["weights"]), "actor npz and distill npz disagree"
y = D["y_action_mean_f64"].astype(np.float64)          # (100000, 6) pre-clip
aff = Q["affine"].astype(np.float64)                   # (6, 2) tick -> action

fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
fig.subplots_adjust(left=0.05, right=0.99, top=0.845, bottom=0.075,
                    wspace=0.17, hspace=0.34)

stats = []
LO, HI = -4.0, 4.0
for o, ax in enumerate(axes.ravel()):
    v = y[:, o]
    outside = float((np.abs(v) > 1).mean())
    step = abs(aff[o, 0])
    stats.append(dict(dim=o, mean=float(v.mean()), std=float(v.std()),
                      min=float(v.min()), max=float(v.max()),
                      pct_outside=outside * 100, step=step))

    # the action values the spiking readout can actually emit (integer ticks)
    t0 = int(np.floor((HI - aff[o, 1]) / aff[o, 0]))
    t1 = int(np.ceil((LO - aff[o, 1]) / aff[o, 0]))
    levels = aff[o, 0] * np.arange(min(t0, t1), max(t0, t1) + 1) + aff[o, 1]
    levels = levels[(levels >= LO) & (levels <= HI)]

    ax.axvspan(-1, 1, color="#eef2f7", zorder=0)
    n, _, _ = ax.hist(v, bins=220, range=(LO, HI), color=C_HIST, zorder=3)
    ymax = max(n.max(), 1.0)

    for L in levels:
        inside = abs(L) <= 1.0
        ax.plot([L, L], [0, ymax * 1.02], color=C_GRID, lw=1.5 if inside else 0.7,
                alpha=0.85 if inside else 0.35, zorder=2)
    for s in (-1.0, 1.0):
        ax.plot([s, s], [0, ymax * 1.10], color=C_CLIP, lw=2.0, zorder=5)

    n_in = int(((levels >= -1) & (levels <= 1)).sum())
    box = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.5)
    ax.annotate(f"{outside * 100:.1f}% outside [-1, 1]",
                xy=(0.02, 0.955), xycoords="axes fraction", ha="left", va="top",
                fontsize=10.5, fontweight="bold", color=C_CLIP, bbox=box, zorder=6)
    ax.annotate(f"{n_in} emittable levels in the band  ·  step {step:.3f}",
                xy=(0.02, 0.865), xycoords="axes fraction", ha="left", va="top",
                fontsize=9, color=MUTE, bbox=box, zorder=6)

    ax.set_title(f"dim {o}   mean {v.mean():+.3f}   sd {v.std():.3f}   "
                 f"range [{v.min():+.2f}, {v.max():+.2f}]",
                 fontsize=10.5, loc="left", color=INK)
    ax.set_xlim(LO, HI)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_yticks([])
    ax.set_xlabel("continuous LUT output (pre-clip action mean)", fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_axisbelow(True)

fig.text(0.05, 0.905,
         "shaded band = the actuator range [-1, 1]   ·   red lines = the clip   ·   "
         "grey lines = the action values the spiking readout can actually emit "
         "(bold inside the band)",
         fontsize=9.5, color=MUTE, ha="left")
fig.suptitle("What the fastlut_lse (exp19) teacher actually outputs — and how coarsely the "
             f"spiking readout can represent it\n{len(y):,} observations · "
             f"{float((np.abs(y) > 1).mean()) * 100:.1f}% of all action components fall "
             "outside the actuator range before clipping",
             fontsize=13, x=0.005, ha="left", color=INK)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
json.dump(stats, open(os.path.join(HERE, "analysis",
                                   "teacher_output_stats.json"), "w"), indent=1)
print("wrote", OUT)
print(f"{'dim':>3} {'mean':>9} {'std':>8} {'min':>9} {'max':>8} {'% outside':>10} {'step':>8}")
for s in stats:
    print(f"{s['dim']:>3} {s['mean']:>+9.4f} {s['std']:>8.4f} {s['min']:>+9.4f} "
          f"{s['max']:>+8.4f} {s['pct_outside']:>9.2f}% {s['step']:>8.4f}")
print(f"\noverall %% outside [-1,1]: {float((np.abs(y) > 1).mean()) * 100:.2f}%")
