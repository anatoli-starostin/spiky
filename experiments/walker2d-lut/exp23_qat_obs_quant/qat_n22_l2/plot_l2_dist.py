"""exp23 L2 — the raw output actually being pulled into [-1,1], per dim, plus what the
emitted 22-level output looks like once it is.

Plots existing probe artefacts only; nothing is retrained or re-evaluated.
"""
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

B = "/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant"
HERE = os.path.join(B, "qat_n22_l2")
AN = os.path.join(HERE, "analysis")
INK, MUTE = "#2b2b2b", "#6b6b6b"
C_P, C_W1, C_W3, C_EFF, C_GRID = "#9aa5b1", "#E8A33D", "#2f5d8a", "#4E79A7", "#7d8794"
LEVELS = np.linspace(-1, 1, 22)
TAU, TAU_M = 0.09036567807197571, 10.0
SCALE = (1.0 / np.log((1.0 + 0.5 / TAU_M) ** 2)) / TAU

raw_p = np.load(f"{B}/qat_n22_full/probe/parent_raw.npy").astype(np.float64)
raw_0 = np.load(f"{B}/qat_n22_full/probe/qat_s0_raw.npy").astype(np.float64)
raw_1 = np.load(f"{HERE}/probe/l2w0p1_s0_raw.npy").astype(np.float64)
raw_3 = np.load(f"{HERE}/probe/l2w0p3_s0_raw.npy").astype(np.float64)
j_p = json.load(open(f"{B}/qat_n22_full/probe/parent.json"))
j_1 = json.load(open(f"{HERE}/probe/l2w0p1_s0.json"))
j_3 = json.load(open(f"{HERE}/probe/l2w0p3_s0.json"))

# ============================================================ FIG 1: raw, per dim, overlaid
fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.4))
fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.08, wspace=0.12, hspace=0.34)
LO, HI = -4.3, 4.3
for o, ax in enumerate(axes.ravel()):
    ax.axvspan(-1, 1, color="#eef2f7", zorder=0)
    for arr, col, lab in ((raw_p, C_P, "parent"), (raw_1, C_W1, "L2 w=0.1"),
                          (raw_3, C_W3, "L2 w=0.3")):
        ax.hist(arr[:, o], bins=190, range=(LO, HI), color=col, alpha=0.62,
                zorder=3, label=lab if o == 0 else None)
    ym = ax.get_ylim()[1]
    for s in (-1.0, 1.0):
        ax.plot([s, s], [0, ym], color="#B4453C", lw=1.5, ls="--", alpha=0.8, zorder=6)
    txt = (f"out-of-band  {j_p['per_dim'][o]['pct_outside']:.1f}%  ->  "
           f"{j_1['per_dim'][o]['pct_outside']:.1f}%  ->  "
           f"{j_3['per_dim'][o]['pct_outside']:.1f}%")
    ax.annotate(txt, xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9,
                color=INK, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.93))
    ax.set_title(f"dim {o}   raw mean {j_p['per_dim'][o]['mean']:+.2f} -> "
                 f"{j_3['per_dim'][o]['mean']:+.2f}", fontsize=9.5, loc="left", color=INK)
    ax.set_xlim(LO, HI); ax.set_yticks([]); ax.set_xlabel("raw action value", fontsize=8.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)
    if o == 0:
        ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("exp23 L2 · the raw pre-clip readout being pulled INTO the actuator band\n"
             f"overall out-of-band {j_p['pct_outside_all']:.1f}% (parent) -> "
             f"{j_1['pct_outside_all']:.1f}% (w=0.1) -> {j_3['pct_outside_all']:.1f}% "
             "(w=0.3) — the penalty supplies the gradient the clamp cannot",
             fontsize=12, x=0.004, ha="left", color=INK)
P1 = os.path.join(AN, "exp23_l2_raw_perdim.png")
os.makedirs(AN, exist_ok=True)
fig.savefig(P1, dpi=135)
print("wrote", P1)


# ============================================================ FIG 2: emitted vs raw
def eff(x):
    """what the actor emits: clip to [-1,1], then snap to the 22-level grid"""
    c = np.clip(x, -1.0, 1.0)
    st = 2.0 / 21.0
    return np.clip(np.round((c + 1.0) / st) * st - 1.0, -1.0, 1.0)


fig2, ax2 = plt.subplots(1, 3, figsize=(17.0, 5.2))
fig2.subplots_adjust(left=0.045, right=0.99, top=0.74, bottom=0.14, wspace=0.16)
for k, (arr, lab, jj) in enumerate(((raw_0, "no penalty (w=0)", None),
                                    (raw_1, "L2 w=0.1", j_1),
                                    (raw_3, "L2 w=0.3", j_3))):
    a = ax2[k]
    v = arr[:, 3]                      # dim 3 — the worst dim in the no-penalty run
    a.axvspan(-1, 1, color="#eef2f7", zorder=0)
    n_, _, _ = a.hist(v, bins=190, range=(LO, HI), color=C_P, zorder=3, label="RAW pre-clip")
    ym = n_.max()
    vals, cnts = np.unique(eff(v), return_counts=True)
    a.bar(vals, cnts / cnts.max() * ym * 0.92, width=0.05, color=C_EFF, zorder=7,
          label="EMITTED (22 levels)")
    for L in LEVELS:
        a.plot([L, L], [0, ym * 0.28], color=C_GRID, lw=0.6, alpha=0.5, zorder=2)
    for s in (-1.0, 1.0):
        a.plot([s, s], [0, ym * 1.06], color="#B4453C", lw=1.1, ls="--", alpha=0.75, zorder=1)
    rail = float(((np.abs(v) > 1)).mean()) * 100
    a.annotate(f"dim 3 out-of-band {rail:.1f}%\n"
               f"{'almost all mass on the +1 rail' if rail > 50 else 'interior levels now carry the mass'}",
               xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9, color=INK,
               fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.93))
    a.set_title(lab, fontsize=10.5, loc="left", color=INK)
    a.set_xlim(LO, HI); a.set_ylim(0, ym * 1.16); a.set_yticks([])
    a.set_xlabel("action value (dim 3)", fontsize=9)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8.5)
    if k == 0:
        a.legend(frameon=False, fontsize=9, loc="center right")
fig2.suptitle("exp23 L2 · dim 3 — the emitted 22-level output stops piling on the rail\n"
              "with no penalty essentially the whole output is the single spike at +1; "
              "with the penalty the interior levels carry real mass",
              fontsize=12, x=0.004, ha="left", color=INK)
P2 = os.path.join(AN, "exp23_l2_emitted_dim3.png")
fig2.savefig(P2, dpi=135)
print("wrote", P2)
