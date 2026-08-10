"""exp011 sanity figure: the LUT substrate trains, and the evolutionary loop grips it.

  A  fit vs size — the curve the whole experiment is about, plus the Pareto front the
     6-round smoke test already recovered
  B  at MATCHED parameter count, which capacity axis is worth spending on
  C  the loop works: 6 rounds shrink the pool 11x at flat fitness
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
C_SWEEP, C_PARETO, C_HI = "#4E79A7", "#59A14F", "#B4453C"

sweep = json.load(open(os.path.join(S, "fit_vs_size.json")))
iso12 = json.load(open(os.path.join(S, "iso_params_12k.json")))
iso49 = json.load(open(os.path.join(S, "iso_params_49k.json")))
ev = json.load(open(os.path.join(S, "evolve_smoke.json")))
CONST = sweep["baselines"]["constant_predictor_mse"]
TEACHER_P = 12288                      # the distillation teacher's own shape: NAP 6 x tph 32

fig, (ax, bx, cx) = plt.subplots(
    1, 3, figsize=(15.2, 4.7), gridspec_kw=dict(width_ratios=[1.25, 1.05, 1.05]),
    layout="constrained")

# ---------------------------------------------------------------- A: fit vs size
p = np.array([r["params"] for r in sweep["rows"]])
m = np.array([r["heldout_mse"] for r in sweep["rows"]])
o = np.argsort(p)
ax.plot(p[o], m[o], "-o", color=C_SWEEP, lw=2.0, ms=6.5,
        label="hand-picked sweep (3000 steps)", zorder=3)
pf = ev["pareto"]
ax.plot([r["params"] for r in pf], [r["mse"] for r in pf], "--s", color=C_PARETO,
        lw=1.8, ms=5.5, label="Pareto front after 6 evolved rounds", zorder=4)
ax.axhline(CONST, color=C_HI, ls="--", lw=1.3, zorder=2)
ax.text(230, CONST * 0.72, f"constant predictor  {CONST:.3f}", fontsize=8.5, color=C_HI)
ax.axvline(TEACHER_P, color=INK, ls=":", lw=1.3, zorder=2)
ax.text(TEACHER_P * 1.15, 0.5, "the teacher's\nown size\n(NAP 6 × 32)", fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("parameters", fontsize=10)
ax.set_ylabel("held-out MSE", fontsize=10)
ax.set_title("A · The substrate trains, and the curve is clean\n"
             "192 params already beat the constant predictor 4×;\n"
             "the knee is at ~10k, and returns are flat past ~50k",
             fontsize=10, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

# ---------------------------------------------------------------- B: which axis to spend on
for js, budget, col, mark in ((iso12, 12288, C_SWEEP, "o"), (iso49, 49152, C_PARETO, "s")):
    rows = [r for r in js["rows"] if r["params"] == budget]     # drop the non-iso NAP12 row
    rows.sort(key=lambda r: r["genome"]["n_anchor_pairs"])
    bx.plot([r["genome"]["n_anchor_pairs"] for r in rows],
            [r["heldout_mse"] for r in rows], "-" + mark, color=col, lw=2.0, ms=6.5,
            label=f"{budget:,} params", zorder=3)
    for r in rows:
        bx.annotate(f"×{r['genome']['tables_per_head']}",
                    (r["genome"]["n_anchor_pairs"], r["heldout_mse"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=7.5,
                    color=col, ha="center")
bx.set_yscale("log")
bx.set_xticks([4, 6, 8, 10, 12])
bx.set_xlabel("n_anchor_pairs (rows per table = 2^NAP)", fontsize=10)
bx.set_ylabel("held-out MSE", fontsize=10)
bx.set_title("B · Spend on TABLES, not on depth\n"
             "at fixed size, NAP 4–6 with many tables beats NAP 10\n"
             "with few by ~5×. Labels are tables_per_head.",
             fontsize=10, loc="left", color=INK)
bx.legend(frameon=False, fontsize=9, loc="upper left")

# ---------------------------------------------------------------- C: the loop grips
h = ev["history"]
rnds = [r["rnd"] for r in h]
pv = [np.array(r["params_vec"]) for r in h]
cx.fill_between(rnds, [v.min() for v in pv], [v.max() for v in pv],
                color=C_SWEEP, alpha=0.16, zorder=2, label="pool min–max")
cx.plot(rnds, [np.median(v) for v in pv], "-o", color=C_SWEEP, lw=2.2, ms=6,
        label="pool median", zorder=4)
cx.axhline(TEACHER_P, color=INK, ls=":", lw=1.3, zorder=3)
cx.text(0.05, TEACHER_P * 1.25, "teacher's size", fontsize=8.5, color=INK)
cx.set_yscale("log")
cx.set_xlabel("round", fontsize=10)
cx.set_ylabel("parameters per candidate", fontsize=10)
f0, f1 = h[0]["best"], h[-1]["best"]
cx.set_title("C · Selection grips: 11× smaller in 6 rounds\n"
             f"pool median 224k → 20k while best fitness holds\n"
             f"({f0:+.5f} → {f1:+.5f}) — the size penalty is doing its job",
             fontsize=10, loc="left", color=INK)
cx.legend(frameon=False, fontsize=9, loc="upper right")

for q in (ax, bx, cx):
    for s in ("top", "right"):
        q.spines[s].set_visible(False)
    q.tick_params(labelsize=9)
    q.grid(color="0.93", lw=0.6)
    q.set_axisbelow(True)

fig.suptitle("exp011 pre-flight — evolving FastMultiHeadLut hyperparameters with backprop "
             "inside the chapter's steady-state loop",
             fontsize=11.5, x=0.004, ha="left", color=INK)
out = os.path.join(D, "exp011_sanity.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
