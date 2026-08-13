"""exp23 full QAT (N=22) — training curves, eval, and what it did to the RAW readout."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe")
OUT = os.path.join(HERE, "analysis", "exp23_qat_n22_full.png")

INK, MUTE = "#2b2b2b", "#6b6b6b"
CS = ["#4E79A7", "#59A14F", "#B4453C"]
C_PAR, C_QAT, C_GRID = "#8b8b8b", "#4E79A7", "#9aa5b1"

H = {s: json.load(open(os.path.join(HERE, f"qat_s{s}.json")))["history"] for s in (0, 1, 2)}
par = json.load(open(os.path.join(PROBE, "parent.json")))
qat = {s: json.load(open(os.path.join(PROBE, f"qat_s{s}.json"))) for s in (0, 1, 2)}
raw_par = np.load(os.path.join(PROBE, "parent_raw.npy")).astype(np.float64)
raw_q = np.load(os.path.join(PROBE, "qat_s0_raw.npy")).astype(np.float64)

EVAL = {"parent": 6037.1, 0: 6456.8, 1: 6476.4, 2: 6411.9}
ESD = {"parent": 946.0, 0: 302.1, 1: 370.5, 2: 329.5}

fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6))
fig.subplots_adjust(left=0.05, right=0.985, top=0.845, bottom=0.075,
                    wspace=0.25, hspace=0.36)
(a1, a2, a3), (b1, b2, b3) = axes

# ---------------------------------------------------------------- A1 return
for s in (0, 1, 2):
    u = [r["update"] for r in H[s]]
    a1.plot(u, [r["ep_ret_mean"] for r in H[s]], lw=1.9, color=CS[s], label=f"seed {s}")
a1.axhline(5966.3, color=C_PAR, ls="--", lw=1.6)
a1.annotate("parent 5966 (its own recorded return)", xy=(20, 5966.3), xytext=(0, -15),
            textcoords="offset points", fontsize=8.5, color=C_PAR)
a1.set_xlabel("update", fontsize=9.5)
a1.set_ylabel("ep_ret_mean (on-policy)", fontsize=9.5)
a1.legend(frameon=False, fontsize=9)
a1.set_title("A · Return over training\nall three seeds track each other; the early dip is "
             "episode-length\nwarm-up, not a regression", fontsize=9.8, loc="left", color=INK)

# ---------------------------------------------------------------- A2 KL
for s in (0, 1, 2):
    a2.plot([r["update"] for r in H[s]], [r["kl"] for r in H[s]], lw=1.6, color=CS[s])
a2.axhline(0.03, color=INK, ls="--", lw=1.6)
a2.annotate("early-stop threshold 0.03", xy=(200, 0.0305), fontsize=8.5,
            color=INK, fontweight="bold")
a2.set_xlabel("update", fontsize=9.5)
a2.set_ylabel("approx_kl", fontsize=9.5)
a2.set_title("B · KL — it rides the threshold for the first ~2/3\nof the run, so many "
             "updates ARE truncated;\nthe cosine decay is what finally brings it under",
             fontsize=9.8, loc="left", color=INK)

# ---------------------------------------------------------------- A3 epochs + lr
ep = np.array([r["epochs_done"] for r in H[0]])
u0 = [r["update"] for r in H[0]]
a3.plot(u0, ep, lw=1.6, color=CS[0], label="epochs completed (seed 0)")
a3.set_ylim(0, 4.4)
a3.set_ylabel("epochs completed of 4", fontsize=9.5)
a3.set_xlabel("update", fontsize=9.5)
ax2 = a3.twiny()
ax2.set_xticks([])
a3.annotate(f"mean {ep.mean():.2f} of 4 epochs\n"
            f"({100*ep.mean()/4:.0f}% of the nominal gradient budget)",
            xy=(0.03, 0.12), xycoords="axes fraction", fontsize=9, color=INK,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
a3.legend(frameon=False, fontsize=9, loc="lower right")
a3.set_title("C · Effective gradient budget\nthe KL early-stop truncates most updates "
             "until the LR\nhas annealed far enough", fontsize=9.8, loc="left", color=INK)

# ---------------------------------------------------------------- B1 eval
labs = ["parent", "QAT s0", "QAT s1", "QAT s2"]
vals = [EVAL["parent"], EVAL[0], EVAL[1], EVAL[2]]
sds = [ESD["parent"], ESD[0], ESD[1], ESD[2]]
cols = [C_PAR] + CS
xs = np.arange(4)
b1.bar(xs, vals, width=0.6, color=cols, zorder=3)
b1.errorbar(xs, vals, yerr=sds, fmt="none", ecolor=INK, elinewidth=1.5, capsize=6, zorder=4)
for x, v in zip(xs, vals):
    b1.annotate(f"{v:,.0f}", xy=(x, v), xytext=(0, 30), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color=INK)
b1.axhline(EVAL["parent"], color=C_PAR, ls="--", lw=1.4, zorder=2)
b1.set_xticks(xs)
b1.set_xticklabels(labs, fontsize=9)
b1.set_ylim(0, 7600)
b1.set_ylabel("deterministic return, both quantizers ON", fontsize=9.5)
b1.set_title(f"D · Eval in the DEPLOYMENT config\nmean {np.mean(vals[1:]):,.0f} vs parent "
             f"{EVAL['parent']:,.0f}  (+{np.mean(vals[1:])-EVAL['parent']:,.0f}),\n"
             "and the spread more than halves", fontsize=9.8, loc="left", color=INK)

# ---------------------------------------------------------------- B2 raw sprawl
LO, HI = -4.2, 4.2
b2.hist(raw_par[:, 0], bins=180, range=(LO, HI), color=C_PAR, alpha=0.75,
        label="parent, RAW pre-clip", zorder=3)
b2.hist(raw_q[:, 0], bins=180, range=(LO, HI), color=C_QAT, alpha=0.62,
        label="after QAT, RAW pre-clip", zorder=4)
ym = b2.get_ylim()[1]
for L in np.linspace(-1, 1, 22):
    b2.plot([L, L], [0, ym], color=C_GRID, lw=0.9, alpha=0.8, zorder=2)
for s in (-1.0, 1.0):
    b2.plot([s, s], [0, ym * 1.06], color="#B4453C", lw=2.0, zorder=6)
b2.set_xlim(LO, HI)
b2.set_yticks([])
b2.set_xlabel("raw action mean, dim 0", fontsize=9.5)
b2.legend(frameon=False, fontsize=8.5, loc="upper left")
b2.set_title("E · The RAW readout did NOT come in-band\n"
             f"{par['pct_outside_all']:.1f}% outside before, "
             f"{np.mean([qat[s]['pct_outside_all'] for s in (0,1,2)]):.1f}% after — the "
             "22-level grid\n(grey) only ever sees the middle", fontsize=9.8, loc="left",
             color=INK)

# ---------------------------------------------------------------- B3 delay span
dims = np.arange(6)
w = 0.38
b3.bar(dims - w / 2, par["delay_span_per_dim"], w, color=C_PAR, label="parent", zorder=3)
b3.bar(dims + w / 2, np.mean([qat[s]["delay_span_per_dim"] for s in (0, 1, 2)], axis=0),
       w, color=C_QAT, label="after QAT (3-seed mean)", zorder=3)
b3.set_xticks(dims)
b3.set_xticklabels([f"d{i}" for i in dims], fontsize=9)
b3.set_xlabel("action dim", fontsize=9.5)
b3.set_ylabel("Stage-3 delay span (ticks)", fontsize=9.5)
b3.legend(frameon=False, fontsize=9)
dm = np.mean([qat[s]["dmax"] for s in (0, 1, 2)])
b3.set_title(f"F · Cost to the spiking build went UP, not down\n"
             f"dmax {par['dmax']} -> {dm:.0f} ticks, episode "
             f"{par['n_ticks_est']} -> {np.mean([qat[s]['n_ticks_est'] for s in (0,1,2)]):.0f}; "
             f"|w|max {par['w_absmax']:.3f} -> "
             f"{np.mean([qat[s]['w_absmax'] for s in (0,1,2)]):.3f}",
             fontsize=9.8, loc="left", color=INK)

for p in (a1, a2, a3, b1, b2, b3):
    for sp in ("top", "right"):
        p.spines[sp].set_visible(False)
    p.grid(color="0.93", lw=0.6, axis="y")
    p.set_axisbelow(True)
    p.tick_params(labelsize=8.5)

fig.suptitle("exp23 · combined QAT fine-tune, N=22 output + 128-bucket input, 384 updates "
             "x 3 seeds from deploy_matched seed 2\n"
             "The policy got substantially better under quantization — but the raw readout "
             "still sprawls, so the spiking build gets no cheaper",
             fontsize=12.5, x=0.005, ha="left", color=INK)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=135)
print("wrote", OUT)
