"""Generate val_curves_tph_halved.png for Section 6.7.

Two curves at matched training-token budget:
  - exp760: hard, tph=256/256/512/256/256, 24K steps, eff bs=24576
            -> 590 M tokens, 5.1 h wall-clock (expected), final val_bpb=1.2048
  - exp764: hard, tph=128/128/256/128/128, 24K steps, eff bs=49152
            -> 1.18 B tokens, 5.7 h wall-clock, final val_bpb=1.2116

exp764's effective batch is 2x larger than exp760's, so each of its steps sees
2x the tokens. To compare the two curves on a token-matched axis we stretch
exp764's step axis by 2 (so its 24K-step run plots as if it were a 48K-step
run at exp760's batch size). The x-axis is then labeled in "equivalent steps
at the exp760 batch size" -- equivalently, training tokens / 24,576.

Run:
    python plot_tph_halved_curves.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPS_ROOT = "/home/starost/spiky-feat/nanochat_exps"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "val_curves_tph_halved.png")

RUNS = [
    ("exp760_exp756_24k",
     "exp760: tph=256/256/512/256/256, 176.2 M params, 5.1 h",
     "tab:red",   "-",  1.0),
    ("exp764_tph_halved_24k_bs96",
     "exp764: tph=128/128/256/128/128,  97.5 M params, 5.7 h",
     "tab:blue",  "-",  2.0),
]


def load_curve(run_dir, x_scale):
    csv_path = os.path.join(EXPS_ROOT, run_dir, "metrics.csv")
    steps, bpbs = [], []
    with open(csv_path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            steps.append(int(row[0]) * x_scale)
            bpbs.append(float(row[2]))
    return steps, bpbs


fig, ax = plt.subplots(figsize=(8.5, 4.8))
for run, label, colour, ls, x_scale in RUNS:
    steps, bpbs = load_curve(run, x_scale)
    ax.plot(steps, bpbs, color=colour, linestyle=ls, linewidth=1.6,
            label=f"{label}, final={bpbs[-1]:.4f}")

ax.set_xlabel("Equivalent training tokens (steps at exp760's effective batch of 24,576)")
ax.set_ylabel("Validation bits per byte")
ax.set_xlim(0, 49000)
ax.set_ylim(1.18, 1.55)
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax.set_title("Halving the table count, trading compute for tokens (narrow backbone, $E=192$, $D=384$)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
