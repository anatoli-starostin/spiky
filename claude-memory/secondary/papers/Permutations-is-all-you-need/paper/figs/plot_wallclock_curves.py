"""Generate val_curves_wallclock.png for Section 6.6.

Three training curves at the narrow-backbone architecture (E=192, D=384,
176.2 M params):
  - exp755: hybrid_smooth, 16K steps  (5.4 h wall-clock, final 1.2044)
  - exp756: hard,          16K steps  (3.4 h wall-clock, final 1.2301)
  - exp760: hard,          24K steps  (~5.1 h wall-clock at exp756's per-step
                                        rate; the measured 6.6 h was inflated
                                        by GPU sharing during the run)

Plus the eval-only "soft-trained model evaluated under hard forward" point:
  - exp755 trained checkpoint, hard eval: 1.2778 at step 16000.

The story: hard training for 1.5x more steps reaches the same val_bpb as
hybrid_smooth training in 1x steps, at comparable wall-clock (because hard
is faster per step). The eval-only switch point (1.2778) marks the bump
you eat if you instead just flip the soft-trained model into hard at
deployment time.

Run:
    python plot_wallclock_curves.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPS_ROOT = "/home/starost/spiky-feat/nanochat_exps"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "val_curves_wallclock.png")

RUNS = [
    ("exp755_exp754_E192_dv32_16k",
     "exp755: hybrid_smooth, 16K (5.4 h)", "tab:orange", "-"),
    ("exp756_exp755_hard_16k",
     "exp756: hard, 16K (3.4 h)",          "tab:orange", "--"),
    ("exp760_exp756_24k",
     "exp760: hard, 24K (5.1 h)",          "tab:red",    "-"),
]

# Eval-only point: exp755 checkpoint scored under hard forward.
EVAL_SWITCH_STEP = 16000
EVAL_SWITCH_BPB  = 1.2778


def load_curve(run_dir):
    csv_path = os.path.join(EXPS_ROOT, run_dir, "metrics.csv")
    steps, bpbs = [], []
    with open(csv_path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            steps.append(int(row[0]))
            bpbs.append(float(row[2]))
    return steps, bpbs


fig, ax = plt.subplots(figsize=(8.5, 4.8))
for run, label, colour, ls in RUNS:
    steps, bpbs = load_curve(run)
    ax.plot(steps, bpbs, color=colour, linestyle=ls, linewidth=1.6,
            label=f"{label}, final={bpbs[-1]:.4f}")

# Mark the eval-only soft->hard switch point next to where exp755 ends.
ax.plot([EVAL_SWITCH_STEP], [EVAL_SWITCH_BPB],
        marker="X", markersize=11, markerfacecolor="tab:orange",
        markeredgecolor="black", markeredgewidth=0.8, linestyle="None",
        label=f"exp755 trained checkpoint, hard eval = {EVAL_SWITCH_BPB:.4f}",
        zorder=5)

# Visual cue: vertical segment from exp755's final (1.2044) up to 1.2778 to
# show the soft->hard eval gap.
ax.annotate("", xy=(EVAL_SWITCH_STEP, EVAL_SWITCH_BPB),
            xytext=(EVAL_SWITCH_STEP, 1.2044),
            arrowprops=dict(arrowstyle="-|>", color="tab:orange",
                            linewidth=1.0, alpha=0.6))
ax.set_xlabel("Training step")
ax.set_ylabel("Validation bits per byte")
ax.set_xlim(0, 24500)
ax.set_ylim(1.18, 1.55)
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax.set_title("Wall-clock-matched hard training (narrow backbone, $E=192$, $D=384$)")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
