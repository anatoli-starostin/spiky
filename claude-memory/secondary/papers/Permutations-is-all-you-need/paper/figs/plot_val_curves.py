"""Generate val_curves.png: val_bpb vs step for the five LUTGPT/vanilla runs
mentioned in Section 6 of lutgpt_research_report.tex.

Run from spiky-feat (the nanochat_exps worktree of the feature branch),
pointed at this paper repo's figs/ directory:

    python plot_val_curves.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPS_ROOT = "/home/starost/spiky-feat/nanochat_exps"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_curves.png")

# (run dir, label, params (M), colour, linestyle)
RUNS = [
    ("exp709_untied_vanilla_bs48_16k",
     "Vanilla baseline (exp709, 35.8M)",          35.8, "tab:gray",   "-"),
    ("exp754_exp752_hybrid_smooth_16k",
     "LUTGPT $E{=}D{=}384$, hybrid_smooth (exp754, 276.8M)",
                                                  276.8, "tab:blue",   "-"),
    ("exp752_exp750_recipe_16k",
     "LUTGPT $E{=}D{=}384$, hard (exp752, 276.8M)",
                                                  276.8, "tab:blue",   "--"),
    ("exp755_exp754_E192_dv32_16k",
     "LUTGPT $E{=}192,\\,D{=}384$, hybrid_smooth (exp755, 176.2M)",
                                                  176.2, "tab:orange", "-"),
    ("exp756_exp755_hard_16k",
     "LUTGPT $E{=}192,\\,D{=}384$, hard (exp756, 176.2M)",
                                                  176.2, "tab:orange", "--"),
]


def load_curve(run_dir):
    csv_path = os.path.join(EXPS_ROOT, run_dir, "metrics.csv")
    steps, bpbs = [], []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            steps.append(int(row[0]))
            bpbs.append(float(row[2]))
    return steps, bpbs


fig, ax = plt.subplots(figsize=(8.5, 4.5))
for run, label, _params, colour, ls in RUNS:
    steps, bpbs = load_curve(run)
    ax.plot(steps, bpbs, color=colour, linestyle=ls, linewidth=1.6,
            label=f"{label}, final={bpbs[-1]:.4f}")

ax.set_xlabel("Training step")
ax.set_ylabel("Validation bits per byte")
ax.set_xlim(0, 16000)
ax.set_ylim(1.15, 1.6)
ax.grid(True, alpha=0.35)
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax.set_title("Validation loss curves: vanilla baseline and LUTGPT configurations")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
