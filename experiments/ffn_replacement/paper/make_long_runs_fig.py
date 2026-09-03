import csv, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-step validation bpb for the long (48k-step) runs, read straight from each run's
# committed metrics.csv (columns: step, train_loss, val_bpb) under ../runs/, so the figure
# tracks the run data rather than a private copy of it. (Upstream this read a duplicated
# long_run_data/ directory; the CSVs there were byte-identical to these, so the copies were
# dropped in favour of the single source of truth.)
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(BASE), "runs")
RUNS = [
    ("exp_n_0151_long48k_untied_vanilla",       "vanilla (0151)",          "#111111", "--"),
    ("exp_n_0152_long48k_tiny_H4d48_nap7_tph64", "LUT 0152 (tph64, nap7)", "#1D4ED8", "-"),
    ("exp_n_0155_long48k_from_0127",            "LUT 0155 (tph128, nap7)", "#C2410C", "-"),
    ("exp_n_0156_long48k_from_0128",            "LUT 0156 (tph64, nap8)",  "#2CA02C", "-"),
]
OMITTED = []

def curve(fn):
    steps, bpb = [], []
    for r in csv.DictReader(open(os.path.join(ROOT, fn, "metrics.csv"))):
        steps.append(int(r["step"])); bpb.append(float(r["val_bpb"]))
    return steps, bpb

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.5))

for fn, label, col, ls in RUNS:
    s, b = curve(fn)
    lw = 2.0 if ls == "--" else 1.6
    axL.plot(s, b, ls=ls, color=col, lw=lw, label=label)
    # log-log panel: drop step 0 (log axis)
    sp = [(x, y) for x, y in zip(s, b) if x > 0]
    axR.plot([x for x, _ in sp], [y for _, y in sp], ls=ls, color=col, lw=lw, label=label)

# LEFT: linear, framed on the convergence descent (top 1.35) so the small ~0.008 bpb gap
# between the runs reads as modest but visible; the early transient above 1.35 runs off the top.
axL.set_ylim(1.14, 1.35)
axL.set_xlim(0, 48000)
axL.set_xlabel("training step", fontsize=13)
axL.set_ylabel("validation bpb", fontsize=13)
axL.set_title("linear scale (convergence, top 1.35)", fontsize=13)
axL.tick_params(labelsize=12)
axL.grid(ls=":", lw=0.5, alpha=0.5)

# RIGHT: log-log, zoomed on the tail (drop the early transient below step 5k) to expose
# the convergence and the small final separation between the runs.
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(5000, 48000)
axR.set_ylim(1.14, 1.32)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
axR.xaxis.set_major_locator(FixedLocator([5000, 10000, 20000, 40000]))
axR.xaxis.set_major_formatter(FixedFormatter(["5k", "10k", "20k", "40k"]))
axR.xaxis.set_minor_locator(NullLocator())
axR.yaxis.set_major_locator(FixedLocator([1.15, 1.20, 1.25, 1.30]))
axR.yaxis.set_major_formatter(FixedFormatter(["1.15", "1.20", "1.25", "1.30"]))
axR.yaxis.set_minor_locator(NullLocator())
axR.set_xlabel("training step (log)", fontsize=13)
axR.set_ylabel("validation bpb (log)", fontsize=13)
axR.set_title("log-log (tail)", fontsize=13)
axR.tick_params(labelsize=12)
axR.grid(which="both", ls=":", lw=0.5, alpha=0.5)

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
fig.suptitle("Long 48k-step runs: validation bpb vs. training step", fontsize=14)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(os.path.join(BASE, "long_runs_fig.pdf"), bbox_inches="tight", pad_inches=0.1)
print("saved long_runs_fig.pdf")
print("plotted:", [r[1] for r in RUNS])
print("omitted (pending):", OMITTED)
