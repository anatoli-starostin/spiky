import csv, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-step validation bpb for the long (48k-step) runs, read from each run's committed
# metrics.csv (columns: step, train_loss, val_bpb), copied into long_run_data/ so the figure
# regenerates standalone. exp_n_0156 (from grid 0128) is omitted: it had no committed
# metrics.csv at figure time (still training) -- see the printed note.
ROOT = "/home/astarostin/projects/ffn-lut-paper/long_run_data"
RUNS = [
    ("0151_vanilla",      "vanilla (0151)",             "#111111", "--"),
    ("0152_tph64_nap7",   "LUT 0152 (tph64, nap7)",     "#1D4ED8", "-"),
    ("0155_tph128_nap7",  "LUT 0155 (tph128, nap7)",    "#C2410C", "-"),
    ("0156_tph64_nap8",   "LUT 0156 (tph64, nap8)",     "#2CA02C", "-"),
]
OMITTED = []

def curve(fn):
    steps, bpb = [], []
    for r in csv.DictReader(open(f"{ROOT}/{fn}.csv")):
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
fig.savefig("/home/astarostin/projects/ffn-lut-paper/long_runs_fig.pdf", bbox_inches="tight", pad_inches=0.1)
print("saved long_runs_fig.pdf")
print("plotted:", [r[1] for r in RUNS])
print("omitted (pending):", OMITTED)
