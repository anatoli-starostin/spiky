import csv, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-step eval bpb for the super-long 144k-step run and its matched vanilla baseline, read from
# each run's committed metrics.csv (columns: step, train_loss, val_bpb), copied into
# superlong_data/ so the figure regenerates standalone. Sources (branch
# research/hyperplane_ffn_next, commit d362c7ac):
#   experiments/hyperplane_ffn/exp_n_0158_long144k_from_0127/metrics.csv  (LUT, 0127 architecture)
#   experiments/hyperplane_ffn/exp_n_0157_long144k_vanilla/metrics.csv    (untied dense baseline)
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "superlong_data")
RUNS = [
    ("vanilla_0157",       "vanilla (exp_n_0157)",              "#111111", "--"),
    ("lut_0158_from_0127", "LUT exp_n_0158 (0127 arch)",        "#1D4ED8", "-"),
]

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
    axR.plot(s, b, ls=ls, color=col, lw=lw, label=label)

# LEFT: linear, full 144k horizon, framed on the convergence descent (top 1.35) so the early
# transient runs off the top and the near-identical descent is the visible part.
axL.set_ylim(1.14, 1.35)
axL.set_xlim(0, 144000)
axL.set_xlabel("training step", fontsize=13)
axL.set_ylabel("validation bpb", fontsize=13)
axL.set_title("linear scale (full 144k, top 1.35)", fontsize=13)
from matplotlib.ticker import FixedLocator, FixedFormatter
axL.xaxis.set_major_locator(FixedLocator([0, 48000, 96000, 144000]))
axL.xaxis.set_major_formatter(FixedFormatter(["0", "48k", "96k", "144k"]))
axL.tick_params(labelsize=12)
axL.grid(ls=":", lw=0.5, alpha=0.5)

# RIGHT: log-log view of the full trajectory (step ~5k onward), matching make_long_runs_fig.py,
# framed tightly on the bpb range so the two curves and their tail convergence are legible.
from matplotlib.ticker import NullLocator
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(5000, 144000)
axR.set_ylim(1.14, 1.57)
axR.xaxis.set_major_locator(FixedLocator([5000, 10000, 20000, 40000, 80000, 144000]))
axR.xaxis.set_major_formatter(FixedFormatter(["5k", "10k", "20k", "40k", "80k", "144k"]))
axR.xaxis.set_minor_locator(NullLocator())
axR.yaxis.set_major_locator(FixedLocator([1.15, 1.20, 1.30, 1.40, 1.50]))
axR.yaxis.set_major_formatter(FixedFormatter(["1.15", "1.20", "1.30", "1.40", "1.50"]))
axR.yaxis.set_minor_locator(NullLocator())
axR.set_xlabel("training step (log)", fontsize=13)
axR.set_ylabel("validation bpb (log)", fontsize=13)
axR.set_title("log-log (full trajectory)", fontsize=13)
axR.tick_params(labelsize=12)
axR.grid(which="both", ls=":", lw=0.5, alpha=0.5)

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
fig.suptitle("Super-long run: 144k steps (~3.5B tokens), tiny reprojection LUT vs dense baseline",
             fontsize=14)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(os.path.join(BASE, "superlong_fig.pdf"),
            bbox_inches="tight", pad_inches=0.1)
print("saved superlong_fig.pdf")
lutf = curve("lut_0158_from_0127")[1][-1]; vanf = curve("vanilla_0157")[1][-1]
print(f"final bpb: LUT {lutf:.6f}  vanilla {vanf:.6f}  gap {lutf-vanf:+.6f}")
