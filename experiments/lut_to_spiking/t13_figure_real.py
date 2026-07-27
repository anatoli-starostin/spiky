"""t13 — figure for the real-data milestone."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paths import out

INK = "#1d2733"; MUTED = "#6b7785"; GRID = "#dfe3e8"
GREEN = "#2f6f4f"; BLUE = "#3b6ea5"; PURPLE = "#7a5ea7"; ORANGE = "#c98a2b"

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.7))

# --- A: spiking circuit fidelity on real tokens, two input encodings ---
a = ax[0]
labels = ["row selected", "exact output\nlatency", "output\npair-order"]
rank = [100.0, 100.0, 100.0]
grid = [92.97, 93.25, 96.74]
x = np.arange(3); w = 0.36
a.bar(x - w/2, rank, w, color=GREEN, label="exact order code (rank latencies)")
a.bar(x + w/2, grid, w, color=BLUE, label="uniform 128-tick grid")
for xi, v in zip(x - w/2, rank):
    a.text(xi, v + 1.0, f"{v:.0f}%", ha="center", fontsize=9, color=GREEN, fontweight="bold")
for xi, v in zip(x + w/2, grid):
    a.text(xi, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, color=BLUE)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=9.5)
a.set_ylim(80, 106); a.set_ylabel("agreement with the real table (%)")
a.set_title("A  Spiking circuit vs the REAL table on REAL tokens\n"
            "exp025 layer 3 out_proj, table #0 (K=128, D=384), 256 val tokens",
            fontsize=10.5, color=INK)
a.legend(fontsize=9, loc="upper left", frameon=True, framealpha=0.95, edgecolor=GRID)

# --- B: model-level cost of coarse timing ---
b = ax[1]
ticks = [8, 16, 32, 64, 128, 256, 512]
dbpb = [0.0397, 0.0091, 0.0021, 0.0003, 0.0003, 0.0000, -0.0000]
b.plot(ticks, [d * 1000 for d in dbpb], "o-", color=PURPLE, lw=2, ms=6)
b.axhline(0, color=GREEN, ls="--", lw=1.5)
b.text(9, -3.4, "exact order code (rank) = 0.0 mb, lossless", color=GREEN, fontsize=9)
b.set_ylim(-6, 44)
b.axvline(64, color=ORANGE, ls=":", lw=1.6)
b.text(70, 22, "64 ticks (6 bit)\n+0.3 mb", color=ORANGE, fontsize=9.5)
b.set_xscale("log"); b.set_xticks(ticks); b.set_xticklabels([str(t) for t in ticks])
b.set_xlabel("input timing resolution of the stage (ticks)")
b.set_ylabel("val bpb penalty (millibits)")
b.set_title("B  What coarse timing costs the whole model\n"
            "layer-3 out_proj input quantised, val bpb baseline 1.2409",
            fontsize=10.5, color=INK)

for axx in ax:
    axx.grid(True, color=GRID, lw=0.8); axx.set_axisbelow(True)
    for s in ("top", "right"):
        axx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axx.spines[s].set_color(GRID)
    axx.tick_params(colors=MUTED, labelsize=9)
    axx.title.set_color(INK)
    axx.xaxis.label.set_color(MUTED); axx.yaxis.label.set_color(MUTED)

fig.tight_layout()
fig.savefig(out("real_table_spiking.png"), dpi=155,
            facecolor="white")
print("saved real_table_spiking.png")
