"""t09 — the two figures for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import out

INK = "#1d2733"; MUTED = "#6b7785"; GRID = "#dfe3e8"
C_EXACT = "#2f6f4f"; C_FAM = "#3b6ea5"; C_CTRL = "#b4553a"; C_ACC = "#7a5ea7"

# ---- panel A: per-table cost/fidelity (t06, real trained table K=64, D=384) ----
fams = [  # (label, synapses, pair-order %)
    ("min-plus bus r=2",   896, 52.1), ("r=4", 1792, 57.9), ("r=8", 3584, 63.6),
    ("r=16", 7168, 68.8), ("r=32", 14336, 78.2),
]
lines = [("fitted bit lines", 4608, 60.1)]
fact = [("factored g=3", 4608, 63.6), ("factored g=2", 6144, 68.2)]
sparse = [("q=.1", 3224, 55.6), ("q=.25", 6802, 67.0), ("q=.5", 12994, 84.4), ("q=.75", 19054, 96.2)]
rnd = [("random table, r=8", 3584, 61.4), ("random, bit lines", 4608, 56.5), ("random, factored g=2", 6144, 63.7)]

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))
a = ax[0]
a.plot([x[1] for x in fams], [x[2] for x in fams], "o-", color=C_FAM, lw=1.8, ms=5,
       label="min-plus bus (rank r)")
a.plot([x[1] for x in sparse], [x[2] for x in sparse], "s-", color=C_ACC, lw=1.8, ms=5,
       label="sparse override (quantile q)")
a.plot([x[1] for x in fact], [x[2] for x in fact], "^-", color="#2f8f8f", lw=1.8, ms=6,
       label="factored addressing")
a.plot([lines[0][1]], [lines[0][2]], "D", color="#c98a2b", ms=7, label="fitted bit lines (12/output)")
a.plot([x[1] for x in rnd], [x[2] for x in rnd], "x", color=C_CTRL, ms=8, mew=2,
       label="same families on a RANDOM table")
a.plot([24576], [100], "*", color=C_EXACT, ms=18, label="exact construction (1 synapse/entry)")
a.axhline(30.3, color=MUTED, ls=":", lw=1.4)
a.text(700, 32, "row ignored (control)", color=MUTED, fontsize=9)
a.set_xscale("log")
a.set_xlabel("synapses for one table (K=64 rows x D=384 outputs)")
a.set_ylabel("pair-order fidelity of the output (%)")
a.set_title("A  One real trained table: nothing compresses\n"
            "(the trained table behaves like a random one)", fontsize=11, color=INK)
a.set_ylim(25, 104)
a.legend(fontsize=8.5, loc="lower right", frameon=False)

# ---- panel B: latency resolution sweep (t08, whole head of 256 tables) ----
span = [1, 3, 7, 15, 30, 60, 120, 255]
pair = [79.6, 80.7, 89.9, 95.2, 97.6, 98.8, 99.4, 99.7]
b = ax[1]
b.plot(span, pair, "o-", color=C_EXACT, lw=2, ms=6)
b.axhline(61.0, color=MUTED, ls=":", lw=1.4)
b.text(1.2, 62.5, "row ignored (control)", color=MUTED, fontsize=9)
b.axvline(15, color=C_ACC, ls="--", lw=1.4)
b.text(16, 84, "15 ticks (4 bit) -> 95%", color=C_ACC, fontsize=9.5)
b.set_xscale("log")
b.set_xticks(span); b.set_xticklabels([str(s) for s in span])
b.set_xlabel("distinguishable latencies per stage (ticks of dynamic range)")
b.set_ylabel("head pair-order fidelity (%)")
b.set_title("B  How much timing resolution a stage really needs\n"
            "(whole head: sum of 256 trained tables)", fontsize=11, color=INK)
b.set_ylim(58, 102)

for axx in ax:
    axx.grid(True, color=GRID, lw=0.8)
    axx.set_axisbelow(True)
    for s in ("top", "right"):
        axx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axx.spines[s].set_color(GRID)
    axx.tick_params(colors=MUTED, labelsize=9)
    axx.title.set_color(INK)
    axx.xaxis.label.set_color(MUTED); axx.yaxis.label.set_color(MUTED)

fig.tight_layout()
fig.savefig(out("lut2spiking_costfidelity.png"), dpi=155,
            facecolor="white")
print("saved lut2spiking_costfidelity.png")
