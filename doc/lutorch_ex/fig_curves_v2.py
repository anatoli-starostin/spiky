"""Figure 4 of lutorch_ex_report_izhikevich_2026-10-09.tex: validation bpb of the nine 48k runs.

Reads fig_curves_v2.csv next to this file (run, step, val_bpb). That file is the val_bpb history of the nine runs of
experiments/lutorch_ex_fused on branch research/lutorch_ex_fused_ablations (00_vanilla, 01_fmh_rational ...
08_qconf_n2), read from their W&B runs (project Spiky, group lutorch_ex_fused; one evaluation every 500 steps). The
final values equal the runs' RESULT.md.

    python fig_curves_v2.py      # writes fig_curves_v2.pdf next to this file (needs matplotlib)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RED, BLUE, GREEN, BLACK = "#c0392b", "#1f77b4", "#2e8b57", "#222222"
# (run, legend label, colour, linestyle): colour = family, solid = two rows per table, dotted = one row,
# dashed / dash-dot = int8 (two rows / one row), black = the dense FFN
SERIES = [
    ("05_conf_n2", "Confidence $n{=}2$", RED, "-"),
    ("00_vanilla", "dense FFN", BLACK, "-"),
    ("08_qconf_n2", "Confidence int8 $n{=}2$", RED, "--"),
    ("06_conf_n1", "Confidence $n{=}1$", RED, ":"),
    ("07_qconf_n1", "Confidence int8 $n{=}1$", RED, "-."),
    ("04_fms_softsign", "Manifesto soft, soft-sign", BLUE, "-"),
    ("03_fms_rational", "Manifesto soft, rational", GREEN, "-"),
    ("01_fmh_rational", "Manifesto hard, rational", GREEN, ":"),
    ("02_fmh_softsign", "Manifesto hard, soft-sign", BLUE, ":"),
]


def load():
    data = {}
    with open(os.path.join(HERE, "fig_curves_v2.csv"), newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["run"], []).append((int(row["step"]), float(row["val_bpb"])))
    return {k: sorted(v) for k, v in data.items()}


def main():
    data = load()
    fig, (ax, az) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0]})
    finals = []
    for run, label, color, ls in SERIES:
        s, v = zip(*data[run])
        for a in (ax, az):
            a.plot(s, v, color=color, linestyle=ls, linewidth=1.2, label=label)
        finals.append((v[-1], s[-1], min(v), color))
    # label the best value of each run at its end; nudge labels that would overlap
    placed = []
    for y, x, best, color in sorted(finals):
        yl = y
        for p in placed:
            if abs(yl - p) < 0.0027:
                yl = p + 0.0027
        placed.append(yl)
        az.annotate(f"{best:.4f}", (x, y), xytext=(x + 600, yl), textcoords="data", fontsize=6.3, va="center",
                    color=color)
    ax.set_xlim(0, 48000); ax.set_ylim(1.07, 1.45)
    ax.set_xlabel("step"); ax.set_ylabel("validation bpb"); ax.set_title("full run", fontsize=9)
    az.set_xlim(24000, 52500); az.set_ylim(1.08, 1.17)
    az.set_xlabel("step"); az.set_title("steps 24k–48k", fontsize=9)
    for a in (ax, az):
        a.grid(True, color="0.88", linewidth=0.6)
        a.tick_params(labelsize=8)
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x <= 48000 else ""))
    ax.legend(fontsize=6.3, frameon=False, loc="upper right", ncol=1)
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(HERE, "fig_curves_v2.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
