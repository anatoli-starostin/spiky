"""Figure 3 of the lutorch_ex report: the seven val_bpb learning curves of the 48k sweep.

Reads experiments/lutorch_ex/*/metrics.csv. Two schemas exist: the four VM runs log eval rows only
(`step,val_bpb,...`), the three worker runs are full wandb histories (`_step,val_bpb,...` with an empty
val_bpb on every non-eval row). Both reduce to (step, val_bpb) pairs after dropping blanks.

    .venv/bin/python make_fig_curves.py      # writes fig_curves.pdf next to this file
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "..", "experiments", "lutorch_ex")

# (run folder, legend label, family colour, linestyle)   -- family: Manifesto / SoftSign / Confidence
SERIES = [
    ("lutorch_ex_abl47_conf_1004_1148",          "Confidence $n{=}2$",          "#c0392b", "-"),
    ("lutorch_ex_abl47_quant_1004_1148",         "QuantConfidence $n{=}2$",     "#c0392b", "--"),
    ("lutorch_ex_abl47_conf_n1_vm_1004_1806",    "Confidence $n{=}1$",          "#c0392b", ":"),
    ("lutorch_ex_abl47_fss_smooth_vm_1004_1806", "FusedSoftSignSmooth",         "#1f77b4", "-"),
    ("lutorch_ex_abl47_fss_hard_wk_1004_1510",   "FusedSoftSignHard",           "#1f77b4", ":"),
    ("lutorch_ex_abl47_fms_wk_1004_1510",        "FusedManifestoSoft",          "#2e8b57", "-"),
    ("lutorch_ex_abl47_fmh_wk_1004_1510",        "FusedManifestoHard",          "#2e8b57", ":"),
]


def load(run):
    steps, vals = [], []
    with open(os.path.join(RUNS, run, "metrics.csv"), newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("val_bpb", "")
            if v in ("", "nan", "NaN"):
                continue
            s = row.get("step") or row.get("_step")
            steps.append(float(s))
            vals.append(float(v))
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    return [steps[i] for i in order], [vals[i] for i in order]


def main():
    fig, (ax, az) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for run, label, color, ls in SERIES:
        s, v = load(run)
        for a in (ax, az):
            a.plot(s, v, color=color, linestyle=ls, linewidth=1.3, label=label)
        az.annotate(f"{min(v):.4f}", (s[-1], v[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, va="center", color=color)
    ax.set_xlim(0, 48000); ax.set_ylim(1.08, 1.45)
    ax.set_xlabel("step"); ax.set_ylabel("validation bpb"); ax.set_title("full run", fontsize=9)
    az.set_xlim(24000, 48000); az.set_ylim(1.095, 1.19)
    az.set_xlabel("step"); az.set_title("steps 24k–48k", fontsize=9)
    for a in (ax, az):
        a.grid(True, color="0.88", linewidth=0.6)
        a.tick_params(labelsize=8)
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.legend(fontsize=6.8, frameon=False, loc="upper right", ncol=1)
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(HERE, "fig_curves.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
