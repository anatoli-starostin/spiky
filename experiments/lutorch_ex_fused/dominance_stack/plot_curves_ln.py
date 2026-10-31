import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = [("A  ManifestoHard", "A_fmh_rational", "#2a7", "-"), ("B  Dominance->MH", "B_dom_fmh", "#c33", "-"),
        ("B1 Dom->LN(affine)->MH", "B1_dom_ln_fmh", "#e80", "-"), ("B2 Dom->LN(no affine)->MH", "B2_dom_lnna_fmh", "#a4c", "-"),
        ("C  Confidence n=2", "C_conf_n2", "#36c", "-"), ("D  MH(tph11)->MH", "D_fmh_fmh", "#777", "--")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for lab, d, c, ls in runs:
    r = list(csv.DictReader(open(f"results/{d}_4k_metrics.csv")))
    s = [int(x["step"]) for x in r]; v = [float(x["val_bpb"]) for x in r]
    for a, lo in zip(ax, (0, 6)):
        a.plot(s[lo:], v[lo:], ls, marker="o", ms=3, color=c, label=f"{lab} ({v[-1]:.4f})" if lo == 0 else None)
ax[0].set_title("val_bpb, 4000-step runs"); ax[1].set_title("zoom: steps >= 1750")
for a in ax: a.set_xlabel("step"); a.set_ylabel("val_bpb"); a.grid(alpha=.3)
ax[1].set_ylim(1.32, 1.5); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("val_bpb_4k_ln.png", dpi=130)
