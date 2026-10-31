import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = list(csv.DictReader(open("results/E2_domconf_ln_conf1_48k_metrics.csv")))
s = [int(x["step"]) for x in r]; v = [float(x["val_bpb"]) for x in r]
refs = [("dense FFN 1.08936", 1.08936, "#555"), ("FusedConfidence n=2 1.08653", 1.08653, "#36c"),
        ("FusedConfidence n=1 1.09698", 1.09698, "#07a"), ("FusedManifestoHard rational 1.13242", 1.13242, "#2a7"),
        ("FusedManifestoHard soft-sign 1.13780", 1.13780, "#e80")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, lo in zip(ax, (0, 20)):
    a.plot(s[lo:], v[lo:], "-", color="#c33", lw=2, label=f"E2 DomConf->LN->Conf1 ({v[-1]:.5f})")
    for lab, y, c in refs: a.axhline(y, ls="--", color=c, lw=1, label=lab + " (report, final)" if lo == 0 else None)
    a.set_xlabel("step"); a.set_ylabel("val_bpb"); a.grid(alpha=.3)
ax[0].set_ylim(1.05, 1.6); ax[0].set_title("E2, 48000 steps"); ax[1].set_ylim(1.07, 1.2); ax[1].set_title("zoom: steps >= 10000")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("val_bpb_48k_E2.png", dpi=130)
