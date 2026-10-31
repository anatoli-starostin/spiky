import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
runs = [("A  FusedManifestoHard rational", "A_fmh_rational", "#2a7"), ("B  Dominance(hard) + ManifestoHard", "B_dom_fmh", "#c33"),
        ("C  FusedConfidence n=2", "C_conf_n2", "#36c")]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for lab, d, c in runs:
    r = list(csv.DictReader(open(f"runs/{d}_4k/metrics.csv")))
    s = [int(x["step"]) for x in r]; v = [float(x["val_bpb"]) for x in r]
    for a, lo in zip(ax, (0, 6)):
        a.plot(s[lo:], v[lo:], "o-", ms=3, color=c, label=f"{lab} ({v[-1]:.4f})" if lo == 0 else None)
ax[0].set_title("val_bpb, 4000-step runs"); ax[1].set_title("zoom: steps >= 1750")
for a in ax: a.set_xlabel("step"); a.set_ylabel("val_bpb"); a.grid(alpha=.3)
ax[1].set_ylim(1.32, 1.5); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("val_bpb_4k.png", dpi=130)
