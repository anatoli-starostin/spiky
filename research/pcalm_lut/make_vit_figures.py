"""Curves and a final-test bar chart for the ViT autoencoder sweep (runs_vit/).

Reads run.json only. Okabe-Ito palette; the dataviz skill validator is not installed on this machine so
the palette check it asks for was NOT run.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "runs_vit"); O = os.path.join(R, "plots")
os.makedirs(O, exist_ok=True)
INK = "#222222"
plt.rcParams.update({"axes.edgecolor": "#BBBBBB", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "font.size": 8.5, "axes.grid": True,
                     "grid.color": "#E4E4E4", "figure.facecolor": "white", "axes.facecolor": "white"})
OK = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B"]
ARMS = ["linear-s5000", "vit-k1-e2d2-nowarm", "vit-k1-e2d2-warm", "vit-k1-e4d4-nowarm",
        "vit-k1-e4d4-warm", "vit-p2-k1-e4d4-warm", "vit-k8-e4d4-warm", "vit-p2-k8-e4d4-warm",
        "vit-k8-e4d4-warm-d32", "vit-k8-e4d4-warm-d128"]
D = {n: json.load(open(f"{R}/{n}/run.json")) for n in ARMS if os.path.exists(f"{R}/{n}/run.json")}
lin = D["linear-s5000"]["summary"]["test_mse"]
mean_te = D["linear-s5000"]["summary"]["mean_baseline_test"]

# two panels rather than one: 10 arms against an 8-colour palette would mean reusing a hue, and the
# rule is that categorical colours are never cycled. Small multiples are the remedy, so the pooled-global
# arms and the structured-bottleneck arms get a panel each, with the linear baseline drawn in both.
GROUPS = [("pooled global latent (k=1): one vector broadcast to every position",
           ["vit-k1-e2d2-nowarm", "vit-k1-e2d2-warm", "vit-k1-e4d4-nowarm", "vit-k1-e4d4-warm",
            "vit-p2-k1-e4d4-warm"]),
          ("structured latent (k=8 tokens x 8 dims) via learned queries and cross-attention",
           ["vit-k8-e4d4-warm", "vit-p2-k8-e4d4-warm", "vit-k8-e4d4-warm-d32",
            "vit-k8-e4d4-warm-d128"])]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
for ax, (title, arms) in zip(axes, GROUPS):
    h = D["linear-s5000"]["hist"]
    ax.plot([r["step"] for r in h], [r["eval/train_mse"] for r in h], "-", color="#555555", lw=1.6,
            label="linear-s5000")
    ax.plot([r["step"] for r in h], [r["eval/test_mse"] for r in h], "--", color="#555555", lw=1.1)
    for i, n in enumerate(arms):
        if n not in D: continue
        c = OK[i % len(OK)]; hh = D[n]["hist"]
        ax.plot([r["step"] for r in hh], [r["eval/train_mse"] for r in hh], "-", color=c, lw=1.6, label=n)
        ax.plot([r["step"] for r in hh], [r["eval/test_mse"] for r in hh], "--", color=c, lw=1.1, alpha=.9)
    ax.axhline(lin, color="#555555", ls=(0, (4, 3)), lw=1.2)
    ax.annotate(f"linear @5000: {lin:.5f}", xy=(5000, lin), xytext=(-5, 6), textcoords="offset points",
                ha="right", fontsize=8, color="#555555")
    ax.axhline(mean_te, color="#999999", ls=(0, (2, 2)), lw=1.0)
    ax.annotate(f"per-pixel mean {mean_te:.4f}", xy=(5000, mean_te), xytext=(-5, 4),
                textcoords="offset points", ha="right", fontsize=8, color="#777777")
    ax.set(xlabel="step", yscale="log", title=title)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(alpha=.3); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
axes[0].set_ylabel("MSE (standardised)")
fig.suptitle("ViT autoencoder at 14x14 — solid = train, dashed = test", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{O}/vit_curves.png", dpi=170)

fig2, ax2 = plt.subplots(figsize=(10, 4.8))
names = [n for n in ARMS if n in D]
vals = [D[n]["summary"]["test_mse"] for n in names]
cols = ["#555555" if n.startswith("linear") else OK[i % len(OK)] for i, n in enumerate(names)]
ax2.bar(range(len(names)), vals, color=cols, edgecolor="white", linewidth=1.2, zorder=3)
ax2.axhline(lin, color="#555555", ls=(0, (4, 3)), lw=1.3, zorder=4)
ax2.annotate(f"linear baseline {lin:.5f}", xy=(len(names)-0.5, lin), xytext=(-4, 5),
             textcoords="offset points", ha="right", fontsize=8.5, color="#555555")
for i, v in enumerate(vals):
    ax2.annotate(f"{v:.4f}", xy=(i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
ax2.set(ylabel="final held-out MSE (standardised)", title="Final test MSE at 5000 steps")
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.grid(alpha=.3, axis="y")
fig2.tight_layout(); fig2.savefig(f"{O}/vit_final_bars.png", dpi=170)
print("wrote", f"{O}/vit_curves.png"); print("wrote", f"{O}/vit_final_bars.png")
