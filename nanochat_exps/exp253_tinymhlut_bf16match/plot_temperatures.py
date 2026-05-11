"""Plot exp250 learnable temperature dynamics from temperatures.csv."""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(EXP_DIR, "temperatures.csv")
out_path = os.path.join(EXP_DIR, "temperatures.png")

with open(csv_path) as f:
    rows = list(csv.reader(f))
header = rows[0]
data = [[float(c) for c in r] for r in rows[1:]]
steps = [r[0] for r in data]
cols = {name: [r[i] for r in data] for i, name in enumerate(header)}

# Group columns by (lut_kind, temp_kind)
groups = {
    ("qk_joint", "T_soft"): [], ("qk_joint", "T_sel"): [],
    ("v_lut", "T_soft"):    [], ("v_lut", "T_sel"):    [],
    ("out_proj", "T_soft"): [], ("out_proj", "T_sel"): [],
    ("out_v2d", "T"):       [],
}
for name in header[1:]:
    # name format: "L<idx>.<kind>.<temp>"
    layer, kind, temp = name.split(".")
    groups[(kind, temp)].append((int(layer[1:]), name))
for k in groups:
    groups[k].sort()

fig, axes = plt.subplots(4, 2, figsize=(13, 14))
axes = axes.flatten()

panels = [
    ("qk_joint", "T_soft", "qk_joint  T_soft"),
    ("qk_joint", "T_sel",  "qk_joint  T_sel"),
    ("v_lut",    "T_soft", "v_lut  T_soft"),
    ("v_lut",    "T_sel",  "v_lut  T_sel"),
    ("out_proj", "T_soft", "out_proj  T_soft"),
    ("out_proj", "T_sel",  "out_proj  T_sel"),
    ("out_v2d",  "T",      "out_v2d  T (NEW: learnable in exp250)"),
]
cmap = plt.get_cmap("viridis")
for ax, (kind, temp, title) in zip(axes, panels):
    g = groups[(kind, temp)]
    for li, name in g:
        ax.plot(steps, cols[name], color=cmap(li / max(len(g) - 1, 1)),
                label=f"L{li}", linewidth=1.6)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("T")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_yscale("log")

# Hide last unused axis
axes[-1].axis("off")

last_step = int(steps[-1])
fig.suptitle(f"exp250 learnable temperatures @ step {last_step} "
             f"(init=0.5 everywhere, log-scale Y)",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out_path, dpi=110)
print(f"Wrote {out_path} (last step {last_step})")
