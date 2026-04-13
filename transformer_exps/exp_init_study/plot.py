"""
plot.py — Load snapshots.pt and plot weight entry distributions.

3 rows (init type) x 3 cols (step: 0, 1k, 5k).
Each cell: all 16 entry vectors from the snapshot, plotted sorted ascending.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
all_snaps = torch.load(os.path.join(EXP_DIR, 'snapshots.pt'), weights_only=False)

INITS = ['gaussian', 'uniform', 'bimodal']
STEPS = [0, 1000, 5000]
colors = plt.cm.tab20(np.linspace(0, 1, 16))

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle(
    'Weight entry distributions (sorted ascending) — layers.2.q_lut, table #42\n'
    'Each line = 1 of 16 entries. Rows: init type. Cols: training step.',
    fontsize=10
)

for row, init_name in enumerate(INITS):
    for col, step in enumerate(STEPS):
        ax = axes[row, col]
        table = all_snaps[init_name][step]   # [16, n_out]

        for i, entry in enumerate(table):
            vals = np.sort(entry)
            ax.plot(vals, color=colors[i], linewidth=1.0, alpha=0.8)

        ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        margin = max(abs(table.max()), abs(table.min())) * 0.15 + 1e-6
        ax.set_ylim(-abs(table).max() - margin, abs(table).max() + margin)
        ax.set_xticks([])
        ax.set_yticks([round(table.min(), 4), 0, round(table.max(), 4)])
        ax.tick_params(labelsize=6)

        if col == 0:
            ax.set_ylabel(init_name, fontsize=9)
        if row == 0:
            ax.set_title(f'step {step:,}', fontsize=9)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'init_study.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'Saved to {out_path}')
