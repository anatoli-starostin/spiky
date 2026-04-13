"""
For each layer: pick 1 random table, show 10 random entries from it.
Each entry is a [n_out=16] output vector — plotted as a bar chart.
This reveals: do different input combinations produce different outputs?
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu')

torch.manual_seed(7)
np.random.seed(7)

N_LAYERS = 6
N_ENTRIES = 10
ROLE = 'q_lut'

fig, axes = plt.subplots(N_LAYERS, N_ENTRIES, figsize=(20, 12))
fig.suptitle(f'10 random entries from 1 random table per layer ({ROLE}) — values sorted ascending\n'
             f'x-axis: rank, y-axis: weight value',
             fontsize=11)

for layer_idx in range(N_LAYERS):
    key = f'layers.{layer_idx}.{ROLE}.projection.weights'
    w = ckpt[key].float()       # [n_tables, 64, n_out]
    n_tables, n_entries, n_out = w.shape

    table_idx = torch.randint(n_tables, (1,)).item()
    table = w[table_idx]        # [64, n_out]

    entry_indices = torch.randperm(n_entries)[:N_ENTRIES].sort().values

    for col, e_idx in enumerate(entry_indices.tolist()):
        ax = axes[layer_idx, col]
        vals = np.sort(table[e_idx].numpy())   # [n_out], sorted ascending

        margin = (vals.max() - vals.min()) * 0.15 + 1e-6
        ax.bar(range(n_out), vals, color='steelblue', width=0.8)
        ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        ax.set_ylim(vals.min() - margin, vals.max() + margin)
        ax.set_xticks([])
        ax.set_yticks([])

        if col == 0:
            ax.set_ylabel(f'L{layer_idx}\ntbl#{table_idx}', fontsize=7, rotation=0, labelpad=30)
        if layer_idx == 0:
            ax.set_title(f'e#{e_idx}', fontsize=7)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'entry_outputs.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'Saved to {out_path}')
