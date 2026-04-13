"""
Plot weight distributions for 10 random tables per layer (60 total).
Shows q_lut weights: for each table, the 64 entries averaged across outputs.
This reveals the "response function" shape each table has learned.
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu')

torch.manual_seed(0)
np.random.seed(0)

N_LAYERS = 6
N_PER_LAYER = 10
ROLE = 'q_lut'

fig, axes = plt.subplots(N_LAYERS, N_PER_LAYER, figsize=(20, 12))
fig.suptitle(f'Random table weight distributions ({ROLE}, mean across outputs)\n'
             f'x-axis: 64 table entries (input combinations), y-axis: mean weight value',
             fontsize=11)

for layer_idx in range(N_LAYERS):
    key = f'layers.{layer_idx}.{ROLE}.projection.weights'
    w = ckpt[key].float()   # [n_tables, 64, n_out]
    n_tables = w.shape[0]

    indices = torch.randperm(n_tables)[:N_PER_LAYER].sort().values

    w_mean = w.mean(dim=-1)   # [n_tables, 64] — average across outputs
    global_min = w_mean.min().item()
    global_max = w_mean.max().item()

    for col, t_idx in enumerate(indices.tolist()):
        ax = axes[layer_idx, col]
        vals = w_mean[t_idx].numpy()   # [64]

        ax.plot(vals, linewidth=0.8, color='steelblue')
        ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
        margin = (vals.max() - vals.min()) * 0.1 + 1e-6
        ax.set_ylim(vals.min() - margin, vals.max() + margin)
        ax.set_xticks([])
        ax.set_yticks([])

        if col == 0:
            ax.set_ylabel(f'L{layer_idx}', fontsize=8, rotation=0, labelpad=20)
        if layer_idx == 0:
            ax.set_title(f'#{t_idx}', fontsize=7)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'weight_examples.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'Saved to {out_path}')
