"""
Plot weight examples from exp146 checkpoint.

6 layers x 1 random table each = 6 plots.
Each plot: all 16 entries sorted ascending (S-curve view).
Also marks D=256 grid lines to show alignment.
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(EXP_DIR, '..', '..'))

_src = open(os.path.join(EXP_DIR, 'train.py')).read()
_src = _src[:_src.rfind('sampler = make_sampler')]
exec(_src, globals())

from spiky.lutorch.multi_head_lut import MultiHeadLut

DEVICE = 'cpu'
torch.manual_seed(0)

model = LUTTransformerLinearUnemb().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True))
model.eval()

D = cfg['quant_reg_D']

# Collect one q_lut per layer
luts = [layer.q_lut for layer in model.layers]

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'exp146: weight entries sorted ascending (q_lut, 1 random table per layer)\nD={D} grid lines shown in red', fontsize=11)

colors = plt.cm.tab20(np.linspace(0, 1, 16))

for idx, (ax, lut) in enumerate(zip(axes.flat, luts)):
    w = lut.projection.weights.detach()  # [n_tables, 16, n_out]
    n_tables = w.shape[0]
    t = rng.integers(0, n_tables)
    table = w[t]  # [16, n_out]

    w_min = table.min().item()
    w_max = table.max().item()

    # Draw D=256 grid lines in range
    grid_lo = int(np.floor(w_min * D))
    grid_hi = int(np.ceil(w_max * D))
    for k in range(grid_lo, grid_hi + 1):
        ax.axhline(k / D, color='red', linewidth=0.3, alpha=0.4, zorder=0)

    # zoom into central 10% of range to see individual grid lines
    mid = (w_min + w_max) / 2
    half = (w_max - w_min) * 0.05  # show ±5% of full range
    y_lo, y_hi = mid - half, mid + half

    for i, entry in enumerate(table):
        vals = np.sort(entry.numpy())
        ax.scatter(range(len(vals)), vals, color=colors[i], s=4, alpha=0.7, linewidths=0, zorder=1)

    ax.set_ylim(y_lo, y_hi)
    ax.set_title(f'layer {idx}, table #{t}  (zoomed: {y_lo:.3f}–{y_hi:.3f})', fontsize=9)
    ax.set_xticks([])
    ax.set_ylabel('weight value', fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
out = os.path.join(EXP_DIR, 'weight_examples.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'Saved to {out}')
