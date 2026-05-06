"""Analyze LUT weight distributions per-layer for exp154.

For each (layer, role ∈ {qk_joint, v_lut, out_proj}):
  - LProjection.weights shape: [n_lookup_tables, n_entries_per_table, n_outputs]
    where n_lookup_tables = H * tph and n_entries_per_table = 2^nap.
  - Compute per-entry L2 norm across (n_outputs,).
  - "Dead entry" = L2 norm < 10× the init scale (0.001 → 0.01 cutoff).
  - Plot histogram of per-entry norms (log-y), per (layer, role).
  - Save fig + text summary.

Usage:
  python nanochat_exps/exp154_bit_attention/analyze_lut_weights.py
"""
import os
import sys
import json
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, 'checkpoint.pt')
OUT_PNG = os.path.join(EXP_DIR, 'lut_weight_distributions.png')
OUT_TXT = os.path.join(EXP_DIR, 'lut_weight_summary.txt')

INIT_NOISE = 0.001  # mhlut_init_std from config
DEAD_CUTOFF = 10 * INIT_NOISE  # entries below this = "essentially unused"

print(f'Loading {CKPT}...')
sd = torch.load(CKPT, map_location='cpu', weights_only=False)

# Find all LProjection weights and group by (layer, role)
lut_weights = {}  # (layer_idx, role) -> tensor [LUTs, entries, outputs]
for k, v in sd.items():
    # Match: layers.{i}.{role}.weights  (NOT layers.{i}.{role}.lookup.*)
    parts = k.split('.')
    if len(parts) != 4 or parts[0] != 'layers' or parts[3] != 'weights':
        continue
    if not parts[1].isdigit():
        continue
    layer_idx = int(parts[1])
    role = parts[2]
    if role not in ('qk_joint', 'v_lut', 'out_proj'):
        continue
    lut_weights[(layer_idx, role)] = v
    print(f'  {k}: {tuple(v.shape)}')

if not lut_weights:
    print('No LProjection weights found in checkpoint!')
    sys.exit(1)

layers = sorted(set(L for L, _ in lut_weights.keys()))
roles = ['qk_joint', 'v_lut', 'out_proj']

# Plot 3 columns (roles) x N_layers rows
fig, axes = plt.subplots(len(layers), 3, figsize=(15, 3 * len(layers)),
                          squeeze=False)

summary_lines = []
summary_lines.append(f'exp154 LUT weight analysis — init noise std={INIT_NOISE}, dead cutoff={DEAD_CUTOFF}\n')
summary_lines.append(f'{"layer":>5} {"role":>10} {"#LUTs":>8} {"#entr/LUT":>10} '
                     f'{"#outs":>6} {"total_ents":>12} {"dead":>10} '
                     f'{"dead%":>7} {"mean":>8} {"max":>8}\n')

for li, layer_idx in enumerate(layers):
    for ri, role in enumerate(roles):
        key = (layer_idx, role)
        if key not in lut_weights:
            axes[li, ri].set_visible(False)
            continue
        w = lut_weights[key]  # [n_luts, n_entries, n_outputs]
        n_luts, n_entries, n_outputs = w.shape

        # Per-entry L2 norm across the n_outputs axis -> [n_luts, n_entries]
        norms = w.float().norm(dim=-1)
        flat = norms.flatten()
        total = flat.numel()
        dead = (flat < DEAD_CUTOFF).sum().item()
        mean = flat.mean().item()
        mx = flat.max().item()

        ax = axes[li, ri]
        ax.hist(flat.numpy(), bins=80, log=True)
        ax.axvline(DEAD_CUTOFF, color='red', linestyle='--', linewidth=0.8,
                   label=f'dead cutoff ({DEAD_CUTOFF})')
        ax.set_title(f'L{layer_idx} {role}: {n_luts} LUTs × {n_entries} entr × {n_outputs} out\n'
                     f'dead={dead}/{total} ({100*dead/total:.1f}%), '
                     f'mean={mean:.3g}, max={mx:.3g}',
                     fontsize=9)
        ax.set_xlabel('per-entry L2 norm')
        ax.set_ylabel('count')

        summary_lines.append(
            f'{layer_idx:>5} {role:>10} {n_luts:>8} {n_entries:>10} '
            f'{n_outputs:>6} {total:>12} {dead:>10} '
            f'{100*dead/total:>6.1f}% {mean:>8.3g} {mx:>8.3g}\n'
        )

fig.suptitle('exp154 — LUT entry usage (per-entry L2 norm distribution)\n'
             '"Dead" entries (red dashed) have norms close to init noise → presumably unused',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=110, bbox_inches='tight')
print(f'\nSaved {OUT_PNG}')

with open(OUT_TXT, 'w') as f:
    f.writelines(summary_lines)
print(f'Saved {OUT_TXT}')

print('\n' + ''.join(summary_lines))
