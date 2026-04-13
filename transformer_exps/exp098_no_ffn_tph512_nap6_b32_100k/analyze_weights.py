"""
Weight analytics for exp098 checkpoint.
Covers:
1. Weight statistics per LUT and layer
2. Table utilization (variance across 64 entries)
3. Anchor pair coverage of input dims
4. Powers distribution
5. Layer depth comparison
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu')

# ── Parse all LUT tensors ────────────────────────────────────────────────────

luts = {}
for k, v in ckpt.items():
    if 'projection.weights' in k:
        name = k.replace('.projection.weights', '')
        luts[name] = {
            'weights': v,                        # [n_tables, 64, n_out]
            'anchor_a': ckpt[k.replace('projection.weights', 'lookup.anchor_pairs_a')],
            'anchor_b': ckpt[k.replace('projection.weights', 'lookup.anchor_pairs_b')],
            'powers':   ckpt[k.replace('projection.weights', 'lookup.powers')],
        }

layer_names = sorted(luts.keys())
print(f"Found {len(luts)} LUTs: {layer_names}\n")

# ── 1. Weight statistics ─────────────────────────────────────────────────────
print("=" * 70)
print("1. WEIGHT STATISTICS (projection.weights per LUT)")
print("=" * 70)
print(f"{'LUT':<35} {'shape':>18}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'|w|>1 %':>8}")
stats_rows = []
for name in layer_names:
    w = luts[name]['weights'].float()  # [T, 64, out]
    mean = w.mean().item()
    std  = w.std().item()
    wmin = w.min().item()
    wmax = w.max().item()
    large_frac = (w.abs() > 1.0).float().mean().item() * 100
    shape_str = str(tuple(w.shape))
    print(f"{name:<35} {shape_str:>18}  {mean:>8.4f}  {std:>8.4f}  {wmin:>8.4f}  {wmax:>8.4f}  {large_frac:>7.2f}%")
    stats_rows.append((name, mean, std, wmin, wmax, large_frac))

# ── 2. Table utilization ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("2. TABLE UTILIZATION  (variance across 64 entries, per table, then averaged)")
print("   High variance = entries are differentiated; low = dead tables")
print("=" * 70)
print(f"{'LUT':<35}  {'mean_var':>10}  {'dead(<0.001)%':>14}  {'min_var':>10}  {'max_var':>10}")
util_rows = []
for name in layer_names:
    w = luts[name]['weights'].float()   # [T, 64, out]
    # variance across the 64 entries for each (table, output) pair, then mean
    var_per_table = w.var(dim=1)        # [T, out]
    mean_var = var_per_table.mean().item()
    dead_pct = (var_per_table < 1e-3).float().mean().item() * 100
    min_var  = var_per_table.min().item()
    max_var  = var_per_table.max().item()
    print(f"{name:<35}  {mean_var:>10.5f}  {dead_pct:>13.2f}%  {min_var:>10.6f}  {max_var:>10.4f}")
    util_rows.append((name, mean_var, dead_pct))

# ── 3. Anchor pair coverage ──────────────────────────────────────────────────
print()
print("=" * 70)
print("3. ANCHOR PAIR COVERAGE  (how uniformly do anchors cover input dims?)")
print("=" * 70)

# gather per-LUT input dims and their anchor usage
for name in layer_names:
    a = luts[name]['anchor_a'].long()  # [T, nap]
    b = luts[name]['anchor_b'].long()
    all_dims = torch.cat([a.flatten(), b.flatten()])
    n_tables, nap = a.shape
    input_dim = all_dims.max().item() + 1  # approximate
    counts = torch.bincount(all_dims, minlength=input_dim).float()
    total  = counts.sum().item()
    entropy = -(counts / total * torch.log2((counts + 1e-9) / total)).sum().item()
    max_entropy = np.log2(input_dim) if input_dim > 1 else 1.0
    coverage_pct = (counts > 0).float().mean().item() * 100
    top5 = counts.topk(min(5, input_dim)).indices.tolist()
    print(f"{name:<35}  input_dim≈{input_dim:3d}  coverage={coverage_pct:5.1f}%  "
          f"entropy={entropy:.2f}/{max_entropy:.2f}  top5_dims={top5}")

# ── 4. Powers distribution ───────────────────────────────────────────────────
print()
print("=" * 70)
print("4. POWERS DISTRIBUTION  (learned per-anchor-pair importance weights)")
print("=" * 70)
print(f"{'LUT':<35}  {'powers (softmax-normalised)':>45}")
for name in layer_names:
    p = luts[name]['powers'].float().squeeze()  # [nap]
    # powers are likely raw logits — show softmax for interpretability
    sm = torch.softmax(p, dim=-1)
    vals = '  '.join(f'{x:.3f}' for x in sm.tolist())
    print(f"{name:<35}  [{vals}]")

# ── 5. Layer depth comparison ────────────────────────────────────────────────
print()
print("=" * 70)
print("5. LAYER DEPTH COMPARISON  (mean std of weights per layer)")
print("=" * 70)

layer_order = ['layers.0', 'layers.1', 'layers.2', 'layers.3', 'layers.4', 'layers.5', 'unembedder']
lut_roles   = ['q_lut', 'k_lut', 'v_lut', 'out_proj']

print(f"{'layer':<12}", end='')
for role in lut_roles:
    print(f"  {role:>12}", end='')
print()

depth_data = {role: [] for role in lut_roles}
for layer in layer_order:
    print(f"{layer:<12}", end='')
    for role in lut_roles:
        key = f"{layer}.{role}" if layer != 'unembedder' else None
        if key in luts:
            w = luts[key]['weights'].float()
            std = w.std().item()
            depth_data[role].append(std)
            print(f"  {std:>12.4f}", end='')
        elif layer == 'unembedder' and role == 'q_lut':
            # unembedder is a single LUT
            key = 'unembedder'
            w = luts[key]['weights'].float()
            std = w.std().item()
            print(f"  {'(unemb)':>5}{std:>7.4f}", end='')
            print('  ' + ' ' * 12 * 3, end='')
            break
        else:
            print(f"  {'—':>12}", end='')
    print()

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('exp098 LUT Weight Analytics', fontsize=14)

# Plot 1: weight std per LUT
ax = axes[0, 0]
names_short = [n.replace('layers.', 'L').replace('.', '/') for n in layer_names]
stds = [r[2] for r in stats_rows]
colors = ['steelblue' if 'q_lut' in n else 'darkorange' if 'k_lut' in n
          else 'green' if 'v_lut' in n else 'red' if 'out_proj' in n else 'purple'
          for n in layer_names]
ax.bar(range(len(names_short)), stds, color=colors)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=90, fontsize=6)
ax.set_title('Weight Std per LUT')
ax.set_ylabel('std')
legend_patches = [plt.Rectangle((0,0),1,1, color=c) for c in ['steelblue','darkorange','green','red','purple']]
ax.legend(legend_patches, ['q','k','v','out_proj','unemb'], fontsize=7)

# Plot 2: % large weights
ax = axes[0, 1]
large = [r[5] for r in stats_rows]
ax.bar(range(len(names_short)), large, color=colors)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=90, fontsize=6)
ax.set_title('% weights with |w| > 1')
ax.set_ylabel('%')

# Plot 3: table utilization (mean var)
ax = axes[0, 2]
mean_vars = [r[1] for r in util_rows]
ax.bar(range(len(names_short)), mean_vars, color=colors)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=90, fontsize=6)
ax.set_title('Table Utilization (mean entry variance)')
ax.set_ylabel('variance')

# Plot 4: dead table fraction
ax = axes[1, 0]
dead = [r[2] for r in util_rows]
ax.bar(range(len(names_short)), dead, color=colors)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=90, fontsize=6)
ax.set_title('Dead Tables % (var < 0.001)')
ax.set_ylabel('%')

# Plot 5: powers heatmap (layers x anchor_pairs)
ax = axes[1, 1]
powers_matrix = []
powers_labels = []
for name in layer_names:
    p = luts[name]['powers'].float().squeeze()
    sm = torch.softmax(p, dim=-1).numpy()
    powers_matrix.append(sm)
    powers_labels.append(name.replace('layers.', 'L').replace('.', '/'))
pm = np.array(powers_matrix)
im = ax.imshow(pm, aspect='auto', cmap='hot')
ax.set_yticks(range(len(powers_labels)))
ax.set_yticklabels(powers_labels, fontsize=6)
ax.set_xlabel('anchor pair index')
ax.set_title('Powers (softmax) per LUT')
plt.colorbar(im, ax=ax)

# Plot 6: weight std across layers for each role
ax = axes[1, 2]
for role, color in zip(lut_roles, ['steelblue','darkorange','green','red']):
    vals = depth_data[role]
    if vals:
        ax.plot(range(len(vals)), vals, marker='o', label=role, color=color)
ax.set_xticks(range(6))
ax.set_xticklabels([f'L{i}' for i in range(6)])
ax.set_title('Weight Std by Layer Depth')
ax.set_ylabel('std')
ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'weight_analytics.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
