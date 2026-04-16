"""
Analyse exp239 checkpoint: table usage across layers.
For each LUT in each layer, measure:
  - Weight std (overall table activity)
  - Within-table variance (how much entries differ within a table)
  - Across-table variance (how much tables differ from each other)
  - Effective rank (how many tables contribute meaningfully)
"""
import sys, os, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EXP_DIR = 'transformer_exps/exp239_no_ffn_nap6_tph2048'
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu')

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

H = cfg['n_heads']
TPH = cfg['tph']
NAP = cfg.get('nap', 6)
TPH_OUT = cfg.get('tph_out', TPH)

# Identify weight tensors per layer
# Structure: layers.{i}.{q_lut,k_lut,v_lut,out_proj}.projection.weights
# Weights shape: [n_tables, n_entries, n_outputs]

print(f'exp239: nap_qk=5, nap_v=6, nap_out=6, tph={TPH}, tph_out={TPH_OUT}')
print(f'Checkpoint keys sample:')
for k in sorted(ckpt.keys())[:10]:
    print(f'  {k}: {ckpt[k].shape}')
print()

# Collect weight info per layer per component
components = ['q_lut', 'k_lut', 'v_lut', 'out_proj']

print('=' * 120)
print(f'{"Layer":>5} {"Component":>10} {"Shape":>25} | {"W_std":>8} {"W_mean":>8} | '
      f'{"IntraTab":>8} {"InterTab":>8} | {"Dead%":>6} {"Top10%":>7} {"Gini":>6}')
print('=' * 120)

for layer_idx in range(cfg['num_layers']):
    for comp in components:
        key = f'layers.{layer_idx}.{comp}.projection.weights'
        if key not in ckpt:
            continue
        w = ckpt[key]  # [n_tables, n_entries, n_outputs]
        n_tables, n_entries, n_outputs = w.shape

        # Overall stats
        w_std = w.std().item()
        w_mean = w.mean().item()

        # Within-table variance: for each table, how different are its entries?
        # Average std across entries within each table
        intra_table = w.std(dim=1).mean().item()  # std over entries, then mean over tables

        # Across-table variance: how different are tables from each other?
        # Collapse entries by mean, then measure std across tables
        table_means = w.mean(dim=1)  # [n_tables, n_outputs]
        inter_table = table_means.std(dim=0).mean().item()  # std across tables, mean over outputs

        # Table contribution: L2 norm of each table's output range
        # A "dead" table has all entries ~same (low intra-table variance)
        table_activity = w.std(dim=1).norm(dim=1)  # [n_tables] - L2 norm of per-output stds
        sorted_activity, _ = table_activity.sort(descending=True)

        # Dead tables: activity < 1% of max
        max_act = sorted_activity[0].item()
        dead_frac = (table_activity < 0.01 * max_act).float().mean().item() * 100

        # Top 10% contribution
        cumsum = sorted_activity.cumsum(0) / sorted_activity.sum()
        top10_idx = max(1, n_tables // 10)
        top10_contribution = cumsum[top10_idx - 1].item() * 100

        # Gini coefficient of table activity
        n = len(sorted_activity)
        activity_np = sorted_activity.flip(0).numpy()  # ascending
        cumsum_act = np.cumsum(activity_np)
        gini = (n + 1 - 2 * np.sum(cumsum_act) / cumsum_act[-1]) / n

        shape_str = f'{n_tables}x{n_entries}x{n_outputs}'
        print(f'{layer_idx:>5} {comp:>10} {shape_str:>25} | {w_std:>8.4f} {w_mean:>8.5f} | '
              f'{intra_table:>8.4f} {inter_table:>8.4f} | {dead_frac:>5.1f}% {top10_contribution:>6.1f}% {gini:>6.3f}')
    print('-' * 120)

# Summary: aggregate by component across layers
print()
print('=== SUMMARY: Mean across layers ===')
print(f'{"Component":>10} | {"W_std":>8} {"IntraTab":>8} {"InterTab":>8} | {"Dead%":>6} {"Top10%":>7} {"Gini":>6}')
print('-' * 70)

for comp in components:
    stats = []
    for layer_idx in range(cfg['num_layers']):
        key = f'layers.{layer_idx}.{comp}.projection.weights'
        if key not in ckpt:
            continue
        w = ckpt[key]
        n_tables, n_entries, n_outputs = w.shape
        intra = w.std(dim=1).mean().item()
        table_means = w.mean(dim=1)
        inter = table_means.std(dim=0).mean().item()
        table_activity = w.std(dim=1).norm(dim=1)
        sorted_act, _ = table_activity.sort(descending=True)
        max_act = sorted_act[0].item()
        dead = (table_activity < 0.01 * max_act).float().mean().item() * 100
        top10_idx = max(1, n_tables // 10)
        cumsum = sorted_act.cumsum(0) / sorted_act.sum()
        top10 = cumsum[top10_idx - 1].item() * 100
        n = len(sorted_act)
        act_np = sorted_act.flip(0).numpy()
        cs = np.cumsum(act_np)
        gini = (n + 1 - 2 * np.sum(cs) / cs[-1]) / n
        stats.append((w.std().item(), intra, inter, dead, top10, gini))

    if stats:
        arr = np.array(stats)
        means = arr.mean(axis=0)
        print(f'{comp:>10} | {means[0]:>8.4f} {means[1]:>8.4f} {means[2]:>8.4f} | {means[3]:>5.1f}% {means[4]:>6.1f}% {means[5]:>6.3f}')

# Layer-by-layer trend for out_proj (the big one)
print()
print('=== OUT_PROJ: Layer-by-layer trend ===')
print(f'{"Layer":>5} | {"W_std":>8} {"IntraTab":>8} {"InterTab":>8} | {"Dead%":>6} {"Gini":>6}')
print('-' * 60)
for layer_idx in range(cfg['num_layers']):
    key = f'layers.{layer_idx}.out_proj.projection.weights'
    if key not in ckpt:
        continue
    w = ckpt[key]
    intra = w.std(dim=1).mean().item()
    table_means = w.mean(dim=1)
    inter = table_means.std(dim=0).mean().item()
    table_activity = w.std(dim=1).norm(dim=1)
    sorted_act, _ = table_activity.sort(descending=True)
    max_act = sorted_act[0].item()
    dead = (table_activity < 0.01 * max_act).float().mean().item() * 100
    n = len(sorted_act)
    act_np = sorted_act.flip(0).numpy()
    cs = np.cumsum(act_np)
    gini = (n + 1 - 2 * np.sum(cs) / cs[-1]) / n
    print(f'{layer_idx:>5} | {w.std().item():>8.4f} {intra:>8.4f} {inter:>8.4f} | {dead:>5.1f}% {gini:>6.3f}')
