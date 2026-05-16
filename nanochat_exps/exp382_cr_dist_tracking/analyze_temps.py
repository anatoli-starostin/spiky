"""Analyse learned T_soft and T_sel temperatures over training."""
import os, re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(EXP_DIR, 'temperatures.csv'))

# Long format: melt
rows = []
for col in df.columns[1:]:
    m = re.match(r'L(\d+)\.(\w+)\.(T_soft|T_sel)', col)
    if not m: continue
    L, mod, kind = int(m.group(1)), m.group(2), m.group(3)
    for _, r in df.iterrows():
        rows.append(dict(step=int(r['step']), layer=L, module=mod, kind=kind, value=float(r[col])))
ldf = pd.DataFrame(rows)

# Summary: per-(module, kind) average and min/max at step 8000
last = ldf[ldf.step == ldf.step.max()]
print('=== End-of-training temperatures (step', ldf.step.max(), ') ===')
print(f'{"module":15s} {"kind":8s}  L0      L1      L2      L3      L4      L5')
for mod in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
    for kind in ('T_soft', 'T_sel'):
        vals = []
        for L in range(6):
            row = last[(last.module == mod) & (last.layer == L) & (last.kind == kind)]
            if len(row): vals.append(row['value'].iloc[0])
        print(f'{mod:15s} {kind:8s}  ' + '  '.join(f'{v:.3f}' for v in vals))

print('\n=== Initial values (step 0) ===')
first = ldf[ldf.step == 0]
print(f'{"module":15s} {"kind":8s}  L0      L1      L2      L3      L4      L5')
for mod in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
    for kind in ('T_soft', 'T_sel'):
        vals = []
        for L in range(6):
            row = first[(first.module == mod) & (first.layer == L) & (first.kind == kind)]
            if len(row): vals.append(row['value'].iloc[0])
        print(f'{mod:15s} {kind:8s}  ' + '  '.join(f'{v:.3f}' for v in vals))

# Plot trajectories: 2 rows (T_soft, T_sel) × 4 cols (modules)
fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
mod_names = ['qkv_lut', 'v_lut', 'out_proj', 'residual_lut']
colors = plt.cm.viridis([i/5 for i in range(6)])
for j, mod in enumerate(mod_names):
    for i, kind in enumerate(('T_soft', 'T_sel')):
        ax = axes[i, j]
        for L in range(6):
            sub = ldf[(ldf.module == mod) & (ldf.layer == L) & (ldf.kind == kind)].sort_values('step')
            ax.plot(sub['step'], sub['value'], color=colors[L], marker='.', markersize=3, label=f'L{L}')
        ax.set_title(f'{mod}: {kind}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        if j == 0: ax.set_ylabel(kind)
        if i == 1: ax.set_xlabel('step')
axes[0, 0].legend(loc='best', fontsize=8, ncols=2)
plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'temps_evolution.png')
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f'\nSaved {out_path}')
