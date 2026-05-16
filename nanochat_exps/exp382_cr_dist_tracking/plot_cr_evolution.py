"""Plot c_r distribution evolution per layer over training."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(EXP_DIR, 'c_r_stats.csv'))

# Module classification: extract layer index and module name
df[['layer', 'mod_name']] = df['module'].str.extract(r'L(\d+)\.(\w+)')
df['layer'] = df['layer'].astype(int)

mod_names = ['qkv_lut', 'v_lut', 'out_proj', 'residual_lut']
metrics = ['touch_frac', 'c_r_mean', 'frac_ge8', 'frac_ge64']

fig, axes = plt.subplots(len(metrics), len(mod_names), figsize=(18, 14), sharex=True)
colors = plt.cm.viridis([i/5 for i in range(6)])

for j, mod_name in enumerate(mod_names):
    for i, metric in enumerate(metrics):
        ax = axes[i, j]
        sub = df[df.mod_name == mod_name]
        for L in range(6):
            row = sub[sub.layer == L].sort_values('step')
            ax.plot(row['step'], row[metric], label=f'L{L}', color=colors[L],
                    marker='.', markersize=4, linewidth=1.5)
        ax.set_title(f'{mod_name}: {metric}')
        ax.grid(True, alpha=0.3)
        if metric in ('touch_frac', 'frac_ge8', 'frac_ge64'):
            ax.set_ylim(-0.05, 1.05)
        elif metric == 'c_r_mean':
            ax.set_yscale('log')
        if j == 0:
            ax.set_ylabel(metric)
        if i == len(metrics) - 1:
            ax.set_xlabel('step')
axes[0, 0].legend(loc='upper right', fontsize=8, ncols=2)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'c_r_evolution.png')
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f'Saved {out_path}')
