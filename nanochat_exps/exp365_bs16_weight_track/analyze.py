"""Quick analysis of weight_deltas.csv: aggregate per module type and plot trajectories."""
import os, re
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(EXP_DIR, 'weight_deltas.csv'))
df['weight_norm'] = df['weight_norm'].astype(float)
df['delta_norm'] = df['delta_norm'].astype(float)
df['rel_delta']  = df['rel_delta'].astype(float)

def classify(name: str) -> tuple[str, int | None]:
    m = re.match(r'layers\.(\d+)\.(qkv_lut|v_lut|out_proj|residual_lut)\.weights', name)
    if m: return m.group(2), int(m.group(1))
    m = re.match(r'layers\.(\d+)\.(qkv_lut|v_lut|out_proj|residual_lut)\.log_soft_score_temp', name)
    if m: return f'{m.group(2)}.T_soft', int(m.group(1))
    m = re.match(r'layers\.(\d+)\.(qkv_lut|v_lut|out_proj|residual_lut)\.log_select_temp', name)
    if m: return f'{m.group(2)}.T_sel', int(m.group(1))
    m = re.match(r'layers\.(\d+)\.(q_norm|k_norm|ln_e)\.(weight|bias)', name)
    if m: return f'{m.group(2)}.{m.group(3)}', int(m.group(1))
    if name.startswith('tok_emb_E'): return 'tok_emb_E', None
    if name.startswith('unembedder'): return 'unembedder', None
    if name.startswith('ln_final'): return 'ln_final', None
    return 'other', None

df[['module_type', 'layer']] = df['param_name'].apply(lambda s: pd.Series(classify(s)))

# Aggregate per (step, module_type) — sum delta_norm, weight_norm; recompute rel as ratio.
agg = (df.groupby(['step', 'module_type'])
         .agg(weight_norm=('weight_norm', 'sum'),
              delta_norm=('delta_norm', 'sum'))
         .reset_index())
agg['rel_delta'] = agg['delta_norm'] / agg['weight_norm']

# ---- Plot 1: rel_delta per module type over training (log y)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) LUT weights only
lut_types = ['qkv_lut', 'v_lut', 'out_proj', 'residual_lut']
ax = axes[0, 0]
for mt in lut_types:
    sub = agg[agg.module_type == mt].sort_values('step')
    ax.plot(sub['step'], sub['rel_delta'], label=mt, marker='.', markersize=3)
ax.set(xlabel='step', ylabel='rel_delta (||ΔW|| / ||W||)', title='LUT weights: relative change per eval-interval', yscale='log')
ax.legend(loc='best'); ax.grid(True, alpha=0.3)

# (b) Non-LUT params
nonlut_types = ['tok_emb_E', 'unembedder', 'ln_final',
                'q_norm.weight', 'k_norm.weight', 'ln_e.weight']
ax = axes[0, 1]
for mt in nonlut_types:
    sub = agg[agg.module_type == mt].sort_values('step')
    if sub.empty: continue
    ax.plot(sub['step'], sub['rel_delta'], label=mt, marker='.', markersize=3)
ax.set(xlabel='step', ylabel='rel_delta', title='Non-LUT params: relative change per eval-interval', yscale='log')
ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)

# (c) Per-layer LUT rel_delta — use out_proj as a representative
ax = axes[1, 0]
sub_layer = (df[df.module_type == 'out_proj']
             .groupby(['step', 'layer']).agg({'weight_norm': 'sum', 'delta_norm': 'sum'}).reset_index())
sub_layer['rel_delta'] = sub_layer['delta_norm'] / sub_layer['weight_norm']
for L in sorted(sub_layer['layer'].unique()):
    s = sub_layer[sub_layer.layer == L].sort_values('step')
    ax.plot(s['step'], s['rel_delta'], label=f'L{L}', marker='.', markersize=3)
ax.set(xlabel='step', ylabel='rel_delta', title='out_proj per layer', yscale='log')
ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)

# (d) weight_norm growth per module type
ax = axes[1, 1]
for mt in lut_types + ['tok_emb_E', 'unembedder']:
    sub = agg[agg.module_type == mt].sort_values('step')
    if sub.empty: continue
    ax.plot(sub['step'], sub['weight_norm'], label=mt, marker='.', markersize=3)
ax.set(xlabel='step', ylabel='||W||_2 (summed across tensors of this type)', title='Weight norm growth')
ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'weight_deltas_analysis.png')
plt.savefig(out_path, dpi=110)
print(f'Saved {out_path}')

# ---- Summary table: total cumulative drift per module type
totals = (df.groupby('module_type')
            .agg(total_delta=('delta_norm', 'sum'),
                 mean_rel_delta=('rel_delta', 'mean'),
                 final_weight_norm=('weight_norm', 'last'),
                 n_params=('param_name', 'nunique'))
            .sort_values('total_delta', ascending=False))
print('\nPer-module-type summary (sorted by cumulative L2 drift):')
print(totals.to_string())
totals.to_csv(os.path.join(EXP_DIR, 'weight_deltas_summary.csv'))

# ---- Headline: which modules move most relative to their own size, on average?
rel_totals = (df.groupby('module_type')['rel_delta'].mean().sort_values(ascending=False))
print('\nModules sorted by AVG rel_delta per eval-interval (higher = changing more per step):')
print(rel_totals.to_string())
