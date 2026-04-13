"""
Table clustering analysis for exp098 checkpoint.

For each LUT, clusters the tables by their weight vectors using K-Means.
Questions:
- How many distinct functions does a LUT actually learn?
- Are most tables near-duplicates (wasted capacity)?
- Do clusters differ across layers / roles (q/k/v/out_proj)?
"""
import sys, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu')

# ── Load LUT weights ───────────────────────────────────────────────────────────
luts = {}
for k, v in ckpt.items():
    if 'projection.weights' in k:
        name = k.replace('.projection.weights', '')
        luts[name] = v.float()   # [n_tables, 64, n_out]

layer_names = sorted(luts.keys())
print(f"Found {len(luts)} LUTs\n")

# ── Cluster each LUT ──────────────────────────────────────────────────────────
K_VALUES = [2, 4, 8, 16, 32]

print(f"{'LUT':<35}  {'n_tables':>8}  {'inertia@8':>10}  {'sil@8':>7}  {'best_k':>6}  {'best_sil':>8}")
print("-" * 85)

results = {}
for name in layer_names:
    w = luts[name]               # [T, 64, n_out]
    T = w.shape[0]
    flat = w.reshape(T, -1).numpy()   # [T, 64*n_out]

    # Try multiple K values, pick best by silhouette
    best_k, best_sil, best_km = None, -1, None
    inertia_at_8 = None
    sil_at_8 = None

    for k in K_VALUES:
        if k >= T:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(flat)
        sil = silhouette_score(flat, labels, sample_size=min(T, 500))
        if k == 8:
            inertia_at_8 = km.inertia_
            sil_at_8 = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_km = km

    results[name] = {
        'flat': flat,
        'best_k': best_k,
        'best_sil': best_sil,
        'best_km': best_km,
        'n_tables': T,
        'inertia_at_8': inertia_at_8,
        'sil_at_8': sil_at_8,
    }

    print(f"{name:<35}  {T:>8}  {inertia_at_8 if inertia_at_8 else 0:>10.1f}  "
          f"{sil_at_8 if sil_at_8 else 0:>7.3f}  {best_k:>6}  {best_sil:>8.3f}")

# ── Cluster size distribution for best K ──────────────────────────────────────
print("\n\nCLUSTER SIZE DISTRIBUTION (best K)")
print("-" * 85)
for name in layer_names:
    r = results[name]
    km = r['best_km']
    if km is None:
        continue
    labels = km.labels_
    counts = np.bincount(labels, minlength=r['best_k'])
    sorted_counts = sorted(counts, reverse=True)
    pct = [f"{c/r['n_tables']*100:.0f}%" for c in sorted_counts]
    print(f"{name:<35}  k={r['best_k']}  sizes: {' '.join(pct)}")

# ── PCA + cluster plots ────────────────────────────────────────────────────────
roles = ['q_lut', 'k_lut', 'v_lut', 'out_proj']
n_layers = 6
fig, axes = plt.subplots(n_layers, len(roles), figsize=(16, 22))
fig.suptitle('Table Clusters (PCA 2D) — exp098', fontsize=13)

for layer_idx in range(n_layers):
    for col, role in enumerate(roles):
        ax = axes[layer_idx, col]
        name = f'layers.{layer_idx}.{role}'
        if name not in results:
            ax.axis('off')
            continue

        r = results[name]
        flat = r['flat']
        km = r['best_km']
        if km is None:
            ax.axis('off')
            continue

        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(flat)
        labels = km.labels_
        k = r['best_k']

        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap='tab10',
                             s=6, alpha=0.6, vmin=0, vmax=9)
        var_explained = pca.explained_variance_ratio_.sum() * 100
        ax.set_title(f'L{layer_idx}.{role}\nk={k} sil={r["best_sil"]:.2f} var={var_explained:.0f}%',
                     fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'table_clusters.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")

# ── Summary: effective capacity ────────────────────────────────────────────────
print("\n\nEFFECTIVE CAPACITY SUMMARY")
print("(best_k / n_tables = fraction of tables that are 'distinct')")
print("-" * 60)
for role in roles:
    vals = []
    for layer_idx in range(n_layers):
        name = f'layers.{layer_idx}.{role}'
        if name in results:
            r = results[name]
            vals.append(r['best_k'] / r['n_tables'])
    if vals:
        print(f"{role:<12}  mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")
