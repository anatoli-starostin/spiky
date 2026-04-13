"""
Analysis of trained exp184 model. Produces:
  - analysis_data.json: numeric stats
  - analysis_plots/: distribution plots
  - analysis_report.html: self-contained HTML report
"""
import sys, os, json, base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.lut_attention import LUTAttentionV3

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(EXP_DIR, 'analysis_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])


# ── Model (with activation capture) ──────────────────────────────────────────

def make_score_attn(seed_offset=0):
    E, P, H = cfg['embedding_dim'], cfg['positional_dim'], cfg['n_heads']
    lut = MultiHeadLut(
        input_dim=2*E+P, n_heads=H, n_outputs=1,
        n_anchor_pairs=cfg['attention_nap'], tables_per_head=cfg['attention_tph'],
        smooth_mode=False, n_alternatives=cfg['n_alternatives'],
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )
    return LUTAttentionV3(lut, seq_len=SEQ_LEN, causal=True, include_diagonal=True)

def make_value_lut(seed_offset=200):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['value_nap'], tables_per_head=cfg['value_tph'],
        smooth_mode=False, n_alternatives=cfg['n_alternatives'],
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )

def make_out_proj(seed_offset=400):
    E, H, d_v = cfg['embedding_dim'], cfg['n_heads'], cfg['d_v']
    return MultiHeadLut(
        input_dim=H*d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_proj_nap'], tables_per_head=cfg['out_proj_tph'],
        smooth_mode=False, n_alternatives=cfg['n_alternatives'],
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        H, d_v = cfg['n_heads'], cfg['d_v']
        self.score_attn = make_score_attn(seed_offset=layer_idx)
        self.value_lut = make_value_lut(seed_offset=200+layer_idx)
        self.out_proj = make_out_proj(seed_offset=400+layer_idx)
        self.H, self.d_v = H, d_v
        self._act = {}

    def forward(self, x, rel_pe):
        B, T, E = x.shape
        H, d_v = self.H, self.d_v
        self._act['input'] = x.detach()
        raw_scores = self.score_attn(x, rel_pe).squeeze(-1).permute(0,3,1,2)
        self._act['raw_scores'] = raw_scores.detach()
        attn_weights = F.softmax(raw_scores, dim=-1)
        self._act['attn_weights'] = attn_weights.detach()
        v = self.value_lut(x.reshape(B*T, E))
        self._act['values'] = v.detach()
        v = v.reshape(B, T, H, d_v).permute(0,2,1,3)
        attn_out = (attn_weights @ v).permute(0,2,1,3).reshape(B, T, H*d_v)
        self._act['attn_out'] = attn_out.detach()
        proj_out = self.out_proj(attn_out.reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, E)
        self._act['proj_out'] = proj_out.detach()
        out = x + proj_out
        self._act['output'] = out.detach()
        return out

class LUTTransformerV3Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        E, P = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.rel_pe = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(E, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer in self.layers:
            x = layer(x, self.rel_pe)
        return self.unembedder(x)


# ── Load ──────────────────────────────────────────────────────────────────────

model = LUTTransformerV3Softmax().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
print("Loaded checkpoint")

sampler = make_sampler(DEVICE, random_seed=1)
report = {}  # collects all numeric data


# ── Helpers ───────────────────────────────────────────────────────────────────

def tensor_stats(t):
    t = t.float().flatten()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'q01': 0, 'q25': 0,
                'q50': 0, 'q75': 0, 'q99': 0, 'zeros_pct': 0, 'numel': t.numel()}
    q = torch.quantile(finite, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=finite.device))
    return {
        'mean': finite.mean().item(), 'std': finite.std().item(),
        'min': finite.min().item(), 'max': finite.max().item(),
        'q01': q[0].item(), 'q25': q[1].item(), 'q50': q[2].item(),
        'q75': q[3].item(), 'q99': q[4].item(),
        'zeros_pct': (finite == 0).float().mean().item(),
        'numel': t.numel(), 'finite_pct': finite.numel() / t.numel(),
    }


def plot_hist(data_dict, title, filename, bins=100):
    """Plot multiple distributions on one figure."""
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, data_dict.items()):
        v = vals.float().cpu().flatten()
        v = v[torch.isfinite(v)].numpy()
        if len(v) == 0:
            ax.text(0.5, 0.5, 'all non-finite', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.hist(v, bins=bins, density=True, alpha=0.7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('value')
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


def plot_heatmap(tensor, title, filename, cmap='RdBu_r'):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tensor.cpu().numpy(), aspect='auto', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path


# ── 1. Weight distributions ──────────────────────────────────────────────────

print("1. Analysing weights...")
report['weights'] = {}
for i, layer in enumerate(model.layers):
    ldata = {}
    for name, w in [
        ('score', layer.score_attn.multi_head_lut.projection.weights.data),
        ('value', layer.value_lut.projection.weights.data),
        ('outproj', layer.out_proj.projection.weights.data),
    ]:
        ldata[name] = tensor_stats(w)
    report['weights'][f'layer_{i}'] = ldata

# Plot weight distributions per layer
for i, layer in enumerate(model.layers):
    plot_hist({
        'score': layer.score_attn.multi_head_lut.projection.weights.data,
        'value': layer.value_lut.projection.weights.data,
        'outproj': layer.out_proj.projection.weights.data,
    }, f'Layer {i} Weight Distributions', f'weights_layer{i}.png')

# All layers combined
all_score = torch.cat([l.score_attn.multi_head_lut.projection.weights.data.flatten() for l in model.layers])
all_value = torch.cat([l.value_lut.projection.weights.data.flatten() for l in model.layers])
all_outproj = torch.cat([l.out_proj.projection.weights.data.flatten() for l in model.layers])
plot_hist({'score': all_score, 'value': all_value, 'outproj': all_outproj},
          'All Layers Combined Weight Distributions', 'weights_all.png')


# ── 2. Activation distributions ──────────────────────────────────────────────

print("2. Analysing activations...")
model.eval()
with torch.no_grad():
    batch = next(iter(sampler.testing_batches_iterator(128)))
    inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=DEVICE)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = batch[:, :-1].long()
    logits = model(inp)

report['activations'] = {}
for i, layer in enumerate(model.layers):
    ldata = {}
    for name, act in layer._act.items():
        ldata[name] = tensor_stats(act)
    report['activations'][f'layer_{i}'] = ldata

    plot_hist({
        'input': layer._act['input'],
        'raw_scores': layer._act['raw_scores'],
        'values': layer._act['values'],
        'proj_out': layer._act['proj_out'],
    }, f'Layer {i} Activations', f'activations_layer{i}.png')

# Attention patterns: average attention map per head
for i, layer in enumerate(model.layers):
    aw = layer._act['attn_weights'].mean(0)  # [H, T, T]
    for h in range(aw.shape[0]):
        plot_heatmap(aw[h], f'Layer {i} Head {h} Attention (avg)', f'attn_layer{i}_head{h}.png', cmap='viridis')

# Attention entropy
report['attention_entropy'] = {}
for i, layer in enumerate(model.layers):
    aw = layer._act['attn_weights']
    ent = -(aw * (aw + 1e-10).log()).sum(-1)
    report['attention_entropy'][f'layer_{i}'] = tensor_stats(ent)

# Logits and output
report['output'] = {
    'logits': tensor_stats(logits),
    'entropy': tensor_stats(-(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)),
}
plot_hist({'logits': logits}, 'Output Logits', 'logits.png')


# ── 3. Gradient distributions ────────────────────────────────────────────────

print("3. Analysing gradients...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

x = sampler.sample_training_batch(128).long()
inp = torch.empty_like(x)
inp[:, 0] = BOS_ID
inp[:, 1:] = x[:, :-1]
logits = model(inp)
B, T, V = logits.shape
loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))
optimizer.zero_grad()
loss.backward()

report['loss'] = loss.item()
report['gradients'] = {}

for i, layer in enumerate(model.layers):
    ldata = {}
    for name, w in [
        ('score', layer.score_attn.multi_head_lut.projection.weights),
        ('value', layer.value_lut.projection.weights),
        ('outproj', layer.out_proj.projection.weights),
    ]:
        if w.grad is not None:
            ldata[f'{name}_grad'] = tensor_stats(w.grad)
            ratio = w.grad.abs() / (w.data.abs() + 1e-8)
            ldata[f'{name}_grad_weight_ratio'] = tensor_stats(ratio)
    report['gradients'][f'layer_{i}'] = ldata

# Plot gradient distributions
for i, layer in enumerate(model.layers):
    grads = {}
    for name, w in [
        ('score', layer.score_attn.multi_head_lut.projection.weights),
        ('value', layer.value_lut.projection.weights),
        ('outproj', layer.out_proj.projection.weights),
    ]:
        if w.grad is not None:
            grads[f'{name}_grad'] = w.grad.data
    plot_hist(grads, f'Layer {i} Gradients', f'gradients_layer{i}.png')

# Grad/weight ratio — log scale, exclude near-zero weights
def plot_log_ratio(data_dict, title, filename):
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, data_dict.items()):
        v = vals.float().cpu().flatten()
        # Exclude zero ratios and take log10
        v = v[v > 0]
        if len(v) == 0:
            ax.text(0.5, 0.5, 'all zero', ha='center', va='center', transform=ax.transAxes)
        else:
            v = torch.log10(v).numpy()
            ax.hist(v, bins=100, density=True, alpha=0.7)
            ax.axvline(x=-2, color='green', linestyle='--', alpha=0.5, label='0.01')
            ax.axvline(x=-1, color='orange', linestyle='--', alpha=0.5, label='0.1')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='1.0')
            ax.legend(fontsize=7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('log10(|grad/weight|)')
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    return path

for i, layer in enumerate(model.layers):
    ratios = {}
    for name, w in [
        ('score', layer.score_attn.multi_head_lut.projection.weights),
        ('value', layer.value_lut.projection.weights),
        ('outproj', layer.out_proj.projection.weights),
    ]:
        if w.grad is not None:
            # Only include weights with |w| > 0.01 to avoid near-zero denominator
            mask = w.data.abs() > 0.01
            if mask.any():
                ratios[f'{name}'] = (w.grad[mask].abs() / w.data[mask].abs()).data
    plot_log_ratio(ratios, f'Layer {i} log10(|grad/weight|)', f'grad_ratio_layer{i}.png')

# Dead weights
report['dead_weights'] = {}
total_zero, total_params = 0, 0
for i, layer in enumerate(model.layers):
    for name, w in [
        ('score', layer.score_attn.multi_head_lut.projection.weights),
        ('value', layer.value_lut.projection.weights),
        ('outproj', layer.out_proj.projection.weights),
    ]:
        if w.grad is not None:
            nz = (w.grad == 0).sum().item()
            nt = w.grad.numel()
            total_zero += nz
            total_params += nt
            report['dead_weights'][f'layer_{i}_{name}'] = {'zero': nz, 'total': nt, 'pct': nz/nt}

report['dead_weights']['total'] = {'zero': total_zero, 'total': total_params, 'pct': total_zero/total_params}


# ── Save data ─────────────────────────────────────────────────────────────────

with open(os.path.join(EXP_DIR, 'analysis_data.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f"Saved analysis_data.json")


# ── Generate HTML report ─────────────────────────────────────────────────────

print("Generating HTML report...")

def embed_img(path):
    with open(path, 'rb') as f:
        return f'data:image/png;base64,{base64.b64encode(f.read()).decode()}'

plots = sorted([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])

html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>exp184 Model Analysis</title>
<style>
body{font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#333}
h1{border-bottom:2px solid #2c3e50;padding-bottom:10px}
h2{color:#2c3e50;margin-top:30px}
h3{color:#495057}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:8px 0}
table{border-collapse:collapse;margin:10px 0}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;font-size:0.85em}
th{background:#f5f5f5}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
</style></head><body>
<h1>exp184 Model Analysis</h1>
<p>6.51M params, best val_loss=1.4411 @ 100K steps</p>
"""

# Weight section
html += "<h2>1. Weight Distributions</h2>\n"
html += f'<img src="{embed_img(os.path.join(PLOT_DIR, "weights_all.png"))}">\n'
html += '<div class="grid">\n'
for i in range(6):
    fn = f'weights_layer{i}.png'
    if fn in plots:
        html += f'<img src="{embed_img(os.path.join(PLOT_DIR, fn))}">\n'
html += '</div>\n'

# Activation section
html += "<h2>2. Activation Distributions</h2>\n"
html += '<div class="grid">\n'
for i in range(6):
    fn = f'activations_layer{i}.png'
    if fn in plots:
        html += f'<img src="{embed_img(os.path.join(PLOT_DIR, fn))}">\n'
html += '</div>\n'

# Attention patterns
html += "<h2>3. Attention Patterns (averaged)</h2>\n"
html += '<div class="grid">\n'
for fn in plots:
    if fn.startswith('attn_layer'):
        html += f'<img src="{embed_img(os.path.join(PLOT_DIR, fn))}">\n'
html += '</div>\n'

# Attention entropy
html += "<h3>Attention Entropy</h3>\n<table><tr><th>Layer</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>\n"
for i in range(6):
    s = report['attention_entropy'][f'layer_{i}']
    html += f"<tr><td>{i}</td><td>{s['mean']:.3f}</td><td>{s['std']:.3f}</td><td>{s['min']:.3f}</td><td>{s['max']:.3f}</td></tr>\n"
html += "</table>\n"

# Gradient section
html += "<h2>4. Gradient Distributions</h2>\n"
html += '<div class="grid">\n'
for i in range(6):
    fn = f'gradients_layer{i}.png'
    if fn in plots:
        html += f'<img src="{embed_img(os.path.join(PLOT_DIR, fn))}">\n'
html += '</div>\n'

# Grad/weight ratio
html += "<h2>5. |Gradient/Weight| Ratio</h2>\n"
html += '<div class="grid">\n'
for i in range(6):
    fn = f'grad_ratio_layer{i}.png'
    if fn in plots:
        html += f'<img src="{embed_img(os.path.join(PLOT_DIR, fn))}">\n'
html += '</div>\n'

# Dead weights
html += "<h2>6. Dead Weights (zero gradients)</h2>\n"
html += "<table><tr><th>Component</th><th>Zero</th><th>Total</th><th>%</th></tr>\n"
for key, val in report['dead_weights'].items():
    html += f"<tr><td>{key}</td><td>{val['zero']:,}</td><td>{val['total']:,}</td><td>{val['pct']:.1%}</td></tr>\n"
html += "</table>\n"

html += f"<p>Training loss at analysis time: {report['loss']:.4f}</p>\n"
html += "</body></html>"

report_path = os.path.join(EXP_DIR, 'analysis_report.html')
with open(report_path, 'w') as f:
    f.write(html)
print(f"Saved {report_path}")
print("Done.")
