"""
Analyse attention patterns in exp208: extract Q/K projections,
compute attention scores manually, visualize patterns.
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(EXP_DIR, 'analysis_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
P = cfg['positional_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']


def _make_lut(input_dim, n_heads, n_outputs, nap, tph, seed_offset):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(E+P, H, d_qk, cfg['qk_nap'], cfg['qk_tph'], layer_idx)
        self.k_lut = _make_lut(E+P, H, d_qk, cfg['qk_nap'], cfg['qk_tph'], 100+layer_idx)
        self.v_lut = _make_lut(E, H, d_v, cfg['v_nap'], cfg['v_tph'], 200+layer_idx)
        self.out_proj = _make_lut(H*d_v, 1, E, cfg['outproj_nap'], cfg['outproj_tph'], 400+layer_idx)
        self.norm1 = nn.LayerNorm(E)
        self.ffn = _make_lut(E, 1, E, cfg['ffn_nap'], cfg['ffn_tph'], 600+layer_idx)
        self.norm2 = nn.LayerNorm(E)
        # Store attention data
        self._attn_data = {}

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = torch.cat([x, pos_emb.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        xp_flat = xp.reshape(B*T, E+P)

        q = self.q_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v = self.v_lut(x.reshape(B*T, _E)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        # Compute scores manually to capture them
        scale = d_qk ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        self._attn_data['q'] = q.detach()
        self._attn_data['k'] = k.detach()
        self._attn_data['scores_pre_softmax'] = scores.detach()
        self._attn_data['attn_weights'] = attn_weights.detach()

        attn_out = attn_weights @ v
        proj = self.out_proj(attn_out.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm1(proj)
        ffn_out = self.ffn(x.reshape(B*T, _E)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm2(ffn_out)
        return x


class LUTRankAttnV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        n_layers = cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, cfg['vocab_size'], bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


# Load model
model = LUTRankAttnV2().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print("Loaded checkpoint")

# Run on test batch
sampler = make_sampler(DEVICE, random_seed=1)
batch = next(iter(sampler.testing_batches_iterator(128)))
inp = torch.empty(128, 32, dtype=torch.long, device=DEVICE)
inp[:, 0] = BOS_ID
inp[:, 1:] = batch[:, :-1].long()

with torch.no_grad():
    logits = model(inp)

print("\n=== ATTENTION ANALYSIS ===\n")

for i, layer in enumerate(model.layers):
    d = layer._attn_data
    q, k = d['q'], d['k']
    scores = d['scores_pre_softmax']
    weights = d['attn_weights']

    # Q/K statistics
    q_norm = q.norm(dim=-1)  # [B, H, T]
    k_norm = k.norm(dim=-1)
    print(f"Layer {i}:")
    print(f"  Q norm: mean={q_norm.mean():.3f} std={q_norm.std():.3f}")
    print(f"  K norm: mean={k_norm.mean():.3f} std={k_norm.std():.3f}")

    # Score statistics (finite only)
    finite_scores = scores[scores.isfinite()]
    print(f"  Scores (pre-softmax): mean={finite_scores.mean():.3f} std={finite_scores.std():.3f} "
          f"min={finite_scores.min():.3f} max={finite_scores.max():.3f}")

    # Attention entropy
    ent = -(weights * (weights + 1e-10).log()).sum(-1)  # [B, H, T]
    print(f"  Attn entropy: mean={ent.mean():.3f} std={ent.std():.3f}")

    # Self-attention (diagonal)
    diag = torch.diagonal(weights, dim1=-2, dim2=-1)  # [B, H, T]
    print(f"  Self-attn (diagonal): mean={diag.mean():.3f}")

    # How many tokens get >10% attention
    big_weights = (weights > 0.1).float().sum(-1)  # [B, H, T]
    print(f"  Tokens with >10% attn: mean={big_weights.mean():.1f}")
    print()

    # Plot attention patterns (averaged over batch)
    avg_weights = weights.mean(0)  # [H, T, T]
    fig, axes = plt.subplots(1, H, figsize=(4*H, 3.5))
    for h in range(H):
        axes[h].imshow(avg_weights[h].cpu().numpy(), aspect='auto', cmap='viridis', vmin=0)
        axes[h].set_title(f'Head {h}')
        axes[h].set_xlabel('key')
        if h == 0:
            axes[h].set_ylabel('query')
    fig.suptitle(f'Layer {i} Attention (avg batch)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'attn_layer{i}.png'), dpi=100)
    plt.close()

    # Plot score distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(finite_scores.cpu().numpy(), bins=100, density=True, alpha=0.7)
    ax.set_title(f'Layer {i} Pre-softmax Score Distribution')
    ax.set_xlabel('score')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'scores_layer{i}.png'), dpi=100)
    plt.close()

    # Plot Q/K norms per position
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    for h in range(H):
        ax1.plot(q_norm[:, h, :].mean(0).cpu().numpy(), label=f'head {h}', alpha=0.7)
        ax2.plot(k_norm[:, h, :].mean(0).cpu().numpy(), label=f'head {h}', alpha=0.7)
    ax1.set_title(f'Layer {i} Q norms by position')
    ax2.set_title(f'Layer {i} K norms by position')
    ax1.legend(); ax2.legend()
    ax1.set_xlabel('position'); ax2.set_xlabel('position')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'qk_norms_layer{i}.png'), dpi=100)
    plt.close()

# Generate HTML report
print("\nGenerating HTML report...")

def embed_img(path):
    with open(path, 'rb') as f:
        return f'data:image/png;base64,{base64.b64encode(f.read()).decode()}'

html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>exp208 Attention Analysis</title>
<style>
body{font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 20px}
h1{border-bottom:2px solid #2c3e50;padding-bottom:10px}
h2{color:#2c3e50;margin-top:30px}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:8px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
</style></head><body>
<h1>exp208 Attention Analysis</h1>
<p>8.3M params, val_loss=1.4256. Per-layer pos embeddings, pos_dim=32.</p>
"""

for i in range(6):
    html += f'<h2>Layer {i}</h2>\n'
    html += f'<img src="{embed_img(os.path.join(PLOT_DIR, f"attn_layer{i}.png"))}">\n'
    html += '<div class="grid">\n'
    html += f'<img src="{embed_img(os.path.join(PLOT_DIR, f"scores_layer{i}.png"))}">\n'
    html += f'<img src="{embed_img(os.path.join(PLOT_DIR, f"qk_norms_layer{i}.png"))}">\n'
    html += '</div>\n'

html += "</body></html>"

with open(os.path.join(EXP_DIR, 'attention_report.html'), 'w') as f:
    f.write(html)

print("Done. Report saved.")
