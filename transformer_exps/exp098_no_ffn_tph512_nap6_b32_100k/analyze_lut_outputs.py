"""
LUT output statistics per layer during inference.
For each LUT, collects: mean, std, min, max, sparsity, per-output-dim stats,
and inter-head variance (q/k/v have multiple heads).
"""
import sys, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

import torch.nn as nn
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention
from transformer_exps.common import CONTEXT_SIZE, VOCAB_SIZE

DEVICE = 'cuda:0'

def make_lut(input_dim, n_heads, n_outputs, tables_per_head, cfg, seed_offset=0):
    return MultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=cfg['n_anchor_pairs'], tables_per_head=tables_per_head,
        smooth_mode=cfg['smooth_mode'], n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'], calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
    )

class LUTBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d, p, h = cfg['embedding_dim'], cfg['positional_dim'], cfg['num_heads']
        d_qk, d_v = cfg['d_qk'], cfg['d_v']
        tph_qkv, tph_op = cfg['qkv_tables_per_head'], cfg['out_proj_tables_per_head']
        s = layer_idx * 10
        self.q_lut    = make_lut(d + p, h, d_qk, tph_qkv, cfg, s + 0)
        self.k_lut    = make_lut(d + p, h, d_qk, tph_qkv, cfg, s + 1)
        self.v_lut    = make_lut(d + p, h, d_v,  tph_qkv, cfg, s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,  tph_op,  cfg, s + 3)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=cfg['rank_attn_temperature'])
        self.n_heads, self.d_qk, self.d_v, self.d = h, d_qk, d_v, d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos_flat = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])
        q = self.q_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = self.k_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = self.v_lut(x_pos_flat).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)
        attn = self.rank_attn(q, k, v, is_causal=True).permute(0, 2, 1, 3).reshape(B * T, self.n_heads * self.d_v)
        return x + self.out_proj(attn)[:, 0, :].reshape(B, T, E)

class LUTTransformerV2(nn.Module):
    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d, p = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(cfg, i) for i in range(cfg['num_layers'])])
        self.unembedder = MultiHeadLut(
            input_dim=d, n_heads=1, n_outputs=cfg['vocab_size'],
            n_anchor_pairs=cfg['n_anchor_pairs'], tables_per_head=cfg['tables_per_head'],
            smooth_mode=cfg['smooth_mode'], n_alternatives=cfg['n_alternatives'],
            normalize_weights=False, calibrate_output=False,
            connected_anchors_mode=cfg['connected_anchors_mode'],
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            random_seed=cfg['random_seed'] + 999, device=DEVICE,
        )
    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x.reshape(-1, x.shape[-1]))[:, 0, :].reshape(B, T, -1)

model = LUTTransformerV2(cfg)
model.load_state_dict(torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── Hooks on MultiHeadLut modules ────────────────────────────────────────────
# MultiHeadLut output: [B, n_heads, n_outputs]
# We collect raw outputs across all forward calls and compute statistics at the end.

buffers = defaultdict(list)  # name -> list of tensors

hooks = []
for name, module in model.named_modules():
    if isinstance(module, MultiHeadLut):
        def make_hook(n):
            def hook(mod, inp, out):
                # out: [B, n_heads, n_outputs]
                buffers[n].append(out.detach().cpu())
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

print(f"Registered {len(hooks)} hooks on MultiHeadLut modules.")

# ── Generate text ─────────────────────────────────────────────────────────────
N_PROMPTS = 50
GEN_LEN   = 200
RAW_VOCAB  = 256
BOS_ID     = 256
CONTEXT_SIZE_ = CONTEXT_SIZE

prefixes = [
    "Once upon a time ", "The history of science ", "In the year 2024 ",
    "The best way to learn ", "Scientists have discovered ", "The city of London ",
    "A new study shows ", "The economy is ", "Technology has changed ",
    "Children learn best when ",
]

print(f"Generating {N_PROMPTS} sequences of length {GEN_LEN}...")
with torch.no_grad():
    for i in range(N_PROMPTS):
        prefix = prefixes[i % len(prefixes)]
        ctx = [BOS_ID] + list(prefix.encode('utf-8', errors='replace'))
        for _ in range(GEN_LEN):
            trunc = ctx[-(CONTEXT_SIZE_ - 1):]
            x = torch.zeros([1, CONTEXT_SIZE_], dtype=torch.long, device=DEVICE)
            x[0, -len(trunc):] = torch.tensor(trunc, dtype=torch.long, device=DEVICE)
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :RAW_VOCAB], dim=-1)[0]
            ctx.append(torch.multinomial(probs, 1).item())
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_PROMPTS} done")

for h in hooks:
    h.remove()

# ── Compute statistics ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("LUT OUTPUT STATISTICS  (across all tokens generated)")
print("  output shape: [B, n_heads, n_outputs] — stats computed over B dimension (per token)")
print("=" * 90)

lut_names = sorted(buffers.keys())

header = f"{'LUT':<35}  {'n_heads':>7}  {'n_out':>5}  {'mean':>8}  {'std':>8}  {'abs_mean':>9}  {'min':>8}  {'max':>8}  {'near0%':>7}"
print(header)

rows = []
for name in lut_names:
    # concatenate all collected outputs: list of [B, n_heads, n_out] -> [N, n_heads, n_out]
    all_out = torch.cat(buffers[name], dim=0).float()  # [N, n_heads, n_out]
    n_heads = all_out.shape[1]
    n_out   = all_out.shape[2]
    flat    = all_out.reshape(-1)
    mean    = flat.mean().item()
    std     = flat.std().item()
    abs_mean = flat.abs().mean().item()
    mn      = flat.min().item()
    mx      = flat.max().item()
    near0   = (flat.abs() < 0.01).float().mean().item() * 100

    # inter-head std: how much do heads differ from each other on average?
    if n_heads > 1:
        head_means = all_out.mean(dim=0)  # [n_heads, n_out]
        inter_head_std = head_means.std(dim=0).mean().item()
    else:
        inter_head_std = float('nan')

    print(f"{name:<35}  {n_heads:>7}  {n_out:>5}  {mean:>8.4f}  {std:>8.4f}  {abs_mean:>9.4f}  {mn:>8.4f}  {mx:>8.4f}  {near0:>6.1f}%")
    rows.append((name, n_heads, n_out, mean, std, abs_mean, mn, mx, near0, inter_head_std, all_out))

# ── Per-output-dim std (which output dims are most active?) ───────────────────
print("\n" + "=" * 90)
print("PER-OUTPUT-DIM STD  (std across tokens for each output dimension, averaged over heads)")
print("High = that output dim varies a lot = informative signal")
print("=" * 90)
print(f"{'LUT':<35}  {'mean_dim_std':>13}  {'min_dim_std':>12}  {'max_dim_std':>12}  {'dead_dims(<0.005)%':>18}  {'inter_head_std':>15}")
for name, n_heads, n_out, mean, std, abs_mean, mn, mx, near0, inter_head_std, all_out in rows:
    # all_out: [N, n_heads, n_out]
    dim_std = all_out.std(dim=0).mean(dim=0)  # [n_out] — std per output dim, averaged over heads
    print(f"{name:<35}  {dim_std.mean().item():>13.5f}  {dim_std.min().item():>12.5f}  {dim_std.max().item():>12.5f}  "
          f"{(dim_std < 0.005).float().mean().item()*100:>17.1f}%  {inter_head_std:>15.5f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
layer_order = [f'layers.{i}' for i in range(6)]
roles = ['q_lut', 'k_lut', 'v_lut', 'out_proj']
role_colors = {'q_lut': 'steelblue', 'k_lut': 'darkorange', 'v_lut': 'green', 'out_proj': 'red'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('exp098: LUT Output Statistics per Layer', fontsize=13)

# Plot 1: abs_mean per LUT per layer
ax = axes[0, 0]
for role in roles:
    vals = [next((r[5] for r in rows if r[0] == f'{l}.{role}'), None) for l in layer_order]
    vals = [v for v in vals if v is not None]
    ax.plot(range(len(vals)), vals, marker='o', label=role, color=role_colors[role])
ax.set_xticks(range(6)); ax.set_xticklabels([f'L{i}' for i in range(6)])
ax.set_title('Mean |output| by layer'); ax.set_ylabel('abs mean'); ax.legend(fontsize=8)

# Plot 2: std per LUT per layer
ax = axes[0, 1]
for role in roles:
    vals = [next((r[4] for r in rows if r[0] == f'{l}.{role}'), None) for l in layer_order]
    vals = [v for v in vals if v is not None]
    ax.plot(range(len(vals)), vals, marker='o', label=role, color=role_colors[role])
ax.set_xticks(range(6)); ax.set_xticklabels([f'L{i}' for i in range(6)])
ax.set_title('Output std by layer'); ax.set_ylabel('std'); ax.legend(fontsize=8)

# Plot 3: % near-zero outputs
ax = axes[0, 2]
for role in roles:
    vals = [next((r[8] for r in rows if r[0] == f'{l}.{role}'), None) for l in layer_order]
    vals = [v for v in vals if v is not None]
    ax.plot(range(len(vals)), vals, marker='o', label=role, color=role_colors[role])
ax.set_xticks(range(6)); ax.set_xticklabels([f'L{i}' for i in range(6)])
ax.set_title('% outputs near zero (|x|<0.01)'); ax.set_ylabel('%'); ax.legend(fontsize=8)

# Plot 4: per-dim std heatmap (layers x output dims) for q_lut
ax = axes[1, 0]
mat = []
for i in range(6):
    row = next((r for r in rows if r[0] == f'layers.{i}.q_lut'), None)
    if row:
        dim_std = row[10].std(dim=0).mean(dim=0).numpy()  # [n_out]
        mat.append(dim_std)
if mat:
    im = ax.imshow(np.array(mat), aspect='auto', cmap='hot')
    ax.set_yticks(range(6)); ax.set_yticklabels([f'L{i}' for i in range(6)])
    ax.set_xlabel('output dim'); ax.set_title('q_lut: per-dim std per layer')
    plt.colorbar(im, ax=ax)

# Plot 5: output distribution histograms (one per layer, out_proj)
ax = axes[1, 1]
for i in range(6):
    row = next((r for r in rows if r[0] == f'layers.{i}.out_proj'), None)
    if row:
        vals = row[10].reshape(-1).numpy()
        ax.hist(vals, bins=80, alpha=0.5, label=f'L{i}', density=True)
ax.set_title('out_proj output distribution per layer')
ax.set_xlabel('value'); ax.set_ylabel('density'); ax.legend(fontsize=7)

# Plot 6: inter-head std per layer (q/k/v)
ax = axes[1, 2]
for role in ['q_lut', 'k_lut', 'v_lut']:
    vals = [next((r[9] for r in rows if r[0] == f'{l}.{role}'), None) for l in layer_order]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    ax.plot(range(len(vals)), vals, marker='o', label=role, color=role_colors[role])
ax.set_xticks(range(6)); ax.set_xticklabels([f'L{i}' for i in range(6)])
ax.set_title('Inter-head std (how much heads differ)'); ax.set_ylabel('std'); ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(EXP_DIR, 'lut_output_stats.png')
plt.savefig(out_path, dpi=100, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
