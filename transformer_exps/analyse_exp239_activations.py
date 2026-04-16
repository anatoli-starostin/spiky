"""
Analyse exp239: table activation patterns on real data.
For each LUT, run data through and measure:
  - How often each table entry is selected (entry utilization)
  - Per-table output variance on real inputs (functional diversity)
  - Correlation between tables (redundancy)
  - Per-layer activation magnitude distributions
"""
import sys, os, json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer_exps.common import make_sampler, BOS_ID, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut, _compute_anchor_data
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy

DEVICE = 'cuda:0'
EXP_DIR = 'transformer_exps/exp239_no_ffn_nap6_tph2048'

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
NAP_QK = cfg.get('nap_qk', 5)
NAP_V = cfg.get('nap_v', 6)
NAP_OUT = cfg.get('nap', 6)
TPH = cfg['tph']
TPH_OUT = cfg.get('tph_out', TPH)

torch.manual_seed(cfg['random_seed'])


def _make_lut(n_heads, n_outputs, nap, tph, seed_offset):
    return MultiHeadLut(
        input_dim=E, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        smooth_mode=False, n_alternatives=1,
        normalize_weights=False, calibrate_output=False,
        anchor_sampling_policy=AnchorSamplingPolicy.FULL_COVERAGE,
        initial_weights_noise=0.001, uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed']+seed_offset, device=DEVICE, recompute_in_backward=True,
    )


# Rebuild model
import torch.nn as nn

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_lut = _make_lut(H, d_qk, NAP_QK, TPH, layer_idx)
        self.k_lut = _make_lut(H, d_qk, NAP_QK, TPH, 100+layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.v_lut = _make_lut(H, d_v, NAP_V, TPH, 200+layer_idx)
        self.out_proj = _make_lut(1, E, NAP_OUT, TPH_OUT, 400+layer_idx)
        self.norm1 = nn.LayerNorm(E)

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = x + pos_emb.unsqueeze(0)
        xp_flat = xp.reshape(B*T, _E)
        x_flat = x.reshape(B*T, _E)
        q = self.q_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_lut(xp_flat).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_lut(x_flat).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        proj = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)).squeeze(1).reshape(B, T, _E)
        x = x + self.norm1(proj)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        n_layers = cfg['num_layers']
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(n_layers)])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(n_layers)])
        self.unembedder = nn.Sequential(
            nn.Linear(E, 128), nn.GELU(), nn.Linear(128, cfg['vocab_size'], bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return self.unembedder(x)


# Load checkpoint
model = Model().to(DEVICE)
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()
print('Model loaded.')

# Prepare data
sampler = make_sampler(DEVICE, random_seed=1)
N_BATCHES = 20
BATCH_SIZE = 128

# Hook to capture LUT inputs and lookup indices
activation_data = {}

def make_hook(name, lut):
    """Hook that captures lookup indices for a LUT."""
    def hook_fn(module, input, output):
        x = input[0]  # [B*T, input_dim]
        with torch.no_grad():
            lookup_indices, _, _, _, _ = _compute_anchor_data(
                x, lut.lookup.anchor_pairs_a, lut.lookup.anchor_pairs_b,
                lut.lookup.powers, lut.lookup.cmp_eps, 1
            )
            # lookup_indices: [B*T, n_tables]
            if name not in activation_data:
                activation_data[name] = {
                    'entry_counts': torch.zeros(lookup_indices.shape[1], lut.lookup.powers.shape[-1] > 0 and 2**lut.lookup.anchor_pairs_a.shape[1] or 32, device='cpu', dtype=torch.long),
                    'output_samples': [],
                    'n_tables': lookup_indices.shape[1],
                    'n_entries': 2**lut.lookup.anchor_pairs_a.shape[1],
                }
            data = activation_data[name]
            # Count entry usage per table
            indices_cpu = lookup_indices.cpu()
            for t in range(data['n_tables']):
                counts = torch.bincount(indices_cpu[:, t], minlength=data['n_entries'])
                data['entry_counts'][t] += counts

            # Store per-table output for correlation analysis (subsample)
            # output shape: [B*T, n_tables, n_outputs] or [B*T, n_heads, n_outputs]
            # For non-smooth: weights[table_indices, lookup_indices] before sum
            weights = lut.projection.weights  # [n_tables, n_entries, n_outputs]
            table_indices = torch.arange(data['n_tables'], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            per_table_out = weights[table_indices, lookup_indices]  # [B*T, n_tables, n_outputs]
            # Subsample to save memory
            if len(data['output_samples']) < 5:
                data['output_samples'].append(per_table_out[:256].cpu())  # first 256 tokens
    return hook_fn


# Register hooks
hooks = []
for layer_idx in range(cfg['num_layers']):
    block = model.layers[layer_idx]
    for comp_name, lut in [('q_lut', block.q_lut), ('k_lut', block.k_lut),
                            ('v_lut', block.v_lut), ('out_proj', block.out_proj)]:
        name = f'L{layer_idx}.{comp_name}'
        h = lut.register_forward_hook(make_hook(name, lut))
        hooks.append(h)

# Run data through model
print(f'Running {N_BATCHES} batches of {BATCH_SIZE}...')
with torch.no_grad():
    for i in range(N_BATCHES):
        x = sampler.sample_training_batch(BATCH_SIZE).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        model(inp)
        if (i + 1) % 5 == 0:
            print(f'  batch {i+1}/{N_BATCHES}')

# Remove hooks
for h in hooks:
    h.remove()

# Analyse
print()
print('=' * 130)
print(f'{"Name":>15} | {"EntryUtil":>9} {"Entropy":>8} {"MaxEntry%":>9} | '
      f'{"OutStd":>7} {"TabCorr":>8} {"EffRank":>8} | {"TopTab10%":>10} {"TabGini":>8}')
print('=' * 130)

for name in sorted(activation_data.keys()):
    data = activation_data[name]
    counts = data['entry_counts'].float()  # [n_tables, n_entries]
    n_tables = data['n_tables']
    n_entries = data['n_entries']

    # Entry utilization: what fraction of entries are used at all?
    used = (counts > 0).float().mean().item() * 100

    # Entry entropy per table (normalized)
    probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    log_probs = torch.log2(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=1)  # [n_tables]
    max_entropy = np.log2(n_entries)
    norm_entropy = (entropy / max_entropy).mean().item()

    # Max entry fraction: most popular entry per table
    max_entry_frac = (counts.max(dim=1).values / counts.sum(dim=1).clamp(min=1)).mean().item() * 100

    # Per-table output analysis
    if data['output_samples']:
        out = torch.cat(data['output_samples'], dim=0)  # [N, n_tables, n_outputs]
        N = out.shape[0]

        # Output std per table, then mean
        out_std = out.std(dim=0).mean().item()  # mean std across tables and outputs

        # Table correlation: average pairwise correlation between tables
        # Flatten each table's output: [N, n_outputs] per table
        # Use mean output per table as representative
        table_means = out.mean(dim=2)  # [N, n_tables]
        if n_tables <= 2048:
            # Correlation matrix
            tm = table_means - table_means.mean(dim=0, keepdim=True)
            norms = tm.norm(dim=0, keepdim=True).clamp(min=1e-8)
            tm_normed = tm / norms
            corr = (tm_normed.T @ tm_normed) / N  # [n_tables, n_tables]
            # Mean off-diagonal correlation
            mask = ~torch.eye(n_tables, dtype=torch.bool)
            mean_corr = corr[mask].mean().item()
        else:
            # Subsample for large n_tables
            idx = torch.randperm(n_tables)[:512]
            tm_sub = table_means[:, idx]
            tm_sub = tm_sub - tm_sub.mean(dim=0, keepdim=True)
            norms = tm_sub.norm(dim=0, keepdim=True).clamp(min=1e-8)
            tm_normed = tm_sub / norms
            corr = (tm_normed.T @ tm_normed) / N
            mask = ~torch.eye(512, dtype=torch.bool)
            mean_corr = corr[mask].mean().item()

        # Effective rank: from singular values of table outputs
        # [N, n_tables] matrix of table contributions
        svd_input = table_means[:min(N, 1024)]
        try:
            s = torch.linalg.svdvals(svd_input.float())
            p = s / s.sum()
            eff_rank = torch.exp(-(p * torch.log(p.clamp(min=1e-10))).sum()).item()
        except:
            eff_rank = float('nan')

        # Table contribution inequality (on actual outputs)
        table_norms = out.std(dim=0).norm(dim=1)  # [n_tables]
        sorted_norms, _ = table_norms.sort(descending=True)
        cumsum = sorted_norms.cumsum(0) / sorted_norms.sum()
        top10_idx = max(1, n_tables // 10)
        top10 = cumsum[top10_idx - 1].item() * 100

        n = len(sorted_norms)
        sn_np = sorted_norms.flip(0).numpy()
        cs = np.cumsum(sn_np)
        gini = (n + 1 - 2 * np.sum(cs) / cs[-1]) / n
    else:
        out_std = mean_corr = eff_rank = top10 = gini = float('nan')

    print(f'{name:>15} | {used:>8.1f}% {norm_entropy:>8.3f} {max_entry_frac:>8.1f}% | '
          f'{out_std:>7.4f} {mean_corr:>8.4f} {eff_rank:>8.1f} | {top10:>9.1f}% {gini:>8.3f}')

# Layer trend for out_proj
print()
print('=== OUT_PROJ activation trend across layers ===')
print(f'{"Layer":>5} | {"EntryUtil":>9} {"Entropy":>8} | {"OutStd":>7} {"TabCorr":>8} {"EffRank":>8}')
print('-' * 60)
for layer_idx in range(cfg['num_layers']):
    name = f'L{layer_idx}.out_proj'
    if name not in activation_data:
        continue
    data = activation_data[name]
    counts = data['entry_counts'].float()
    n_entries = data['n_entries']
    probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    log_probs = torch.log2(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=1)
    norm_entropy = (entropy / np.log2(n_entries)).mean().item()
    used = (counts > 0).float().mean().item() * 100

    if data['output_samples']:
        out = torch.cat(data['output_samples'], dim=0)
        out_std = out.std(dim=0).mean().item()
        table_means = out.mean(dim=2)
        N = table_means.shape[0]
        idx = torch.randperm(data['n_tables'])[:512]
        tm_sub = table_means[:, idx]
        tm_sub = tm_sub - tm_sub.mean(dim=0, keepdim=True)
        norms = tm_sub.norm(dim=0, keepdim=True).clamp(min=1e-8)
        tm_normed = tm_sub / norms
        corr = (tm_normed.T @ tm_normed) / N
        mask = ~torch.eye(512, dtype=torch.bool)
        mean_corr = corr[mask].mean().item()
        svd_input = table_means[:min(N, 1024)]
        s = torch.linalg.svdvals(svd_input.float())
        p = s / s.sum()
        eff_rank = torch.exp(-(p * torch.log(p.clamp(min=1e-10))).sum()).item()
    else:
        out_std = mean_corr = eff_rank = float('nan')

    print(f'{layer_idx:>5} | {used:>8.1f}% {norm_entropy:>8.3f} | {out_std:>7.4f} {mean_corr:>8.4f} {eff_rank:>8.1f}')
