"""Yuval's diagnostics for exp265: visit-frequency + low-rank analysis.

(1) Per-LUT empirical visit-frequency histogram on a validation batch.
    Reports normalised entropy (1 = uniform), unvisited-fraction,
    top-10%-mass fraction.

(2) Per-LUT SVD of the entry matrix (weights). Reports effective rank to
    capture 90% / 99% of Frobenius norm, plus top-k mass for small k.

Both are run deterministically (noise OFF on lookup_indices), since the
goal is to measure what the model actually routes to, not what noise
spreads it over.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']

_POS_EMB_CFG = cfg.get('pos_emb_dim', 0)
_POS_EMB_ACTIVE = isinstance(_POS_EMB_CFG, int) and _POS_EMB_CFG > 0
def _pos_emb_dim(layer_idx):
    return _POS_EMB_CFG if _POS_EMB_ACTIVE else E

print(f'Loading tokenizer ...')
tokenizer = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()

# ---- LUT factories matching exp265 ----
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=0.0,  # OFF for analysis
)
_TINY_MULTIALT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='ste',
    n_alternatives=3,
    argmax_noise_eps=0.0,  # OFF for analysis
    learnable_temps=cfg.get('multialt_learnable_temps', False),
    uncertainty_T_init=cfg.get('uncertainty_T_init', 1.0),
)

def _make_qk_joint(layer_idx, seed_offset):
    n_inputs = E + (_pos_emb_dim(layer_idx) if _POS_EMB_ACTIVE else 0)
    qk_kwargs = dict(_TINY_SOFT_KWARGS)
    qk_kwargs['initial_weights_noise'] = cfg.get('qk_lut_init_std',
                                                 cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=n_inputs, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **qk_kwargs,
    )

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_MULTIALT_KWARGS,
    )

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
def _make_out(layer_idx, seed_offset):
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']
    return TinyMultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=tph,
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

# ---- Model ----
class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint(layer_idx, layer_idx)
        self.v_lut    = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, 400 + layer_idx)
        canon_t = cfg.get('canon_temperature', 0.1)
        v2d_init_t = cfg.get('out_v2d_temperature_init', canon_t)
        v2d_learnable = cfg.get('out_v2d_learnable_temperature', False)
        self.out_v2d = VectorToDominance(E, smooth_mode=False,
                                          temperature=v2d_init_t,
                                          learnable_temperature=v2d_learnable)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.pos_dim = _pos_emb_dim(layer_idx)
        if _POS_EMB_ACTIVE:
            self.qk_input_ln = nn.LayerNorm(E + self.pos_dim)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        if _POS_EMB_ACTIVE:
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1)
            xp = torch.cat([x, pos], dim=-1)
            xp = self.qk_input_ln(xp).reshape(B * T, E + self.pos_dim)
        else:
            xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)
        qk_out = self.qk_joint(xp)
        q_vec = qk_out[..., :d_qk]
        k_vec = qk_out[..., d_qk:]
        q_vec = self.q_norm(q_vec); k_vec = self.k_norm(k_vec)
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        v_vec = self.v_lut(x_flat)
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        out_dom = self.out_v2d(out_real)
        out_rank = self.out_d2v(out_dom).reshape(B, T, E)
        return out_rank

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        _pos_dim_fn = _pos_emb_dim if _POS_EMB_ACTIVE else (lambda i: E)
        _pos_init_scale = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, _pos_dim_fn(i)) * _pos_init_scale)
            for i in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        unembed_hidden = cfg.get('unembed_hidden', concat_dim * 8)
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, unembed_hidden, bias=False),
            nn.GELU(),
            nn.Linear(unembed_hidden, VOCAB_SIZE, bias=False),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
        return x

print('Building model + loading checkpoint ...')
model = Model().to(DEVICE)
sd = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE)
model.load_state_dict(sd, strict=True)
model.eval()
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

# ---- Collect LUT modules ----
LUT_NAMES = ['qk_joint', 'v_lut', 'out_proj']
luts = {}  # name -> (module, mode_label)
for li, blk in enumerate(model.layers):
    for lut_name in LUT_NAMES:
        mod = getattr(blk, lut_name)
        luts[f'L{li}.{lut_name}'] = mod

# ---- Forward pre-hook: capture LUT inputs ----
captured = {}  # name -> list of input tensors
def make_hook(name):
    def hook(module, args):
        x = args[0]
        captured.setdefault(name, []).append(x.detach())
    return hook

handles = []
for name, mod in luts.items():
    h = mod.register_forward_pre_hook(make_hook(name))
    handles.append(h)

# ---- Run validation forward ----
N_BATCHES = 64  # 64 * 8 * 512 = 262,144 tokens; plenty for stable stats
val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE,
)
print(f'Running {N_BATCHES} validation batches ({N_BATCHES * DEVICE_BS} sequences = '
      f'{N_BATCHES * DEVICE_BS * CONTEXT_SIZE:,} tokens) ...')
with torch.no_grad():
    for b in range(N_BATCHES):
        x_in, _ = next(val_loader)
        model(x_in)

for h in handles:
    h.remove()


def compute_lookup_indices(module, x):
    """Deterministic lookup_indices for any TinyMHL module (matches its
    internal bit-pack convention; ignores noise)."""
    anchor_a = module.lookup.anchor_pairs_a.long()
    anchor_b = module.lookup.anchor_pairs_b.long()
    B = x.shape[0]
    n_tables = anchor_a.shape[0]
    NAP = anchor_a.shape[1]
    idx_a = anchor_a.reshape(1, -1).expand(B, -1)
    idx_b = anchor_b.reshape(1, -1).expand(B, -1)
    x_a = x.gather(1, idx_a).view(B, n_tables, NAP)
    x_b = x.gather(1, idx_b).view(B, n_tables, NAP)
    bits = (x_a > x_b).to(torch.int64)
    if module.backward_mode == 'soft':
        # MSB-first packing (matches `_soft_lut_fwd_body`).
        powers = (1 << torch.arange(NAP - 1, -1, -1, device=x.device, dtype=torch.int64))
    else:
        # LSB-first packing (matches `_multi_alt_fwd_body`).
        powers = (1 << torch.arange(NAP, device=x.device, dtype=torch.int64))
    return (bits * powers).sum(dim=-1)  # [B, n_tables]


print('\n=== DIAGNOSTIC 1: visit-frequency histogram ===\n', flush=True)
visit_stats = {}
for name, mod in luts.items():
    if name not in captured:
        continue
    inputs = torch.cat(captured[name], dim=0)
    table_dim = mod.table_dim
    n_tables = mod.n_heads * mod.tables_per_head
    counts_flat = torch.zeros(n_tables * table_dim, dtype=torch.int64, device=DEVICE)
    BATCH = 4096
    table_offset = (torch.arange(n_tables, device=DEVICE, dtype=torch.int64) * table_dim).view(1, -1)
    for i in range(0, inputs.shape[0], BATCH):
        chunk = inputs[i:i+BATCH]
        idx = compute_lookup_indices(mod, chunk)  # [B, n_tables]
        flat_idx = (table_offset + idx).reshape(-1)
        counts_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
    counts = counts_flat.view(n_tables, table_dim)
    counts_cpu = counts.cpu().numpy()  # [n_tables, table_dim]
    print(f'  {name} counts done', flush=True)
    total_per_table = counts_cpu.sum(axis=1, keepdims=True)
    p = counts_cpu / np.maximum(total_per_table, 1)
    # Per-table normalised entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.where(p > 0, p * np.log(p), 0).sum(axis=1)
    norm_ent = ent / np.log(table_dim)
    unvisited = (counts_cpu == 0).sum(axis=1) / table_dim
    sorted_p = -np.sort(-p, axis=1)
    top10_count = max(1, table_dim // 10)
    top10_mass = sorted_p[:, :top10_count].sum(axis=1)
    visit_stats[name] = {
        'mode': mod.backward_mode,
        'table_dim': table_dim,
        'n_tables': n_tables,
        'norm_entropy_mean': float(norm_ent.mean()),
        'norm_entropy_min': float(norm_ent.min()),
        'norm_entropy_max': float(norm_ent.max()),
        'unvisited_frac_mean': float(unvisited.mean()),
        'unvisited_frac_max': float(unvisited.max()),
        'top10_mass_mean': float(top10_mass.mean()),
        'top10_mass_max': float(top10_mass.max()),
    }
    print(f'{name:18s} (mode={mod.backward_mode:5s}, K={table_dim}, T={n_tables}): '
          f'norm_H={norm_ent.mean():.3f} (min {norm_ent.min():.3f}), '
          f'unvisited={unvisited.mean()*100:.1f}% (max {unvisited.max()*100:.1f}%), '
          f'top-10%={top10_mass.mean()*100:.1f}% (max {top10_mass.max()*100:.1f}%)')


print('\n=== DIAGNOSTIC 2: SVD of per-table entry matrix ===\n')
svd_stats = {}
for name, mod in luts.items():
    weights = mod.weights.detach().to(torch.float32).cpu()  # [n_lookup_tables, K, n_out]
    n_tables, K, n_out = weights.shape
    full_rank = min(K, n_out)
    sing = []
    rank_90 = []
    rank_99 = []
    top4_mass = []
    for t in range(n_tables):
        W = weights[t]
        s = torch.linalg.svdvals(W).numpy()
        sing.append(s)
        cum = np.cumsum(s ** 2)
        total = cum[-1] if cum[-1] > 0 else 1.0
        r90 = int(np.searchsorted(cum / total, 0.90) + 1)
        r99 = int(np.searchsorted(cum / total, 0.99) + 1)
        rank_90.append(r90)
        rank_99.append(r99)
        top4 = min(4, full_rank)
        top4_mass.append(float(cum[top4 - 1] / total))
    rank_90 = np.array(rank_90); rank_99 = np.array(rank_99); top4_mass = np.array(top4_mass)
    svd_stats[name] = {
        'mode': mod.backward_mode,
        'table_dim': K,
        'n_out': n_out,
        'full_rank': full_rank,
        'n_tables': n_tables,
        'rank_90_mean': float(rank_90.mean()),
        'rank_99_mean': float(rank_99.mean()),
        'top4_mass_mean': float(top4_mass.mean()),
    }
    print(f'{name:18s} (mode={mod.backward_mode:5s}, K={K}, n_out={n_out}, full_rank={full_rank}): '
          f'rank@90%={rank_90.mean():.1f} ({rank_90.mean()/full_rank*100:.0f}% of full), '
          f'rank@99%={rank_99.mean():.1f} ({rank_99.mean()/full_rank*100:.0f}% of full), '
          f'top-4 mass={top4_mass.mean()*100:.1f}%')


print('\n=== SUMMARY ===\n')
print(f'{"LUT":18s} | {"K":>4s} | {"util_H":>6s} | {"unvis":>5s} | {"top10%":>6s} | '
      f'{"r@90%":>6s}/{"r@99%":>6s} (of {"full":>4s}) | {"top4":>5s}')
print('-' * 100)
for name in visit_stats:
    v = visit_stats[name]
    s = svd_stats[name]
    print(f'{name:18s} | {v["table_dim"]:>4d} | {v["norm_entropy_mean"]:>6.3f} | '
          f'{v["unvisited_frac_mean"]*100:>4.1f}% | {v["top10_mass_mean"]*100:>5.1f}% | '
          f'{s["rank_90_mean"]:>6.1f}/{s["rank_99_mean"]:>6.1f} (of {s["full_rank"]:>4d}) | '
          f'{s["top4_mass_mean"]*100:>4.1f}%')

# Save JSON for downstream
out_path = os.path.join(EXP_DIR, 'analysis.json')
with open(out_path, 'w') as f:
    json.dump({'visit_stats': visit_stats, 'svd_stats': svd_stats}, f, indent=2)
print(f'\nWrote {out_path}')
