"""Analyze attention entropy in exp144 (converged checkpoint).

Loads model, runs a batch, captures attention probabilities A from each
LUTBlock (replacing F.scaled_dot_product_attention with explicit math so
A is observable), summarizes the per-row entropy distribution.

Per-row entropy = -sum(A * log(A)). Bounded: 0 (one-hot) to log(T_eff).
Normalized: H_norm = entropy / log(T_eff) ∈ [0, 1].
  H_norm = 0   → one-hot (attends to exactly one token)
  H_norm ≈ 0.5 → middle
  H_norm = 1   → uniform over T_eff

Run from repo root:
    PYTHONPATH=/home/starost/nanochat .venv/bin/python \
        nanochat_exps/exp144_qk_tph384/analyze_attn.py
"""
import sys, os, json, math
import torch
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

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])
import torch.nn as nn

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']; H = cfg['n_heads']
d_qk = cfg['d_qk']; d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()

_TINY_KWARGS = dict(weight_dtype=torch.float32,
                    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                    initial_weights_noise=cfg.get('mhlut_init_std', 0.001))

def _make_qk_joint(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2*d_qk,
                            n_anchor_pairs=cfg['qk_input_nap'],
                            tables_per_head=cfg['qk_tph'],
                            random_seed=cfg['random_seed']+seed_offset,
                            device=DEVICE, **_TINY_KWARGS)

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                            n_anchor_pairs=cfg['v_input_nap'],
                            tables_per_head=cfg['v_tph'],
                            random_seed=cfg['random_seed']+seed_offset,
                            device=DEVICE, **_TINY_KWARGS)

def _make_out(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                            n_anchor_pairs=cfg['out_input_nap'],
                            tables_per_head=cfg['out_tph'],
                            random_seed=cfg['random_seed']+seed_offset,
                            device=DEVICE, **_TINY_KWARGS)

# Capture attention probabilities A here (per layer, per call).
ATTN_PROBS = []  # list of [B, H, T, T] tensors

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.qk_joint = _make_qk_joint(layer_idx, layer_idx)
        self.v_lut = _make_v(layer_idx, 200+layer_idx)
        self.out_proj = _make_out(layer_idx, 400+layer_idx)
        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.out_v2d = VectorToDominance(E, smooth_mode=False, temperature=canon_t)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, E)
        qk_out = self.qk_joint(xp)
        q_vec = qk_out[..., :d_qk]; k_vec = qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec); k_dom = self.qk_v2d(k_vec)
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v_vec = self.v_lut(x.reshape(B*T, E))
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        # Explicit attention so we can capture A.
        ramp = float(cfg.get('attn_scale_target_ramp', 1.0))
        s = torch.matmul(q * (self.attn_scale * ramp), k.transpose(-2, -1)) / math.sqrt(D_QK_P)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(causal_mask, float('-inf'))
        a = torch.softmax(s, dim=-1)
        a = torch.nan_to_num(a, nan=0.0)
        ATTN_PROBS.append(a.detach().cpu())
        attn = torch.matmul(a, v)
        out_in = attn.permute(0, 2, 1, 3).reshape(B*T, H*d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        out_dom = self.out_v2d(out_real)
        out_rank = self.out_d2v(out_dom).reshape(B, T, E)
        return out_rank

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        _pos_init_scale = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, E) * _pos_init_scale)
            for _ in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, VOCAB_SIZE),
        )
    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos in zip(self.layers, self.pos_embs):
            x = layer(x, pos); outs.append(x)
        return self.unembedder(torch.cat(outs, dim=-1))

print('Building model...')
model = Model().to(DEVICE)
sd = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE)
model.load_state_dict(sd)
model.eval()

print('Running a batch...')
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, cfg['device_batch_size'], CONTEXT_SIZE, split='val', device=DEVICE
)
x, _ = next(loader)
with torch.no_grad():
    _ = model(x)

print(f'Captured attention from {len(ATTN_PROBS)} layers')

# Per-row entropy. A: [B, H, T, T] (causal — row i has i+1 valid keys).
def row_entropy(A):
    """Returns entropy and effective T per row.

    For a causal row i, only positions 0..i have non-zero probs (mass=1).
    Normalized entropy = entropy / log(i+1) ∈ [0, 1].
    Entropy = 0 → one-hot, 1 → uniform over i+1 keys.
    """
    B, H, T, _ = A.shape
    # Mask of valid (non-zero) positions per row
    # log(0) → -inf; mask before sum
    A_clamp = A.clamp_min(1e-12)
    ent = -(A * A_clamp.log()).sum(-1)  # [B, H, T]
    # Normalize by log(i+1) — row i has i+1 valid keys
    i = torch.arange(T, device=ent.device)
    norm = torch.log(i + 1.0).clamp_min(1e-12)
    ent_norm = ent / norm  # broadcasts over B, H
    # Mask out row 0 (single key, entropy is 0 trivially)
    return ent, ent_norm

print('\n=== Per-layer attention entropy summary ===')
print('  Normalized entropy: 0 = one-hot (single key), 1 = uniform over valid keys')
print('  Reporting fraction in bins, plus mean and median.\n')

bins = [(0.0, 0.1, 'one-hot   '),
        (0.1, 0.3, 'sharp     '),
        (0.3, 0.5, 'mid-low   '),
        (0.5, 0.7, 'mid-high  '),
        (0.7, 0.9, 'diffuse   '),
        (0.9, 1.01, 'uniform   ')]

for layer_idx, A in enumerate(ATTN_PROBS):
    ent, ent_norm = row_entropy(A)
    # Skip row 0 (trivially 0)
    en = ent_norm[..., 1:].flatten()
    print(f'Layer {layer_idx}: head-mean={ent_norm.mean(dim=(0, 2)).numpy()}')
    print(f'         overall mean={en.mean():.3f}  median={en.median():.3f}  '
          f'std={en.std():.3f}  min={en.min():.3f}  max={en.max():.3f}')
    counts = []
    for lo, hi, name in bins:
        frac = ((en >= lo) & (en < hi)).float().mean().item()
        counts.append((name.strip(), frac))
    counts_str = ' | '.join(f'{n}={f*100:.1f}%' for n, f in counts)
    print(f'         {counts_str}')
    print()

# Aggregate across all layers
all_en = torch.cat([row_entropy(A)[1][..., 1:].flatten() for A in ATTN_PROBS])
print('=== Aggregate (all layers) ===')
print(f'mean={all_en.mean():.3f}  median={all_en.median():.3f}  std={all_en.std():.3f}')
for lo, hi, name in bins:
    frac = ((all_en >= lo) & (all_en < hi)).float().mean().item()
    print(f'  {name.strip():10s} ({lo:.1f}..{hi:.1f}): {frac*100:5.1f}%')

# Also report attention "concentration" — max prob per row
print('\n=== Max-prob (concentration) per row ===')
print('  max_prob: 1.0 = exactly one-hot, 1/(i+1) = uniform')
all_maxp = torch.cat([A.max(-1).values[..., 1:].flatten() for A in ATTN_PROBS])
print(f'mean max_prob = {all_maxp.mean():.3f}')
print(f'median max_prob = {all_maxp.median():.3f}')
for thresh in [0.99, 0.9, 0.8, 0.5, 0.3]:
    frac = (all_maxp >= thresh).float().mean().item()
    print(f'  rows with max_prob >= {thresh}: {frac*100:5.1f}%')
