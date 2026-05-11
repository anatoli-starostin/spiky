"""Measure tie frequency in exp144's attention scores.

For each (layer, batch, head, q_row), look at the *raw popcount-based*
attention logits S = (q · k) for the causally-valid keys, and ask:

  - How often is top-1 SCORE tied (multiple keys with the same max)?
  - Distribution of (top1 - top2) gap in popcount-step units.
  - Among the top 5 keys, how many share the top-1 score?

Run from repo root:
    PYTHONPATH=/home/starost/nanochat .venv/bin/python \
        nanochat_exps/exp144_qk_tph384/analyze_ties.py
"""
import sys, os, json, math
import torch
import torch.nn as nn

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']; H = cfg['n_heads']
d_qk = cfg['d_qk']; d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2  # 496

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()

_TINY_KWARGS = dict(weight_dtype=torch.float32,
                    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                    initial_weights_noise=cfg.get('mhlut_init_std', 0.001))

# Capture raw integer-valued popcount-style scores (Q @ K^T BEFORE * scale).
# For Q, K ∈ {±1}^P, Q @ K^T is integer in [-P, +P] with steps of 2.
RAW_SCORES = []  # per layer: [B, H, T, T] (causal, with -inf at masked)

class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.qk_joint = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2*d_qk,
                                         n_anchor_pairs=cfg['qk_input_nap'],
                                         tables_per_head=cfg['qk_tph'],
                                         random_seed=cfg['random_seed']+layer_idx,
                                         device=DEVICE, **_TINY_KWARGS)
        self.v_lut = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                                      n_anchor_pairs=cfg['v_input_nap'],
                                      tables_per_head=cfg['v_tph'],
                                      random_seed=cfg['random_seed']+200+layer_idx,
                                      device=DEVICE, **_TINY_KWARGS)
        self.out_proj = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                                         n_anchor_pairs=cfg['out_input_nap'],
                                         tables_per_head=cfg['out_tph'],
                                         random_seed=cfg['random_seed']+400+layer_idx,
                                         device=DEVICE, **_TINY_KWARGS)
        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.out_v2d = VectorToDominance(E, smooth_mode=False, temperature=canon_t)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg.get('learnable_attn_scale_init', 0.25))))

    def forward(self, x, pos_emb):
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B*T, E)
        qk_out = self.qk_joint(xp)
        q_dom = self.qk_v2d(qk_out[..., :d_qk])
        k_dom = self.qk_v2d(qk_out[..., d_qk:])
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v = self.v_lut(x.reshape(B*T, E)).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        # RAW QK^T (no scale, no softmax) — integer-valued popcount-based.
        s_raw = torch.matmul(q, k.transpose(-2, -1))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=s_raw.device), 1)
        s_raw_masked = s_raw.masked_fill(causal, float('-inf'))
        RAW_SCORES.append(s_raw_masked.detach().cpu())
        # Continue forward as usual (so model state is consistent).
        s = (q * self.attn_scale) @ k.transpose(-2, -1) / math.sqrt(D_QK_P)
        s = s.masked_fill(causal, float('-inf'))
        a = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        attn = a @ v
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
        self.unembedder = nn.Sequential(nn.LayerNorm(concat_dim),
                                        nn.Linear(concat_dim, VOCAB_SIZE))
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
    tokenizer, cfg['device_batch_size'], CONTEXT_SIZE, split='val', device=DEVICE,
)
x, _ = next(loader)
with torch.no_grad():
    _ = model(x)

print(f'Captured raw scores from {len(RAW_SCORES)} layers\n')
print(f'Score values are integers in [-{D_QK_P}, +{D_QK_P}] with step 2 (since popcount step = 2 for ±1).\n')
print('=' * 90)
print(' Tie analysis: for each (layer, batch, head, q_row), look at the top of the score row.')
print(' Skipping rows where causal_count <= 2 (need at least 3 valid keys).')
print('=' * 90)

per_layer_summary = []

for layer_idx, S in enumerate(RAW_SCORES):
    # S: [B, H, T, T] with -inf at causally-masked positions.
    B, H_, T, _ = S.shape
    # Per (B, H, q_row): sort scores descending, look at top1 - top2 gap and tie counts.
    # Only consider rows with >=3 valid keys.
    valid_count = (S != float('-inf')).sum(-1)  # [B, H, T] — = q_row+1 for causal
    rows_mask = valid_count >= 3
    # Sort scores descending; replace -inf with -large for sort stability
    S_finite = S.masked_fill(S == float('-inf'), -1e9)
    sorted_scores, _ = torch.sort(S_finite, dim=-1, descending=True)
    top1 = sorted_scores[..., 0]
    top2 = sorted_scores[..., 1]
    gap_top1_top2 = (top1 - top2)  # in popcount-step units (= 2 per integer step)

    # Count how many keys have the top1 score per row (= |top1 - score| < eps among valid keys).
    # We use exact integer match.
    top1_unsq = top1.unsqueeze(-1)
    same_as_top1 = (S == top1_unsq) & (S != float('-inf'))
    n_top1 = same_as_top1.sum(-1)  # [B, H, T] — count of keys tied at top1

    # Filter to valid rows only.
    valid = rows_mask.flatten()
    gap = gap_top1_top2.flatten()[valid]
    n_top1_flat = n_top1.flatten()[valid]

    print(f'\nLayer {layer_idx}:')
    # Distribution of n_top1 (how many keys share the top-1 score).
    print('  How many keys share the top-1 score (rows with >=3 valid keys):')
    for k in [1, 2, 3, 4, 5, 10, 20]:
        if k == 1:
            frac = (n_top1_flat == 1).float().mean().item()
            print(f'    unique top-1 (no tie):     {frac*100:5.1f}%')
        elif k <= 5:
            frac = (n_top1_flat == k).float().mean().item()
            print(f'    tied with exactly {k} keys: {frac*100:5.1f}%')
        else:
            frac = (n_top1_flat >= k).float().mean().item()
            print(f'    tied with >={k} keys:         {frac*100:5.1f}%')
    mean_n = n_top1_flat.float().mean().item()
    median_n = n_top1_flat.float().median().item()
    print(f'    mean  n_top1: {mean_n:.2f}  median: {median_n:.0f}  max: {n_top1_flat.max().item()}')

    # Gap distribution (in popcount-step units, where step=2).
    print('  Top-1 vs Top-2 gap (popcount-step = 2 per integer step):')
    print(f'    gap = 0  (tied):              {((gap == 0)).float().mean().item()*100:5.1f}%')
    for thr in [2, 4, 6, 10]:
        frac = ((gap > 0) & (gap <= thr)).float().mean().item()
        print(f'    gap in (0, {thr}]:                  {frac*100:5.1f}%')
    print(f'    mean gap: {gap.mean().item():.2f}, median: {gap.median().item():.0f}')

    per_layer_summary.append({
        'layer': layer_idx,
        'tied_top1_pct': (n_top1_flat > 1).float().mean().item() * 100,
        'mean_n_top1': mean_n,
        'mean_gap': gap.mean().item(),
    })

print('\n' + '=' * 90)
print(' Per-layer summary')
print('=' * 90)
print(' layer | tied_top-1 % | mean n_top1 | mean gap (in pop-step)')
for s in per_layer_summary:
    print(f"   {s['layer']}   |    {s['tied_top1_pct']:5.1f}    |    {s['mean_n_top1']:5.2f}     |   {s['mean_gap']:5.2f}")

# Aggregate across layers.
all_n_top1 = []
all_gap = []
for S in RAW_SCORES:
    valid_count = (S != float('-inf')).sum(-1)
    rows_mask = valid_count >= 3
    S_finite = S.masked_fill(S == float('-inf'), -1e9)
    sorted_scores, _ = torch.sort(S_finite, dim=-1, descending=True)
    top1 = sorted_scores[..., 0]; top2 = sorted_scores[..., 1]
    gap = (top1 - top2)
    n_top1 = ((S == top1.unsqueeze(-1)) & (S != float('-inf'))).sum(-1)
    all_n_top1.append(n_top1.flatten()[rows_mask.flatten()])
    all_gap.append(gap.flatten()[rows_mask.flatten()])

all_n_top1 = torch.cat(all_n_top1).float()
all_gap = torch.cat(all_gap)
print(f'\nAggregate across all 6 layers:')
print(f'  rows with unique top-1:  {(all_n_top1 == 1).float().mean()*100:5.1f}%')
print(f'  rows tied (n_top1 > 1):  {(all_n_top1 > 1).float().mean()*100:5.1f}%')
print(f'  mean n_top1: {all_n_top1.mean():.2f}, median: {all_n_top1.median():.0f}, max: {int(all_n_top1.max())}')
print(f'  mean top1-top2 gap (pop-step): {all_gap.mean():.2f}, median: {all_gap.median():.0f}')
