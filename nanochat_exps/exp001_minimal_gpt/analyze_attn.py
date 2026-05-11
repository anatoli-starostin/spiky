"""Analyze attention entropy in exp001 (vanilla GPT-2 baseline, val_bpb=1.6256).

Same diagnostics as exp144's analyze_attn.py for direct comparison.

Run from repo root:
    PYTHONPATH=/home/starost/nanochat .venv/bin/python \
        nanochat_exps/exp001_minimal_gpt/analyze_attn.py
"""
import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

n_embd = cfg['n_embd']
n_head = cfg['n_head']
n_layer = cfg['depth']
seq_len = cfg['seq_len']
d_per_head = n_embd // n_head

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()

# Capture attention probabilities A here (per layer, per call).
ATTN_PROBS = []

class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.n_head = n_head
        self.layer_idx = layer_idx
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Explicit attention so we can capture A.
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(causal_mask, float('-inf'))
        a = torch.softmax(s, dim=-1)
        a = torch.nan_to_num(a, nan=0.0)
        ATTN_PROBS.append(a.detach().float().cpu())
        y = torch.matmul(a, v)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head, layer_idx)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks  = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

print('Building model...')
model = MinimalGPT(VOCAB_SIZE, n_embd, n_head, n_layer, seq_len).to(DEVICE)
sd = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE)
model.load_state_dict(sd)
model.eval()

print('Running a batch...')
loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, cfg['device_batch_size'], seq_len, split='val', device=DEVICE,
)
x, _ = next(loader)
with torch.no_grad():
    _ = model(x)

print(f'Captured attention from {len(ATTN_PROBS)} layers')

def row_entropy(A):
    A_clamp = A.clamp_min(1e-12)
    ent = -(A * A_clamp.log()).sum(-1)
    B, H, T, _ = A.shape
    i = torch.arange(T, device=ent.device)
    norm = torch.log(i + 1.0).clamp_min(1e-12)
    ent_norm = ent / norm
    return ent, ent_norm

print('\n=== Per-layer attention entropy summary (vanilla exp001) ===')
print('  Normalized entropy: 0 = one-hot (single key), 1 = uniform over valid keys\n')

bins = [(0.0, 0.1, 'one-hot   '),
        (0.1, 0.3, 'sharp     '),
        (0.3, 0.5, 'mid-low   '),
        (0.5, 0.7, 'mid-high  '),
        (0.7, 0.9, 'diffuse   '),
        (0.9, 1.01, 'uniform   ')]

for layer_idx, A in enumerate(ATTN_PROBS):
    _, ent_norm = row_entropy(A)
    en = ent_norm[..., 1:].flatten()
    head_means = ent_norm.mean(dim=(0, 2)).numpy()
    print(f'Layer {layer_idx}: head-mean={head_means}')
    print(f'         overall mean={en.mean():.3f}  median={en.median():.3f}  '
          f'std={en.std():.3f}  min={en.min():.3f}  max={en.max():.3f}')
    counts_str = ' | '.join(f'{n.strip()}={(((en >= lo) & (en < hi)).float().mean().item())*100:.1f}%' for lo, hi, n in bins)
    print(f'         {counts_str}\n')

all_en = torch.cat([row_entropy(A)[1][..., 1:].flatten() for A in ATTN_PROBS])
print('=== Aggregate (all layers, vanilla exp001) ===')
print(f'mean={all_en.mean():.3f}  median={all_en.median():.3f}  std={all_en.std():.3f}')
for lo, hi, name in bins:
    frac = ((all_en >= lo) & (all_en < hi)).float().mean().item()
    print(f'  {name.strip():10s} ({lo:.1f}..{hi:.1f}): {frac*100:5.1f}%')

print('\n=== Max-prob (concentration) per row, vanilla exp001 ===')
all_maxp = torch.cat([A.max(-1).values[..., 1:].flatten() for A in ATTN_PROBS])
print(f'mean max_prob = {all_maxp.mean():.3f}')
print(f'median max_prob = {all_maxp.median():.3f}')
for thresh in [0.99, 0.9, 0.8, 0.5, 0.3]:
    frac = (all_maxp >= thresh).float().mean().item()
    print(f'  rows with max_prob >= {thresh}: {frac*100:5.1f}%')
