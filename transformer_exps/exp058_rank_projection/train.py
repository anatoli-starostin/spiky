"""
exp058_rank_projection — Replace all LUTs with RankProjection (pairwise rank features + linear).
d=32, h=4, d_head=8, M=496 pairs per projection, smooth_mode=False, temperature=1.0, 50k steps.
"""
import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, Trainer,
    CONTEXT_SIZE, VOCAB_SIZE,
)
from spiky.lutorch.ranking_tools import RankAttention, RankProjection

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

class RankBlock(nn.Module):
    """Single transformer block with RankProjection for all projections."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']
        h = cfg['num_heads']
        d_head = d // h
        g = torch.Generator().manual_seed(cfg['random_seed'] + layer_idx * 10)

        self.q_proj = RankProjection(d, h * d_head, smooth_mode=False, temperature=1.0, generator=g)
        self.k_proj = RankProjection(d, h * d_head, smooth_mode=False, temperature=1.0, generator=g)
        self.v_proj = RankProjection(d, h * d_head, smooth_mode=False, temperature=1.0, generator=g)
        self.out_proj = RankProjection(d, d, smooth_mode=False, temperature=1.0, generator=g)
        self.ffn = RankProjection(d, d, smooth_mode=False, temperature=1.0, generator=g)
        self.rank_attn = RankAttention(d_head, d_head, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_head = d_head
        self.d = d

    def forward(self, x):
        B, T, E = x.shape
        x_flat = x.reshape(-1, E)

        q = self.q_proj(x_flat).reshape(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x_flat).reshape(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x_flat).reshape(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, E)

        x = x + self.out_proj(attn).reshape(B, T, E)
        x = x + self.ffn(x.reshape(-1, E)).reshape(B, T, E)

        return x


class RankTransformer(nn.Module):

    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, d) * 0.1)
        self.layers = nn.ModuleList([RankBlock(cfg, i) for i in range(cfg['num_layers'])])
        g = torch.Generator().manual_seed(cfg['random_seed'] + 999)
        self.unembedder = RankProjection(d, cfg['vocab_size'], smooth_mode=False, temperature=1.0, generator=g)

    def forward(self, tokens):
        B, T = tokens.shape
        z = self.token_embedder(tokens)
        z = z + self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            z = layer(z)
        return self.unembedder(z.reshape(-1, z.shape[-1])).reshape(B, T, -1)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)
model = RankTransformer(cfg).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
