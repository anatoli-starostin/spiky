"""
exp001_vanilla_baseline — Vanilla causal transformer, byte-level LM on fineweb.
Baseline for LUT transformer comparisons.
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

class SDPATransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.permute(0,2,1,3).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = x + self.norm1(self.dropout(attn))
        x = x + self.norm2(self.dropout(self.ffn(x)))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len, dropout, ff_mult):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb.weight.data.uniform_(-1, 1)
        self.register_buffer('pos_emb', torch.empty(max_len, d_model).uniform_(-1, 1))
        self.layers = nn.ModuleList([
            SDPATransformerLayer(d_model, n_heads, d_model * ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.token_emb(x) + self.pos_emb[:x.shape[1]]
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, cfg['random_seed'])

model = CharTransformer(
    vocab_size=cfg['vocab_size'],
    d_model=cfg['d_model'],
    n_heads=cfg['n_heads'],
    num_layers=cfg['num_layers'],
    max_len=CONTEXT_SIZE,
    dropout=cfg['dropout'],
    ff_mult=cfg['dim_feedforward_multiplier'],
).to(DEVICE)

Trainer(model, sampler, cfg, EXP_DIR).run()
