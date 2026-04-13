"""
exp103_rank_proj2 — All LUTs replaced by RankProjection2.

Architecture:
- q/k/v projections: RankProjection2(d_in=48, d_out=d_qk) per head
- out_proj: RankProjection2(d_in=h*d_v, d_out=d) single instance
- unembedder: nn.Linear(d, vocab_size)
- RankAttention unchanged
- smooth_mode=False (STE) for all RankProjection2
"""
import sys, os, json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, Trainer,
    CONTEXT_SIZE, VOCAB_SIZE,
)
from spiky.lutorch.ranking_tools import RankAttention, RankProjection2

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

class LUTBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d       = cfg['embedding_dim']      # 32
        p       = cfg['positional_dim']     # 16
        h       = cfg['num_heads']          # 4
        d_qk    = cfg['d_qk']               # 16
        d_v     = cfg['d_v']                # 16
        lut_input_dim = d + p               # 48
        smooth  = cfg['rank_proj_smooth_mode']

        self.q_proj = RankProjection2(lut_input_dim, d_qk * h, smooth_mode=smooth)
        self.k_proj = RankProjection2(lut_input_dim, d_qk * h, smooth_mode=smooth)
        self.v_proj = RankProjection2(lut_input_dim, d_v  * h, smooth_mode=smooth)
        self.out_proj = RankProjection2(h * d_v, d, smooth_mode=smooth)

        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=0.5)
        self.n_heads = h
        self.d_qk = d_qk
        self.d_v = d_v
        self.d = d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos_flat = torch.cat([x, pos], dim=-1).reshape(B * T, E + pos.shape[-1])  # [B*T, 48]

        # q, k, v: [B*T, h*d_qk] -> [B, h, T, d_qk/d_v]
        q = self.q_proj(x_pos_flat).reshape(B, T, self.n_heads, self.d_qk).permute(0, 2, 1, 3)
        k = self.k_proj(x_pos_flat).reshape(B, T, self.n_heads, self.d_qk).permute(0, 2, 1, 3)
        v = self.v_proj(x_pos_flat).reshape(B, T, self.n_heads, self.d_v  ).permute(0, 2, 1, 3)

        attn = self.rank_attn(q, k, v, is_causal=True)
        # [B, h, T, d_v] -> [B*T, h*d_v]
        attn_flat = attn.permute(0, 2, 1, 3).reshape(B * T, self.n_heads * self.d_v)

        attn_out = self.out_proj(attn_flat).reshape(B, T, E)
        return x + attn_out


class LUTTransformerV2(nn.Module):

    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        p = cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(cfg, i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(d, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)
model = LUTTransformerV2(cfg).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
