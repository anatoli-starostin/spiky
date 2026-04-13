"""
exp082_mini_pairvoting — Fork of exp075 replacing all MultiHeadLut with PairVoting.

Each of the C(d,2) input pairs has a dedicated learnable output vector.
output = sum_i r_i * v_i  where r_i = (x[a_i]-x[b_i]) / (t + |x[a_i]-x[b_i]|).

No tables, no AnchorPairsLookup — pure pair-based projection.
Architecture otherwise mirrors exp075: embed=24, pos=8 separate, d_qk=16, 6 layers.

Param count per layer:
  q/k: C(32,2)*d_qk*h = 496*16*4 = 31,744  (all heads share one PairVoting, split output)
  v:   C(32,2)*d_v*h  = 496*6*4  = 11,904
  out_proj/ffn: C(24,2)*24 = 276*24 = 6,624 each
Total ~609k params.
"""
import sys, os, json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, Trainer,
    CONTEXT_SIZE, VOCAB_SIZE,
)
from spiky.lutorch.ranking_tools import RankAttention, PairVoting

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

def make_pv(input_dim, n_outputs, cfg):
    return PairVoting(
        input_dim=input_dim,
        n_outputs=n_outputs,
        smooth_mode=cfg['pv_smooth_mode'],
        temperature=cfg['pv_temperature'],
    )


class PVBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d    = cfg['embedding_dim']   # 24
        p    = cfg['positional_dim']  # 8
        h    = cfg['num_heads']       # 4
        d_qk = cfg['d_qk']           # 16
        d_v  = d // h                 # 6
        lut_input_dim = d + p         # 32

        # q and k: one PairVoting outputting h*d_qk, then split across heads
        self.q_pv    = make_pv(lut_input_dim, h * d_qk, cfg)
        self.k_pv    = make_pv(lut_input_dim, h * d_qk, cfg)
        self.v_pv    = make_pv(lut_input_dim, h * d_v,  cfg)
        self.out_proj = make_pv(d, d, cfg)
        self.ffn      = make_pv(d, d, cfg)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_qk = d_qk
        self.d_v = d_v
        self.d = d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos_flat = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])

        # [B*T, h*d_qk] -> [B*T, h, d_qk] -> [h, B*T, d_qk]
        q = self.q_pv(x_pos_flat).reshape(B * T, self.n_heads, self.d_qk).permute(1, 0, 2)
        k = self.k_pv(x_pos_flat).reshape(B * T, self.n_heads, self.d_qk).permute(1, 0, 2)
        v = self.v_pv(x_pos_flat).reshape(B * T, self.n_heads, self.d_v).permute(1, 0, 2)

        q = q.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = k.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = v.reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)

        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, E)

        x = x + self.out_proj(attn).reshape(B, T, E)
        x = x + self.ffn(x.reshape(-1, E)).reshape(B, T, E)
        return x


class PVTransformer(nn.Module):
    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        p = cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([PVBlock(cfg) for _ in range(cfg['num_layers'])])
        self.unembedder = make_pv(d, cfg['vocab_size'], cfg)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x.reshape(-1, x.shape[-1])).reshape(B, T, -1)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)
model = PVTransformer(cfg).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
