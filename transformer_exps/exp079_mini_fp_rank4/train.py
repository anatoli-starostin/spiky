"""
exp079_mini_fp_rank4 — Fork of exp075 with FactoredProjection rank=1 everywhere.

Same architecture as exp075 (embed=24, pos=8 separate, d_qk=16, tph=256, nap=4,
FULL_COVERAGE), but MultiHeadLut uses FactoredProjection instead of LProjection.
FactoredProjection: scalar_weights[n_tables, n_entries, rank] + prototypes[n_tables, rank, n_outputs].
At rank=4 this is ~2.5x fewer projection params than LProjection for small n_outputs,
~15x fewer for the unembedder (n_outputs=257).
"""
import sys, os, json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, Trainer,
    CONTEXT_SIZE, VOCAB_SIZE,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

def make_lut(input_dim, n_heads, n_outputs, cfg, seed_offset=0):
    return MultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=cfg['n_anchor_pairs'],
        tables_per_head=cfg['tables_per_head'],
        smooth_mode=cfg['smooth_mode'],
        n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'],
        calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        projection_rank=cfg['projection_rank'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']       # 24
        p = cfg['positional_dim']      # 8
        h = cfg['num_heads']           # 4
        d_qk = cfg['d_qk']            # 16
        d_v = d // h                   # 6
        lut_input_dim = d + p          # 32
        s = layer_idx * 10

        self.q_lut = make_lut(lut_input_dim, h, d_qk, cfg, s + 0)
        self.k_lut = make_lut(lut_input_dim, h, d_qk, cfg, s + 1)
        self.v_lut = make_lut(lut_input_dim, h, d_v,  cfg, s + 2)
        self.out_proj = make_lut(d, 1, d, cfg, s + 3)
        self.ffn      = make_lut(d, 1, d, cfg, s + 4)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_qk = d_qk
        self.d_v = d_v
        self.d = d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos = torch.cat([x, pos], dim=-1)
        x_pos_flat = x_pos.reshape(-1, E + pos.shape[-1])

        q = self.q_lut(x_pos_flat).permute(1, 0, 2)
        k = self.k_lut(x_pos_flat).permute(1, 0, 2)
        v = self.v_lut(x_pos_flat).permute(1, 0, 2)

        q = q.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = k.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = v.reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)

        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, E)

        attn_out = self.out_proj(attn)[:, 0, :]
        x = x + attn_out.reshape(B, T, E)

        ffn_out = self.ffn(x.reshape(-1, E))[:, 0, :].reshape(B, T, E)
        x = x + ffn_out

        return x


class LUTTransformerV2(nn.Module):

    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        p = cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(cfg, i) for i in range(cfg['num_layers'])])
        self.unembedder = MultiHeadLut(
            input_dim=d,
            n_heads=1,
            n_outputs=cfg['vocab_size'],
            n_anchor_pairs=cfg['n_anchor_pairs'],
            tables_per_head=cfg['tables_per_head'],
            smooth_mode=cfg['smooth_mode'],
            n_alternatives=cfg['n_alternatives'],
            normalize_weights=cfg['normalise_weights'],
            calibrate_output=False,
            connected_anchors_mode=cfg['connected_anchors_mode'],
            initial_weights_noise=cfg['initial_weights_noise'],
            uncertainty_mode=UncertaintyMode.INVERSE_L1,
            projection_rank=cfg['projection_rank'],
            random_seed=cfg['random_seed'] + 999,
            device=DEVICE,
        )

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        logits = self.unembedder(x.reshape(-1, x.shape[-1]))
        return logits[:, 0, :].reshape(B, T, -1)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)
model = LUTTransformerV2(cfg).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
