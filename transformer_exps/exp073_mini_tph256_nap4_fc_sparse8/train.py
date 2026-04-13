"""
exp073_mini_tph256_nap4_fc_sparse8 — Fork of exp066 (tph=256, nap=4, FULL_COVERAGE) with
sparse_output_dim=8 applied to out_proj, ffn, and unembedder. q/k/v LUTs unchanged
(n_outputs=d_head=8, sparse_output_dim must be strictly less). 50k steps.
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

def make_lut(input_dim, n_heads, n_outputs, cfg, seed_offset=0, sparse_output_dim=None):
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
        sparse_output_dim=sparse_output_dim,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    """Single transformer block with sparse_output_dim on out_proj and ffn."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']
        h = cfg['num_heads']
        d_head = d // h
        s = layer_idx * 10
        sod = cfg['sparse_output_dim']

        # q/k/v: n_outputs=d_head=8, sparse_output_dim not applicable
        self.q_lut = make_lut(d, h, d_head, cfg, s + 0)
        self.k_lut = make_lut(d, h, d_head, cfg, s + 1)
        self.v_lut = make_lut(d, h, d_head, cfg, s + 2)
        # out_proj, ffn: n_outputs=d=32, apply sparse_output_dim
        self.out_proj = make_lut(d, 1, d, cfg, s + 3, sparse_output_dim=sod)
        self.ffn = make_lut(d, 1, d, cfg, s + 4, sparse_output_dim=sod)
        self.rank_attn = RankAttention(d_head, d_head, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_head = d_head
        self.d = d

    def forward(self, x):
        B, T, E = x.shape

        x_flat = x.reshape(-1, E)
        q = self.q_lut(x_flat).permute(1, 0, 2)
        k = self.k_lut(x_flat).permute(1, 0, 2)
        v = self.v_lut(x_flat).permute(1, 0, 2)

        q = q.reshape(self.n_heads, B, T, self.d_head).permute(1, 0, 2, 3)
        k = k.reshape(self.n_heads, B, T, self.d_head).permute(1, 0, 2, 3)
        v = v.reshape(self.n_heads, B, T, self.d_head).permute(1, 0, 2, 3)

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
        sod = cfg['sparse_output_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, d) * 0.1)
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
            sparse_output_dim=sod,
            random_seed=cfg['random_seed'] + 999,
            device=DEVICE,
        )

    def forward(self, tokens):
        B, T = tokens.shape
        z = self.token_embedder(tokens)
        z = z + self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            z = layer(z)
        logits = self.unembedder(z.reshape(-1, z.shape[-1]))
        return logits[:, 0, :].reshape(B, T, -1)


# ── Run ────────────────────────────────────────────────────────────────────────
sampler = make_sampler(DEVICE, random_seed=1)
model = LUTTransformerV2(cfg).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
