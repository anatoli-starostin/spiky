"""
exp074_mini_tph256_nap4_fc_dqk16 — Fork of exp066 (tph=256, nap=4, FULL_COVERAGE) with
d_qk=16 instead of d_head=8. q/k LUTs output 16 per head; v stays at d_head=8.
RankAttention uses d_qk=16, d_v=8. ~6.5M params. 50k steps.
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
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    """Single transformer block with d_qk > d_head for q/k projections."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']
        h = cfg['num_heads']
        d_head = d // h
        d_qk = cfg['d_qk']
        s = layer_idx * 10

        self.q_lut = make_lut(d, h, d_qk, cfg, s + 0)
        self.k_lut = make_lut(d, h, d_qk, cfg, s + 1)
        self.v_lut = make_lut(d, h, d_head, cfg, s + 2)
        self.out_proj = make_lut(d, 1, d, cfg, s + 3)
        self.ffn = make_lut(d, 1, d, cfg, s + 4)
        self.rank_attn = RankAttention(d_qk, d_head, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_head = d_head
        self.d_qk = d_qk
        self.d = d

    def forward(self, x):
        B, T, E = x.shape

        x_flat = x.reshape(-1, E)
        q = self.q_lut(x_flat).permute(1, 0, 2)   # [h, B*T, d_qk]
        k = self.k_lut(x_flat).permute(1, 0, 2)
        v = self.v_lut(x_flat).permute(1, 0, 2)   # [h, B*T, d_head]

        q = q.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)   # [B, h, T, d_qk]
        k = k.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = v.reshape(self.n_heads, B, T, self.d_head).permute(1, 0, 2, 3)  # [B, h, T, d_head]

        attn = self.rank_attn(q, k, v, is_causal=True)   # [B, h, T, d_head]
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
