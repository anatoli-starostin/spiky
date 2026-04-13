"""
exp051_mini_recursive_no_sharing — Same as exp050 but always separate refine_lut (no weight reuse).
Every LUT gets an independent square refine_lut for refinement passes.
embedding_dim=32, n_heads=4, d_qk=d_v=8, nap=8, tph=16, hard RankAttn t=1.0, 50k steps.
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
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.lutorch.ranking_tools import RankAttention

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Helper ─────────────────────────────────────────────────────────────────────

class RecursiveMultiHeadLut(nn.Module):
    """
    Applies n_iters-1 refinement passes (with residual) followed by one final pass.

    If the lut already fits (n_heads=1, n_outputs==input_dim), its own weights are
    reused for refinement — no extra parameters.  Otherwise a separate square
    refine_lut is created for the intermediate passes.
    """
    def __init__(self, lut: MultiHeadLut, n_iters: int = 2, refine_seed_offset: int = 1000):
        super().__init__()
        self.lut = lut
        self.n_iters = n_iters
        if n_iters > 1:
            self.refine_lut = MultiHeadLut(
                input_dim=lut.input_dim,
                n_heads=1,
                n_outputs=lut.input_dim,
                n_anchor_pairs=lut.n_anchor_pairs,
                tables_per_head=lut.tables_per_head,
                smooth_mode=lut.smooth_mode,
                n_alternatives=lut.n_alternatives,
                normalize_weights=lut.normalize_weights,
                calibrate_output=lut.calibrate_output,
                connected_anchors_mode=False,
                initial_weights_noise=0.001,
                uncertainty_mode=UncertaintyMode.INVERSE_L1,
                random_seed=lut.input_dim + refine_seed_offset,
                device=next(lut.parameters()).device,
            )
        else:
            self.refine_lut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        for _ in range(self.n_iters - 1):
            x = x + self.refine_lut(x)[:, 0, :]
        # final pass: original LUT output (B, n_heads, n_out)
        return self.lut(x)

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
        connected_anchors_mode=cfg['connected_anchors_mode'],
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )

def make_recursive_lut(input_dim, n_heads, n_outputs, cfg, seed_offset=0, n_iters=2):
    lut = make_lut(input_dim, n_heads, n_outputs, cfg, seed_offset)
    return RecursiveMultiHeadLut(lut, n_iters=n_iters, refine_seed_offset=seed_offset + 500)


class LUTBlock(nn.Module):
    """q/k/v/out_proj/ffn all use RecursiveMultiHeadLut (n_iters=2)."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']
        h = cfg['num_heads']
        d_head = d // h
        s = layer_idx * 10

        self.q_lut = make_recursive_lut(d, h, d_head, cfg, s + 0)
        self.k_lut = make_recursive_lut(d, h, d_head, cfg, s + 1)
        self.v_lut = make_recursive_lut(d, h, d_head, cfg, s + 2)
        self.out_proj = make_recursive_lut(d, 1, d, cfg, s + 3)
        self.ffn = make_recursive_lut(d, 1, d, cfg, s + 4)
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
