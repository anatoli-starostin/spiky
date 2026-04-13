"""
exp041_rank_dqk_full — RankAttention d_qk=embedding_dim=32 (496 pairs), d_v=d_head=8,
smooth RankAttn (temperature=0.1), LUT non-smooth, double FFN.
"""
import sys, os, json, time
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

# ── Model ──────────────────────────────────────────────────────────────────────

def make_lut(input_dim, n_heads, n_outputs, cfg, seed_offset=0, tables_per_head_override=None):
    return MultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=cfg['n_anchor_pairs'],
        tables_per_head=tables_per_head_override if tables_per_head_override is not None else cfg['tables_per_head'],
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


class LUTBlock(nn.Module):
    """Single transformer block: q/k/v LUTs → RankAttention (d_qk=d, d_v=d_head) → out_proj LUT → double FFN LUT."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        d = cfg['embedding_dim']
        h = cfg['num_heads']
        d_head = d // h
        s = layer_idx * 10

        # q/k LUTs: H heads, each outputting full d-dim; tph reduced 4x to keep param count same
        qk_tph = cfg['tables_per_head'] // (d // d_head)
        t0 = time.time(); self.q_lut = make_lut(d, h, d, cfg, s + 0, tables_per_head_override=qk_tph); print(f"  layer{layer_idx} q_lut: {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self.k_lut = make_lut(d, h, d, cfg, s + 1, tables_per_head_override=qk_tph); print(f"  layer{layer_idx} k_lut: {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self.v_lut = make_lut(d, h, d_head, cfg, s + 2); print(f"  layer{layer_idx} v_lut: {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self.out_proj = make_lut(d, 1, d, cfg, s + 3); print(f"  layer{layer_idx} out_proj: {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self.ffn1 = make_lut(d, 1, d, cfg, s + 4); print(f"  layer{layer_idx} ffn1: {time.time()-t0:.2f}s", flush=True)
        t0 = time.time(); self.ffn2 = make_lut(d, 1, d, cfg, s + 5); print(f"  layer{layer_idx} ffn2: {time.time()-t0:.2f}s", flush=True)
        # d_qk=d (full embedding, 496 pairs), d_v=d_head — RankAttention handles the split
        self.rank_attn = RankAttention(d, d_head, smooth_mode=True, temperature=0.1)
        print(f"  layer{layer_idx} done", flush=True)
        self.n_heads = h
        self.d_head = d_head
        self.d = d

    def forward(self, x):
        B, T, E = x.shape

        x_flat = x.reshape(-1, E)

        # q/k: H heads, full d-dim each; v: H heads, d_head each
        q = self.q_lut(x_flat).permute(1, 0, 2)  # [H, B*T, d]
        k = self.k_lut(x_flat).permute(1, 0, 2)  # [H, B*T, d]
        v = self.v_lut(x_flat).permute(1, 0, 2)  # [H, B*T, d_head]

        # reshape to (B, H, T, dim)
        q = q.reshape(self.n_heads, B, T, self.d).permute(1, 0, 2, 3)          # [B, H, T, d]
        k = k.reshape(self.n_heads, B, T, self.d).permute(1, 0, 2, 3)          # [B, H, T, d]
        v = v.reshape(self.n_heads, B, T, self.d_head).permute(1, 0, 2, 3)     # [B, H, T, d_head]

        attn = self.rank_attn(q, k, v, is_causal=True)               # [B, H, T, d_head]
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, E)

        attn_out = self.out_proj(attn)[:, 0, :]
        x = x + attn_out.reshape(B, T, E)

        ffn1_out = self.ffn1(x.reshape(-1, E))[:, 0, :].reshape(B, T, E)
        ffn2_out = self.ffn2((x + ffn1_out).reshape(-1, E))[:, 0, :].reshape(B, T, E)
        x = x + ffn1_out + ffn2_out

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
print("making sampler...", flush=True)
sampler = make_sampler(DEVICE, random_seed=1)
print("building model...", flush=True)
t0 = time.time(); model = LUTTransformerV2(cfg); print(f"model built: {time.time()-t0:.2f}s", flush=True)
t0 = time.time(); model = model.to(DEVICE); print(f"model.to(DEVICE): {time.time()-t0:.2f}s", flush=True)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])

Trainer(model, sampler, cfg, EXP_DIR, optimizer=optimizer).run()
