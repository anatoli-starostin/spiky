"""Compare attention matrices: exp071 (LUT-Q/K + V2D ranking) vs exp076 (linear Q/K).

Loads the two checkpoints, forwards an identical short prompt through both,
captures attention matrices manually (since F.scaled_dot_product_attention
doesn't expose weights), and saves three figures:

  1. attention_overview.png       — 6 layers × 2 cols (head-averaged), exp071 | exp076
  2. attention_layer0_heads.png   — layer 0 only, 6 heads × 2 models
  3. attention_layer5_heads.png   — layer 5 only, 6 heads × 2 models

Runs on CPU to avoid contention with any running training.

Caveat for exp076: attn_scale=0.25 was suboptimal for linear Q/K (designed
for ±1 ranking sums). Its attention is partially under-sharpened compared
to a fully-tuned vanilla transformer.

Launch:
    PYTHONPATH=/home/starost/nanochat \\
        /home/starost/spiky/.venv/bin/python -u \\
        nanochat_exps/analyze_attn_exp071_vs_exp076.py
"""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NANOCHAT_ROOT = '/home/starost/nanochat'
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import VectorToDominance, DominanceToVector

DEVICE = 'cpu'
TOKENIZER_DIR = os.path.join(get_base_dir(), 'tokenizer')
print(f"Loading tokenizer from {TOKENIZER_DIR}")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Vocab: {VOCAB_SIZE}")

ROOT = '/home/starost/spiky'
def load_cfg(name):
    with open(os.path.join(ROOT, 'nanochat_exps', name, 'config.json')) as f:
        return json.load(f)

cfg_071 = load_cfg('exp071_canonicalize_out')
cfg_076 = load_cfg('exp076_linear_qk')


def _tiny_kwargs(cfg):
    return dict(
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    )


# --- exp071 LUTBlock with attention probe -----------------------------------
class LUTBlock071(nn.Module):
    def __init__(self, layer_idx, cfg):
        super().__init__()
        self.cfg = cfg
        self.E = E = cfg['embedding_dim']
        self.H = H = cfg['n_heads']
        self.d_qk = d_qk = cfg['d_qk']
        self.d_v = d_v = cfg['d_v']
        self.D_QK_P = d_qk * (d_qk - 1) // 2
        canon_t = cfg.get('canon_temperature', 0.1)
        seed = cfg['random_seed']
        TINY = _tiny_kwargs(cfg)
        out_tph = cfg['out_tph_per_layer'][layer_idx]

        self.qk_joint = TinyMultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=2 * d_qk,
            n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
            random_seed=seed + layer_idx, device=DEVICE, **TINY,
        )
        self.v_lut = TinyMultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
            random_seed=seed + 200 + layer_idx, device=DEVICE, **TINY,
        )
        self.out_proj = TinyMultiHeadLut(
            input_dim=H * d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=cfg['out_input_nap'], tables_per_head=out_tph,
            random_seed=seed + 400 + layer_idx, device=DEVICE, **TINY,
        )
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.out_v2d = VectorToDominance(E, smooth_mode=False, temperature=canon_t)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg['learnable_attn_scale_init'])))

    def forward_attention(self, x, pos_emb):
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, self.E)
        x_flat = x.reshape(B * T, self.E)
        qk_out = self.qk_joint(xp)
        q_dom = self.qk_v2d(qk_out[..., :self.d_qk])
        k_dom = self.qk_v2d(qk_out[..., self.d_qk:])
        q = q_dom.reshape(B, T, self.H, self.D_QK_P).permute(0, 2, 1, 3) * self.attn_scale
        k = k_dom.reshape(B, T, self.H, self.D_QK_P).permute(0, 2, 1, 3)
        # Manual attention with causal mask + 1/sqrt(d) scaling.
        scale = 1.0 / math.sqrt(self.D_QK_P)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        v_vec = self.v_lut(x_flat)
        v = v_vec.reshape(B, T, self.H, self.d_v).permute(0, 2, 1, 3)
        attn_out = torch.matmul(attn, v)
        out_in = attn_out.permute(0, 2, 1, 3).reshape(B * T, self.H * self.d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        out_dom = self.out_v2d(out_real)
        out_rank = self.out_d2v(out_dom).reshape(B, T, self.E)
        return out_rank, attn


# --- exp076 LUTBlock with attention probe -----------------------------------
class LUTBlock076(nn.Module):
    def __init__(self, layer_idx, cfg):
        super().__init__()
        self.cfg = cfg
        self.E = E = cfg['embedding_dim']
        self.H = H = cfg['n_heads']
        self.d_qk = d_qk = cfg['d_qk']
        self.d_v = d_v = cfg['d_v']
        canon_t = cfg.get('canon_temperature', 0.1)
        seed = cfg['random_seed']
        TINY = _tiny_kwargs(cfg)
        out_tph = cfg['out_tph_per_layer'][layer_idx]

        self.q_proj = nn.Linear(E, H * d_qk, bias=True)
        self.k_proj = nn.Linear(E, H * d_qk, bias=True)
        self.v_lut = TinyMultiHeadLut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
            random_seed=seed + 200 + layer_idx, device=DEVICE, **TINY,
        )
        self.out_proj = TinyMultiHeadLut(
            input_dim=H * d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=cfg['out_input_nap'], tables_per_head=out_tph,
            random_seed=seed + 400 + layer_idx, device=DEVICE, **TINY,
        )
        self.out_v2d = VectorToDominance(E, smooth_mode=False, temperature=canon_t)
        self.out_d2v = DominanceToVector(E, normalise=True)
        self.attn_scale = nn.Parameter(torch.tensor(float(cfg['learnable_attn_scale_init'])))

    def forward_attention(self, x, pos_emb):
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, self.E)
        x_flat = x.reshape(B * T, self.E)
        q = self.q_proj(xp).reshape(B, T, self.H, self.d_qk).permute(0, 2, 1, 3) * self.attn_scale
        k = self.k_proj(xp).reshape(B, T, self.H, self.d_qk).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.d_qk)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        v_vec = self.v_lut(x_flat)
        v = v_vec.reshape(B, T, self.H, self.d_v).permute(0, 2, 1, 3)
        attn_out = torch.matmul(attn, v)
        out_in = attn_out.permute(0, 2, 1, 3).reshape(B * T, self.H * self.d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        out_dom = self.out_v2d(out_real)
        out_rank = self.out_d2v(out_dom).reshape(B, T, self.E)
        return out_rank, attn


# --- Outer Model ------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, BlockCls, cfg):
        super().__init__()
        torch.manual_seed(cfg['random_seed'])
        E = cfg['embedding_dim']
        N_LAYERS = cfg['num_layers']
        T = cfg['context_size']
        self.cfg = cfg
        self.E = E
        self.N_LAYERS = N_LAYERS
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.pos_embs = nn.ParameterList([nn.Parameter(torch.zeros(T, E)) for _ in range(N_LAYERS)])
        self.layers = nn.ModuleList([BlockCls(i, cfg) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        self.unembedder = nn.Sequential(nn.LayerNorm(concat_dim), nn.Linear(concat_dim, VOCAB_SIZE))

    def forward_with_attn(self, tokens):
        x = self.token_embedder(tokens)
        attns = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x, attn = layer.forward_attention(x, pos_emb[:tokens.size(1)])
            attns.append(attn)
        return attns


def load_model(BlockCls, cfg, ckpt_path):
    print(f"Building model + loading {ckpt_path}")
    m = Model(BlockCls, cfg).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:   print(f"  WARNING: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected: print(f"  WARNING: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    m.eval()
    return m


m_071 = load_model(LUTBlock071, cfg_071,
                   os.path.join(ROOT, 'nanochat_exps', 'exp071_canonicalize_out', 'checkpoint.pt'))
m_076 = load_model(LUTBlock076, cfg_076,
                   os.path.join(ROOT, 'nanochat_exps', 'exp076_linear_qk', 'checkpoint.pt'))


# --- Sample input -----------------------------------------------------------
PROMPT = "The history of artificial intelligence began in the 1950s with the work of researchers such as Alan Turing, who pioneered the theoretical"
ids = tokenizer.encode(PROMPT, prepend="<|bos|>")
T = min(len(ids), 64)
ids = ids[:T]
print(f"Prompt: {PROMPT!r}")
print(f"Tokens (T={T}): first 10 = {ids[:10]}")
tokens = torch.tensor([ids], dtype=torch.long, device=DEVICE)


print("Forwarding exp071 (CPU) ...")
with torch.no_grad():
    attns_071 = m_071.forward_with_attn(tokens)  # list of [1, H, T, T]
print("Forwarding exp076 (CPU) ...")
with torch.no_grad():
    attns_076 = m_076.forward_with_attn(tokens)


N_LAYERS = cfg_071['num_layers']
H = cfg_071['n_heads']
print(f"Got attention: N_LAYERS={N_LAYERS}, H={H}, T={T}")

# Decode token strings for axis labels.
tok_strs = [tokenizer.decode([t]) for t in ids]
def short(s):
    s = s.replace('\n', '\\n')
    return s if len(s) <= 8 else s[:7] + '…'
labels = [short(s) for s in tok_strs]


# ---------------------------------------------------------------------------
# Figure 1: head-averaged overview, all layers
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(N_LAYERS, 2, figsize=(10, 2.4 * N_LAYERS))
for L in range(N_LAYERS):
    a071 = attns_071[L][0].mean(0).cpu().numpy()  # [T, T]
    a076 = attns_076[L][0].mean(0).cpu().numpy()
    vmax = max(a071.max(), a076.max())
    for j, (a, name) in enumerate([(a071, 'exp071'), (a076, 'exp076')]):
        ax = axes[L, j]
        im = ax.imshow(a, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')
        ax.set_title(f"L{L} {name} (mean over heads)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Head-averaged attention | T={T} | exp071 (LUT-Q/K + V2D ranking) vs exp076 (linear Q/K)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out1 = os.path.join(ROOT, 'nanochat_exps', 'attention_overview.png')
plt.savefig(out1, dpi=120)
plt.close()
print(f"Saved {out1}")


# ---------------------------------------------------------------------------
# Per-head detail figures (layer 0 and layer 5)
# ---------------------------------------------------------------------------
def per_head_figure(layer_idx, out_path):
    fig, axes = plt.subplots(2, H, figsize=(2.2 * H, 5))
    for h in range(H):
        for j, (attns, name) in enumerate([(attns_071, 'exp071'), (attns_076, 'exp076')]):
            ax = axes[j, h]
            mat = attns[layer_idx][0, h].cpu().numpy()
            ax.imshow(mat, vmin=0, vmax=mat.max(), cmap='viridis', aspect='auto')
            ax.set_title(f"L{layer_idx} H{h} {name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Per-head attention | layer {layer_idx} | T={T}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


per_head_figure(0, os.path.join(ROOT, 'nanochat_exps', 'attention_layer0_heads.png'))
per_head_figure(N_LAYERS - 1, os.path.join(ROOT, 'nanochat_exps', f'attention_layer{N_LAYERS-1}_heads.png'))


# ---------------------------------------------------------------------------
# Quantitative summary: attention entropy per layer / head
# ---------------------------------------------------------------------------
def attn_entropy(attn):
    # attn: [B, H, T, T]; per-row entropy averaged over rows and batch.
    p = attn[0].clamp_min(1e-12)  # [H, T, T]
    # Normalize over keys (already done by softmax) and compute -sum p log p per query.
    ent = -(p * p.log()).sum(-1)  # [H, T]
    return ent.mean(-1).cpu().numpy()  # [H]


print("\nAttention entropy per (layer, head) — lower = more peaked, higher = more uniform:")
print(f"  log(T)={math.log(T):.3f} = max entropy (uniform)")
print(f"{'layer':>5} | {'exp071 (mean)':>14} | {'exp076 (mean)':>14}")
for L in range(N_LAYERS):
    e071 = attn_entropy(attns_071[L]).mean()
    e076 = attn_entropy(attns_076[L]).mean()
    print(f"{L:>5} | {e071:>14.3f} | {e076:>14.3f}")


print("\nDone.")
