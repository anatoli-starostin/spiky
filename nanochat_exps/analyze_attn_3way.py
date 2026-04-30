"""3-way attention comparison: exp071 (LUT-rank Q/K) vs exp001_minimal_gpt (vanilla GPT) vs exp076 (linear Q/K, hybrid).

Loads three checkpoints, forwards an identical short prompt through all of
them, captures attention matrices manually, saves figures.

  1. attention_3way_overview.png      — head-averaged, all layers, 3 cols
  2. attention_3way_layer0_heads.png  — per-head, layer 0, 3 rows
  3. attention_3way_layer5_heads.png  — per-head, layer 5, 3 rows
  Plus per-layer entropy summary printed.

Runs on CPU.
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
ROOT = '/home/starost/spiky'
TOKENIZER_DIR = os.path.join(get_base_dir(), 'tokenizer')
print(f"Loading tokenizer from {TOKENIZER_DIR}")
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()


def load_cfg(name):
    with open(os.path.join(ROOT, 'nanochat_exps', name, 'config.json')) as f:
        return json.load(f)


# === MinimalGPT (vanilla, exp001) ============================================
class MinimalAttentionWithProbe(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        d_head = C // self.n_head
        scale = 1.0 / math.sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)
        out = self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return out, attn


class MinimalBlockWithProbe(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttentionWithProbe(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        attn_out, attn = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn


class MinimalGPTWithProbe(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks  = nn.ModuleList([MinimalBlockWithProbe(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward_with_attn(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        attns = []
        for block in self.blocks:
            x, attn = block(x)
            attns.append(attn)
        return attns


# === exp071 LUT-rank LUTBlock =================================================
def _tiny_kwargs(cfg):
    return dict(
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    )


class LUTBlock071(nn.Module):
    def __init__(self, layer_idx, cfg):
        super().__init__()
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


class LUTBlock076(nn.Module):
    def __init__(self, layer_idx, cfg):
        super().__init__()
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


class ModelLUT(nn.Module):
    def __init__(self, BlockCls, cfg):
        super().__init__()
        torch.manual_seed(cfg['random_seed'])
        E = cfg['embedding_dim']
        N_LAYERS = cfg['num_layers']
        T = cfg['context_size']
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


# --- load all three models ---------------------------------------------------
def load_lut(BlockCls, cfg, ckpt_path):
    print(f"Loading LUT model: {ckpt_path}")
    m = ModelLUT(BlockCls, cfg).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:    print(f"  missing: {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected: print(f"  unexpected: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    m.eval()
    return m


def load_min(cfg, ckpt_path):
    print(f"Loading MinimalGPT: {ckpt_path}")
    m = MinimalGPTWithProbe(
        vocab_size=VOCAB_SIZE,
        n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_layer=cfg['depth'], seq_len=cfg['seq_len'],
    ).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:    print(f"  missing: {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected: print(f"  unexpected: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    m.eval()
    return m


cfg_071 = load_cfg('exp071_canonicalize_out')
cfg_076 = load_cfg('exp076_linear_qk')
cfg_min = load_cfg('exp001_minimal_gpt')

m_071 = load_lut(LUTBlock071, cfg_071,
                 os.path.join(ROOT, 'nanochat_exps', 'exp071_canonicalize_out', 'checkpoint.pt'))
m_076 = load_lut(LUTBlock076, cfg_076,
                 os.path.join(ROOT, 'nanochat_exps', 'exp076_linear_qk', 'checkpoint.pt'))
m_min = load_min(cfg_min,
                 os.path.join(ROOT, 'nanochat_exps', 'exp001_minimal_gpt', 'checkpoint.pt'))


# --- Sample input -----------------------------------------------------------
PROMPT = "The history of artificial intelligence began in the 1950s with the work of researchers such as Alan Turing, who pioneered the theoretical"
ids = tokenizer.encode(PROMPT, prepend="<|bos|>")
T = min(len(ids), 64)
ids = ids[:T]
print(f"Prompt tokens (T={T}): {ids[:8]}...")
tokens = torch.tensor([ids], dtype=torch.long, device=DEVICE)


print("Forwarding all 3 models on CPU ...")
with torch.no_grad():
    attns_071 = m_071.forward_with_attn(tokens)
    attns_076 = m_076.forward_with_attn(tokens)
    attns_min = m_min.forward_with_attn(tokens)


N_LAYERS = cfg_071['num_layers']  # 6 — all three have 6 layers
H = cfg_071['n_heads']           # 6 — all three have 6 heads
print(f"All three models: N_LAYERS={N_LAYERS}, H={H}, T={T}")


# === Figure 1: head-averaged, all layers, 3 cols =============================
fig, axes = plt.subplots(N_LAYERS, 3, figsize=(13, 2.4 * N_LAYERS))
labels = [('exp001 vanilla GPT', attns_min), ('exp071 LUT-rank', attns_071), ('exp076 linear-Q/K', attns_076)]
for L in range(N_LAYERS):
    mats = [a[L][0].mean(0).cpu().numpy() for _, a in labels]
    vmax = max(m.max() for m in mats)
    for j, (name, mat) in enumerate(zip([n for n, _ in labels], mats)):
        ax = axes[L, j]
        im = ax.imshow(mat, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')
        ax.set_title(f"L{L} {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Head-averaged attention | T={T} | 3-way comparison", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out1 = os.path.join(ROOT, 'nanochat_exps', 'attention_3way_overview.png')
plt.savefig(out1, dpi=120)
plt.close()
print(f"Saved {out1}")


# === Per-head figures (layer 0 and last layer) ===============================
def per_head_3way(layer_idx, out_path):
    fig, axes = plt.subplots(3, H, figsize=(2.2 * H, 7))
    rows = [('exp001 vanilla', attns_min), ('exp071 LUT-rank', attns_071), ('exp076 linear-Q/K', attns_076)]
    for h in range(H):
        for j, (name, attns) in enumerate(rows):
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


per_head_3way(0, os.path.join(ROOT, 'nanochat_exps', 'attention_3way_layer0_heads.png'))
per_head_3way(N_LAYERS - 1, os.path.join(ROOT, 'nanochat_exps', f'attention_3way_layer{N_LAYERS-1}_heads.png'))


# === Entropy summary =========================================================
def attn_entropy(attn):
    p = attn[0].clamp_min(1e-12)
    ent = -(p * p.log()).sum(-1)
    return ent.mean(-1).cpu().numpy()


print(f"\nAttention entropy per layer (lower = more peaked). Max = log(T) = {math.log(T):.3f}")
print(f"{'layer':>5} | {'vanilla':>10} | {'LUT-rank':>10} | {'linear-Q/K':>10}")
for L in range(N_LAYERS):
    e_min = attn_entropy(attns_min[L]).mean()
    e_071 = attn_entropy(attns_071[L]).mean()
    e_076 = attn_entropy(attns_076[L]).mean()
    print(f"{L:>5} | {e_min:>10.3f} | {e_071:>10.3f} | {e_076:>10.3f}")

print("\nDone.")
