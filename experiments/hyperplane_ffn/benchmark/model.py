"""Rebuild an FFN-slot experiment's model from its config.json, for benchmarking.

Mirrors the model classes the experiment trainers define, so a run can be
reconstructed on any machine from its config alone (and its checkpoint, if the
weights matter). Three FFN families are supported, selected from the config the
same way the trainers select them:

  ffn_type == "dense"                      -> the vanilla 4x MLP (the baseline)
  ffn_type == "compression"                -> CompressionMultiHeadLUT
      with lut_inner_in_dim/-out_dim == -1 -> PureTernaryHyperplaneMHL, imported
                                              from the experiment's own
                                              local_ternary_ffn.py

Nothing here is machine-specific: paths come from arguments and the device is
whatever CUDA reports.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, dev=None):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                           device=dev) / head_dim))
        t = torch.arange(max_seq_len, device=dev, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


def _build_ffn(cfg, exp_dir, n_embd, layer_idx):
    """The FFN slot for one block, chosen exactly as the trainers choose it."""
    kind = cfg.get('ffn_type', 'compression')
    if kind == 'dense':
        return 'dense', nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False))

    lut_in = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
    lut_out = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
    common = dict(
        input_dim=n_embd, output_dim=n_embd, inner_in_dim=lut_in, inner_out_dim=lut_out,
        nap=cfg.get('lut_n_anchor_pairs'), tph=cfg.get('lut_tables_per_head'),
        n_heads=cfg.get('lut_n_heads', 1),
        joint_head_compression=cfg.get('lut_joint_head_compression', False),
        forward_mode=cfg.get('lut_forward_mode', 'hard'),
        use_bf16=cfg.get('lut_use_bf16', False),
        initial_weights_noise=cfg.get('lut_init_weights_noise', 1e-3),
        learnable_temps=bool(cfg.get('lut_learnable_temps', False)),
        random_seed=cfg.get('lut_base_seed', 1000) + layer_idx)

    if lut_in == -1 and lut_out == -1:
        # ternary "pure" family: the class lives beside the experiment
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        from local_ternary_ffn import PureTernaryHyperplaneMHL
        from spiky.lutorch.ternary_hyperplane_multi_head_lut import max_entropy_temp
        t = cfg.get('lut_ternary_temp_init', 0.5)
        if t == 'max_entropy':
            t = float(max_entropy_temp())
        return 'ternary', PureTernaryHyperplaneMHL(
            hyperplane_init=cfg.get('lut_hyperplane_init', 'anchor_pairs'),
            hyperplane_init_scale=cfg.get('lut_hyperplane_init_scale', None),
            ternary_temp_init=t,
            trainable_bias=bool(cfg.get('lut_trainable_bias', False)),
            normalize_projection=cfg.get('lut_normalize_projection', False),
            normalize_weights=bool(cfg.get('lut_normalize_weights', False)),
            decompress_heads=bool(cfg.get('lut_decompress_heads', False)),
            inner_out=cfg.get('lut_inner_out', None),
            nonzero_penalty_weight=float(cfg.get('lut_nonzero_penalty_weight', 0.0)),
            target_nonzero_frac=float(cfg.get('lut_target_nonzero_frac', 0.0)),
            **common)

    from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
    return 'compression', CompressionMultiHeadLUT(**common)


class Block(nn.Module):
    def __init__(self, cfg, exp_dir, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.kind, ffn = _build_ffn(cfg, exp_dir, n_embd, layer_idx)
        if self.kind == 'dense':
            self.mlp = ffn
        else:
            self.ffn = ffn

    def ffn_slot(self, h):
        """The FFN sub-layer alone, [B, T, C] -> [B, T, C]. Timed separately."""
        if self.kind == 'dense':
            return self.mlp(h)
        B, T, C = h.shape
        return self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ffn_slot(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, exp_dir):
        super().__init__()
        V = cfg['tokenizer_vocab_size']
        C, Hd, L = cfg['n_embd'], cfg['n_head'], cfg['depth']
        self.tok_emb = nn.Embedding(V, C)
        self.rope = RotaryEmbedding(C // Hd, max_seq_len=cfg['seq_len'])
        self.blocks = nn.ModuleList(
            [Block(cfg, exp_dir, C, Hd, i) for i in range(L)])
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            if b.kind == 'dense':
                nn.init.zeros_(b.mlp[-1].weight)
            elif getattr(b.ffn, 'has_decompress', False):
                nn.init.zeros_(b.ffn.decompress.weight)
        if bool(cfg.get('tie_unembedder', False)):
            self.head.weight = self.tok_emb.weight
        self.vocab_size = V

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.rope.cos, self.rope.sin)
        return self.head(self.ln_f(x))


def build(exp_dir, load_checkpoint=False, dev=None):
    """Rebuild an experiment's model. Returns (config, model).

    load_checkpoint=True loads checkpoint.pt from the experiment dir; use it when
    the measurement depends on the trained values (e.g. realized ternary sparsity).
    Pure timing does not need it -- the gather reads a row whatever is in it.
    """
    dev = dev or device()
    cfg = json.load(open(os.path.join(exp_dir, 'config.json')))
    torch.manual_seed(cfg.get('random_seed', 0))
    m = GPT(cfg, exp_dir).to(dev)
    if load_checkpoint:
        ck = os.path.join(exp_dir, 'checkpoint.pt')
        if not os.path.exists(ck):
            raise FileNotFoundError(f'no checkpoint.pt in {exp_dir}')
        missing, unexpected = m.load_state_dict(
            torch.load(ck, map_location=dev), strict=False)
        crit = [k for k in missing if 'rope' not in k]
        if crit or unexpected:
            raise RuntimeError(f'state_dict mismatch: missing={crit[:4]} '
                               f'unexpected={list(unexpected)[:4]}')
    m.eval()
    return cfg, m


def lut_modules(m):
    """Every FastMultiHeadLut in the model (empty for the dense baseline)."""
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    return [x for x in m.modules() if isinstance(x, FastMultiHeadLut)]
