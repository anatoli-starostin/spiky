"""Config-driven model builder for the ffn_replacement runs — shared by the fixed-eval
trainer (`train_fixed.py`) and the standalone scorer (`tools/score_checkpoint.py`) so both
rebuild the *identical* architecture from a run's `config.json`.

MinimalGPT + RoPE. The FFN slot of every block is one of:
  * ffn_type="dense"       -> vanilla 384->1536->384 GELU MLP (baselines).
  * ffn_type="compression" -> CompressionMultiHeadLUT (+ optional parallel Linear when gamma=1).
      - ffn_lut_kind (default "compression") may be "fastmhl_raw" -> a raw FastMultiHeadLut
        FFN driven by the config's raw_nap / raw_tph / raw_n_heads keys (used by
        exp_n_0136; note its top-level lut_* keys are inert in that case).
Unembedder is tied (head.weight = tok_emb.weight) when tie_unembedder=True, else untied.

The module structure / attribute names match the original per-run train.py exactly, so a
checkpoint saved by those trainers loads into this model with 0 missing / 0 unexpected keys.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut          # noqa: E402
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT       # noqa: E402


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


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


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = cfg.get('ffn_type', 'compression')
        gamma = int(cfg.get('gamma', 0))
        if self.ffn_type == 'dense':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd, bias=False))
        else:
            self.lin = nn.Linear(n_embd, n_embd, bias=True) if gamma == 1 else None
            fwd = cfg.get('lut_forward_mode', 'hard')
            bf16 = cfg.get('lut_use_bf16', False)
            noise = cfg.get('lut_init_weights_noise', 1e-3)
            learn = bool(cfg.get('lut_learnable_temps', False))
            seed = cfg.get('lut_base_seed', 1000) + layer_idx
            if cfg.get('ffn_lut_kind', 'compression') == 'fastmhl_raw':
                # Raw FastMHL: no compress/decompress; tables emit full n_embd. (exp_n_0136)
                self.ffn = FastMultiHeadLut(
                    input_dim=n_embd, n_heads=int(cfg['raw_n_heads']), n_outputs=n_embd,
                    n_anchor_pairs=int(cfg['raw_nap']), tables_per_head=int(cfg['raw_tph']),
                    forward_mode=fwd, backward_topk=cfg.get('lut_backward_topk', 0),
                    use_bf16=bf16, initial_weights_noise=noise,
                    learnable_temps=learn, random_seed=seed)
            else:
                self.ffn = CompressionMultiHeadLUT(
                    input_dim=n_embd, output_dim=n_embd,
                    inner_in_dim=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
                    inner_out_dim=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')),
                    nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
                    n_heads=cfg.get('lut_n_heads', 1),
                    joint_head_compression=cfg.get('lut_joint_head_compression', False),
                    forward_mode=fwd, backward_topk=cfg.get('lut_backward_topk', 0),
                    use_bf16=bf16, initial_weights_noise=noise,
                    learnable_temps=learn, random_seed=seed,
                    # LookupFFN-line knobs; both default to the pre-existing behaviour
                    lut_impl=cfg.get('lut_impl', 'fast'),
                    forward_confidence=cfg.get('lut_forward_confidence', False),
                    confidence_form=cfg.get('lut_confidence_form', 'bounded'),
                    confidence_gain=cfg.get('lut_confidence_gain', 1.0),
                    # Optional skip INSIDE the FFN: decompress(lut(z) + z). Adds no
                    # parameters and requires eff_in == eff_out. Default False, so every
                    # existing config builds a bit-identical model to before this line
                    # existed (verified by param/buffer sha256 on exp_n_0185's config).
                    inner_residual=cfg.get('lut_inner_residual', False))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        o = self.ffn(h.reshape(B * T, C))
        if o.dim() == 3:                 # raw FastMHL returns [N, n_heads, C]; sum heads
            o = o.sum(dim=1)
        o = o.reshape(B, T, C).to(h.dtype)
        if self.lin is not None:
            o = o + self.lin(h)
        return x + o


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        n_embd, n_head, n_layer, seq_len = cfg['n_embd'], cfg['n_head'], cfg['depth'], cfg['seq_len']
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i, cfg) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if block.ffn_type == 'dense':
                nn.init.zeros_(block.mlp[-1].weight)
            else:
                if getattr(block.ffn, 'has_decompress', False):
                    nn.init.zeros_(block.ffn.decompress.weight)
                if getattr(block, 'lin', None) is not None:
                    nn.init.zeros_(block.lin.weight)
        if bool(cfg.get('tie_unembedder', False)):
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


def build_model(cfg, vocab_size, device='cuda'):
    """Build a MinimalGPT from a run's config dict and move it to `device`."""
    return MinimalGPT(vocab_size, cfg).to(device)
