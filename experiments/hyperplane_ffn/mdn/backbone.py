"""Reusable MinimalGPT+RoPE backbone for the Multi-Map 3D Unembedder (MDN head) work.

Faithful copy of the flex-trainer model (exp043+ CompressionMHL sweep) refactored so the
FFN config is passed in rather than read from module globals — lets us load ANY existing
exp checkpoint (dense or CompressionMHL FFN, tied or untied) and pull hidden states / the
unembedder W for the MDN head experiments (E0..E4). Backbone stays frozen throughout.
"""
import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT


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
    def __init__(self, n_embd, n_head, layer_idx, fcfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = fcfg['ffn_type']
        if self.ffn_type == 'dense':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd, bias=False))
        else:
            self.lin = nn.Linear(n_embd, n_embd, bias=True) if fcfg['gamma'] == 1 else None
            self.ffn = CompressionMultiHeadLUT(
                input_dim=n_embd, output_dim=n_embd,
                inner_in_dim=fcfg['lut_in'], inner_out_dim=fcfg['lut_out'],
                nap=fcfg['lut_nap'], tph=fcfg['lut_tph'], n_heads=fcfg['lut_heads'],
                joint_head_compression=fcfg['lut_joint'], forward_mode=fcfg['lut_fwd'],
                use_bf16=fcfg['lut_bf16'], initial_weights_noise=fcfg['lut_noise'],
                random_seed=fcfg['lut_seed'] + layer_idx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        out = self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)
        if self.lin is not None:
            out = out + self.lin(h)
        return x + out


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len, fcfg, tie=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i, fcfg) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie:
            self.head.weight = self.tok_emb.weight

    def get_device(self):
        return self.tok_emb.weight.device

    def hidden(self, idx):
        """Return h = ln_f(final x), shape [B, T, n_embd] — the pre-head hidden state."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        return self.ln_f(x)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits = self.head(self.hidden(idx))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


def fcfg_from_config(cfg):
    return dict(
        ffn_type=cfg.get('ffn_type', 'compression'),
        gamma=int(cfg.get('gamma', 0)),
        lut_in=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
        lut_out=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')),
        lut_nap=cfg.get('lut_n_anchor_pairs'),
        lut_tph=cfg.get('lut_tables_per_head'),
        lut_heads=cfg.get('lut_n_heads', 1),
        lut_joint=cfg.get('lut_joint_head_compression', False),
        lut_fwd=cfg.get('lut_forward_mode', 'hard'),
        lut_bf16=cfg.get('lut_use_bf16', False),
        lut_noise=cfg.get('lut_init_weights_noise', 1e-3),
        lut_seed=cfg.get('lut_base_seed', 1000),
    )


def load_pretrained(exp_dir, device='cpu'):
    """Build MinimalGPT from <exp_dir>/config.json and load <exp_dir>/checkpoint.pt.
    Returns (model, cfg). Model is in eval mode with grads off."""
    with open(os.path.join(exp_dir, 'config.json')) as f:
        cfg = json.load(f)
    fcfg = fcfg_from_config(cfg)
    model = MinimalGPT(cfg['tokenizer_vocab_size'], cfg['n_embd'], cfg['n_head'],
                       cfg['depth'], cfg['seq_len'], fcfg, tie=bool(cfg.get('tie_unembedder', False)))
    ckpt = torch.load(os.path.join(exp_dir, 'checkpoint.pt'), map_location='cpu', weights_only=False)
    sd = ckpt
    for k in ('model', 'state_dict', 'model_state_dict'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]; break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, (missing, unexpected)
