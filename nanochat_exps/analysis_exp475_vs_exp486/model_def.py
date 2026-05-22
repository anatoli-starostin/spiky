"""Standalone reconstruction of the exp475/exp486 LUT-LM model.

exp475 (bs=16) and exp486 (bs=48) share IDENTICAL architecture and random_seed
(42), so a single config-driven builder reconstructs either. Anchor pairs are
sampled deterministically from random_seed, so the bit->row mapping is the SAME
across both checkpoints — per-row weight comparison is meaningful.

Usage:
    from model_def import build_model, load_checkpoint
    model, cfg = load_checkpoint('.../checkpoint.pt', device='cuda')
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


def build_model(cfg, device='cuda'):
    """Reconstruct the exp475/exp486 Model from its config dict."""
    E    = cfg['embedding_dim']
    D    = cfg['residual_dim']
    H    = cfg['n_heads']
    d_qk = cfg['d_qk']
    d_v  = cfg['d_v']
    N_LAYERS = cfg['num_layers']
    CONTEXT  = cfg['context_size']
    ROPE_BASE = cfg.get('rope_base', 10000.0)
    VOCAB = cfg['vocab_size']           # injected by load_checkpoint
    NOISE = cfg.get('argmax_noise_eps', 0.0)

    soft_kwargs = dict(
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
        backward_mode='soft',
        soft_score_temp=cfg.get('soft_score_temp', 0.5),
        select_temp=cfg.get('select_temp', 0.5),
        learnable_temps=cfg.get('soft_learnable_temps', True),
        use_bf16=cfg.get('soft_use_bf16', True),
        argmax_noise_eps=NOISE,
    )

    def make_lut(input_dim, n_heads, n_outputs, nap, tph, seed_off, init_std=None):
        kw = dict(soft_kwargs)
        if init_std is not None:
            kw['initial_weights_noise'] = init_std
        return TinyMultiHeadLut(
            input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
            n_anchor_pairs=nap, tables_per_head=tph,
            random_seed=cfg['random_seed'] + seed_off, device=device, **kw)

    class LUTBlock(nn.Module):
        def __init__(self, li):
            super().__init__()
            self.qkv_lut = make_lut(E, H, 2 * d_qk + d_v, cfg['qkv_input_nap'],
                                    cfg['qkv_tph'], li,
                                    cfg.get('qkv_lut_init_std', cfg.get('mhlut_init_std', 0.001)))
            self.v_lut = make_lut(E, H, d_v, cfg['v_input_nap'], cfg['v_tph'], 200 + li)
            self.out_proj = make_lut(H * d_v, 1, E, cfg['out_input_nap'], cfg['out_tph'], 400 + li)
            self.residual_lut = make_lut(E, 1, D, cfg['residual_input_nap'], cfg['residual_tph'], 600 + li)
            self.q_norm = nn.LayerNorm(d_qk)
            self.k_norm = nn.LayerNorm(d_qk)
            self.ln_pre = RMSNorm(E)
            self.ln_post = RMSNorm(E)

        def forward(self, x, cos, sin):
            B, T, _ = x.shape
            x_flat = x.reshape(B * T, E)
            x_pre = self.ln_pre(x_flat)
            qkv_out = self.qkv_lut(x_pre)
            q_vec = self.q_norm(qkv_out[..., :d_qk])
            k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
            v_branch = qkv_out[..., 2 * d_qk:]
            q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
            k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
            q, k = apply_rope(q, k, cos[:T], sin[:T])
            v_lut_out = self.v_lut(x_pre)
            v_vec = v_lut_out + v_branch
            v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
            out_e = self.out_proj(out_in).squeeze(1)
            x_lut_next_flat = x_flat + out_e
            x_lut_next = x_lut_next_flat.reshape(B, T, E)
            r_in = self.ln_post(x_lut_next_flat)
            r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
            return x_lut_next, r_out

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb_E = nn.Embedding(VOCAB, E)
            self.unembedder = nn.Linear(D, VOCAB, bias=False)
            self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT, base=ROPE_BASE, device=device)
            self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
            self.ln_final = nn.LayerNorm(D)
            self.E, self.D, self.H, self.d_qk, self.d_v = E, D, H, d_qk, d_v

        def forward(self, tokens, targets=None):
            B, T = tokens.shape
            x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
            x_lut = self.tok_emb_E(tokens)
            for layer in self.layers:
                x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
                x_resid = x_resid + r
            x_resid = self.ln_final(x_resid)
            logits = self.unembedder(x_resid)
            if targets is not None:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                       reduction='mean', ignore_index=-1)
            return logits

    return Model().to(device)


def load_checkpoint(path, device='cuda', vocab_size=32768):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck['config'])
    cfg['vocab_size'] = vocab_size
    model = build_model(cfg, device=device)
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    # buffers (anchor pairs, bit matrices) are deterministic from seed; tolerate
    # non-persistent buffer absence in the checkpoint.
    real_missing = [k for k in missing if not any(
        s in k for s in ('rope.cos', 'rope.sin'))]
    if real_missing:
        print(f'[WARN] {path}: missing keys (first 10): {real_missing[:10]}')
    if unexpected:
        print(f'[WARN] {path}: unexpected keys (first 10): {unexpected[:10]}')
    model.eval()
    return model, cfg


# Module names for iterating LUTs uniformly.
LUT_NAMES = ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut')


@torch.no_grad()
def compute_lut_indices(lut_module, x):
    """Replicate the soft-mode forward selection: return [B, n_tables] argmax row
    indices. x: [B, input_dim]. Uses the module's own anchor/bit buffers + T_soft."""
    T_soft = lut_module.log_soft_score_temp.exp().float()
    aa = lut_module.soft_anchor_a_long
    bb = lut_module.soft_anchor_b_long
    bm = lut_module.soft_bit_matrix.float()
    d = x[:, aa] - x[:, bb]                        # [B, n_tables, NAP]
    p = d / (T_soft + d.abs())
    ts = torch.einsum('btp,pk->btk', p, bm)         # [B, n_tables, K]
    return ts.argmax(dim=-1)                         # [B, n_tables]
