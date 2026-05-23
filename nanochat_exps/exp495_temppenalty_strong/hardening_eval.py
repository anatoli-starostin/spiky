"""Hardenability test for exp494 (softmax + temp penalty).

Train-soft-deploy-hard: does the soft-trained model keep its bpb when the softmax
routing is replaced by argmax (T_sel->0, = single-lookup matmul-free inference)?
Eval the SAME val batches with (a) trained softmax routing, (b) hardened argmax.
Small gap => hardenable => exp444-class quality at matmul-free inference.
"""
import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
for p in (NANOCHAT_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)
from spiky.lutorch.tiny_multi_head_lut import MatmulMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

DEVICE = 'cuda'
CKPT = os.path.join(HERE, 'checkpoint.pt')
N_BATCHES = 20


def build_model(cfg):
    E, D, H = cfg['embedding_dim'], cfg['residual_dim'], cfg['n_heads']
    d_qk, d_v, L = cfg['d_qk'], cfg['d_v'], cfg['num_layers']
    CTX, V = cfg['context_size'], 32768
    base = math.factorial  # noqa (unused) keep imports tidy

    def mk(input_dim, n_heads, n_out, nap, tph, seed_off, init_std=None):
        return MatmulMultiHeadLut(
            input_dim=input_dim, n_heads=n_heads, n_outputs=n_out, n_anchor_pairs=nap,
            tables_per_head=tph, random_seed=cfg['random_seed'] + seed_off, device=DEVICE,
            weight_dtype=torch.float32, anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            initial_weights_noise=init_std or cfg.get('mhlut_init_std', 0.001),
            soft_score_temp=cfg.get('soft_score_temp', 0.5), select_temp=cfg.get('select_temp', 0.5),
            learnable_temps=True, use_bf16=cfg.get('soft_use_bf16', True),
            gate_mode='softmax', use_bias=False)

    class RoPE(nn.Module):
        def __init__(s):
            super().__init__()
            inv = 1.0 / (cfg.get('rope_base', 10000.0) ** (torch.arange(0, d_qk, 2, dtype=torch.float32, device=DEVICE) / d_qk))
            t = torch.arange(CTX, device=DEVICE, dtype=torch.float32)
            emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
            s.register_buffer('cos', emb.cos(), persistent=False)
            s.register_buffer('sin', emb.sin(), persistent=False)

    def rot(x):
        a, b = x.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)

    def rope(q, k, cos, sin):
        cos, sin = cos[None, None], sin[None, None]
        return q * cos + rot(q) * sin, k * cos + rot(k) * sin

    class RMS(nn.Module):
        def __init__(s): super().__init__(); s.eps = 1e-6
        def forward(s, x): return x / (x.abs().mean(-1, keepdim=True) + s.eps)

    class Block(nn.Module):
        def __init__(s, li):
            super().__init__()
            s.qkv_lut = mk(E, H, 2*d_qk+d_v, cfg['qkv_input_nap'], cfg['qkv_tph'], li, cfg.get('qkv_lut_init_std'))
            s.v_lut = mk(E, H, d_v, cfg['v_input_nap'], cfg['v_tph'], 200+li)
            s.out_proj = mk(H*d_v, 1, E, cfg['out_input_nap'], cfg['out_tph'], 400+li)
            s.residual_lut = mk(E, 1, D, cfg['residual_input_nap'], cfg['residual_tph'], 600+li)
            s.q_norm = nn.LayerNorm(d_qk); s.k_norm = nn.LayerNorm(d_qk)
            s.ln_pre = RMS(); s.ln_post = RMS()
        def forward(s, x, cos, sin):
            B, T, _ = x.shape; xf = x.reshape(B*T, E); xp = s.ln_pre(xf)
            qkv = s.qkv_lut(xp); q = s.q_norm(qkv[..., :d_qk]); k = s.k_norm(qkv[..., d_qk:2*d_qk]); vb = qkv[..., 2*d_qk:]
            q = q.reshape(B, T, H, d_qk).permute(0, 2, 1, 3); k = k.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
            q, k = rope(q, k, cos[:T], sin[:T])
            v = (s.v_lut(xp) + vb).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            oi = a.permute(0, 2, 1, 3).reshape(B*T, H*d_v); oe = s.out_proj(oi).squeeze(1)
            xn = xf + oe; r = s.residual_lut(s.ln_post(xn)).squeeze(1).reshape(B, T, D)
            return xn.reshape(B, T, E), r

    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.tok_emb_E = nn.Embedding(V, E); s.unembedder = nn.Linear(D, V, bias=False)
            s.rope = RoPE(); s.layers = nn.ModuleList([Block(i) for i in range(L)]); s.ln_final = nn.LayerNorm(D)
            s.D = D
        def forward(s, tok):
            B, T = tok.shape
            xr = torch.zeros(B, T, s.D, device=tok.device, dtype=s.tok_emb_E.weight.dtype)
            xl = s.tok_emb_E(tok)
            for ly in s.layers:
                xl, r = ly(xl, s.rope.cos, s.rope.sin); xr = xr + r
            return s.unembedder(s.ln_final(xr))
    return Model().to(DEVICE)


@torch.no_grad()
def bpb(model, batches, token_bytes):
    nats, nb = 0.0, 0
    for x, y in batches:
        logp = F.log_softmax(model(x).float(), -1)
        yv = y.view(-1)
        nats += F.nll_loss(logp.view(-1, logp.size(-1)), yv, ignore_index=-1, reduction='sum').item()
        nb += token_bytes[yv[yv != -1]].sum().item()
    return nats / nb / math.log(2)


def main():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = ck['config']
    model = build_model(cfg)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, cfg['device_batch_size'], cfg['context_size'], split='val', device=DEVICE)
    batches = [(x.clone(), y.clone()) for x, y in (next(loader) for _ in range(N_BATCHES))]
    token_bytes = get_token_bytes(device=DEVICE)

    # trained softmax routing
    soft = bpb(model, batches, token_bytes)
    # report mean trained T_sel
    tsel = [getattr(l, n).log_select_temp.exp().item() for l in model.layers for n in ('qkv_lut','v_lut','out_proj','residual_lut')]
    print(f'mean trained T_sel = {sum(tsel)/len(tsel):.3f}  (min {min(tsel):.3f}, max {max(tsel):.3f})')
    print(f'SOFT  (trained softmax) bpb = {soft:.4f}')

    # hardened: T_sel -> ~0  => softmax -> argmax one-hot = single-lookup matmul-free
    saved = [getattr(l, n).log_select_temp.detach().clone() for l in model.layers for n in ('qkv_lut','v_lut','out_proj','residual_lut')]
    for l in model.layers:
        for n in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
            getattr(l, n).log_select_temp.data.fill_(math.log(1e-4))
    hard = bpb(model, batches, token_bytes)
    print(f'HARD  (argmax / T_sel->0) bpb = {hard:.4f}')
    print(f'hardening gap = {hard - soft:+.4f}  (small => train-soft-deploy-hard works)')


if __name__ == '__main__':
    main()
