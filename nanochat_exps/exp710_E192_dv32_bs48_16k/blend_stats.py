#!/usr/bin/env python3
"""Measure hybrid_smooth blend statistics on exp710.

For each LUT module across all 6 layers, per (token, table):
- d_min: confidence (smaller -> more uncertain)
- u = sigmoid(-2 d_min / (T_soft + d_min) / T_sel): blend weight on alt row
- cos_sim(row_main, row_alt): how contrastive the two candidate rows are
- ||u * row_alt|| / ||main_w * row_main + u * row_alt||: alt's magnitude share

Aggregates per module: u distribution, fraction skippable at various eps,
cos_sim distribution, alt-contribution distribution.
"""
import sys, os, json, math, csv, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/starost/spiky/src')

# Load model module from exp710's train.py — we want the Model class.
# We monkey-patch sys.argv / os.environ to avoid the training loop running.
SPEC = importlib.util.spec_from_file_location('exp710_train', os.path.join(EXP_DIR, 'train.py'))
# Defer loading because train.py runs side-effecting setup; instead recreate just the model definition.

# Reuse the canonical TinyMultiHeadLut directly.
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

CFG = json.load(open(os.path.join(EXP_DIR, 'config.json')))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
E = CFG['embedding_dim']; D = CFG['residual_dim']; H = CFG['n_heads']
d_qk = CFG['d_qk']; d_v = CFG['d_v']; N_LAYERS = CFG['num_layers']
CONTEXT_SIZE = CFG['context_size']
_ROPE_BASE = CFG.get('rope_base', 10000.0)
_NOISE_EPS = CFG.get('argmax_noise_eps', 0.0)


# ----- Tokenizer + dataloader (val) -----
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()


# ----- LUT factories (clone from exp710 train.py) -----
_TINY_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=CFG.get('mhlut_init_std', 0.001),
    backward_mode=CFG.get('backward_mode', 'soft'),
    soft_score_temp=CFG.get('soft_score_temp', 0.5),
    select_temp=CFG.get('select_temp', 0.5),
    learnable_temps=CFG.get('soft_learnable_temps', True),
    use_bf16=CFG.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS,
)

def _make(input_dim, n_heads, n_outputs, nap, tph, seed):
    return TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph,
        random_seed=CFG['random_seed'] + seed, device=DEVICE,
        **_TINY_KWARGS)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q*cos + _rotate_half(q)*sin, k*cos + _rotate_half(k)*sin


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.qk_lut = _make(E, H, 2*d_qk, CFG['qkv_input_nap'], CFG['qkv_tph'], idx)
        self.v_lut  = _make(E, H, d_v,    CFG['v_input_nap'],   CFG['v_tph'],   200+idx)
        self.out_proj = _make(H*d_v, 1, E, CFG['out_input_nap'], CFG['out_tph'], 400+idx)
        self.residual_lut = _make(E, 1, D, CFG['residual_input_nap'], CFG['residual_tph'], 600+idx)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E); self.ln_resid = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B*T, E)
        x_pre = self.ln_pre(x_flat)
        qk = self.qk_lut(x_pre)
        q = self.q_norm(qk[..., :d_qk]); k = self.k_norm(qk[..., d_qk:2*d_qk])
        q = q.reshape(B, T, H, d_qk).permute(0,2,1,3)
        k = k.reshape(B, T, H, d_qk).permute(0,2,1,3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v = self.v_lut(x_pre).reshape(B, T, H, d_v).permute(0,2,1,3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0,2,1,3).reshape(B*T, H*d_v)
        out_e = self.out_proj(out_in).squeeze(1)
        x_lut_next = x_flat + out_e
        r_in = self.ln_resid(x_lut_next)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next.reshape(B, T, E), r_out, x_pre, out_in, r_in   # also return LUT inputs


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.emb_resid_lut = _make(E, 1, D, CFG['emb_resid_input_nap'], CFG['emb_resid_tph'], 800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)


    def forward(self, tokens):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B*T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D)
        capture = {'emb_resid_lut': x_emb_pre}
        for li, layer in enumerate(self.layers):
            x_lut, r, x_pre, out_in, r_in = layer(x_lut, self.rope.cos, self.rope.sin)
            capture[f'L{li}.qk_lut']       = x_pre
            capture[f'L{li}.v_lut']        = x_pre
            capture[f'L{li}.out_proj']     = out_in
            capture[f'L{li}.residual_lut'] = r_in
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        return logits, capture


def compute_blend_stats(lut_module, x, chunk=128):
    """Recompute the hybrid_smooth forward decisions for stats.
    Chunked over tokens to keep peak memory low."""
    a_idx = lut_module.soft_anchor_a_long
    b_idx = lut_module.soft_anchor_b_long
    powers = lut_module.soft_powers
    W = lut_module.weights                    # [n_tables, 2^NAP, n_out]
    T_soft = float(lut_module.log_soft_score_temp.detach().exp())
    T_sel  = float(lut_module.log_select_temp.detach().exp())
    n_tables = a_idx.shape[0]
    n_out = W.shape[2]
    B_eff = x.shape[0]
    table_idx_row = torch.arange(n_tables, device=W.device)

    us, dms, coss, alts = [], [], [], []
    with torch.no_grad():
        for start in range(0, B_eff, chunk):
            end = min(start + chunk, B_eff)
            xc = x[start:end]
            d = xc[:, a_idx] - xc[:, b_idx]                                  # [c, n_tables, NAP]
            abs_d = d.abs()
            bits = (d > 0).to(torch.int64)
            main_idx = (bits * powers.view(1,1,-1)).sum(dim=-1)
            p_star = abs_d.argmin(dim=-1)
            flip_mask = powers.to(main_idx.dtype)[p_star]
            alt_idx = main_idx ^ flip_mask
            d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
            delta_ts = 2.0 * d_min / (T_soft + d_min)
            u = torch.sigmoid(-delta_ts / T_sel)
            main_w = 1.0 - u
            c = end - start
            tix = table_idx_row.view(1,-1).expand(c,-1)
            row_main = W[tix, main_idx]                                       # [c, n_tables, n_out]
            row_alt  = W[tix, alt_idx]
            dot_ma   = (row_main * row_alt).sum(dim=-1)
            nm = row_main.norm(dim=-1).clamp_min(1e-12)
            na = row_alt.norm(dim=-1).clamp_min(1e-12)
            cos_sim = dot_ma / (nm * na)
            # Per-element contributions
            main_n = main_w * nm
            alt_n  = u * na
            # ||main_contrib + alt_contrib||^2 = main_n^2 + alt_n^2 + 2*main_w*u*dot_ma
            out_n_sq = main_n*main_n + alt_n*alt_n + 2.0*main_w*u*dot_ma
            out_n = out_n_sq.clamp_min(1e-24).sqrt()
            alt_share = alt_n / (out_n + 1e-12)
            us.append(u.flatten().cpu())
            dms.append(d_min.flatten().cpu())
            coss.append(cos_sim.flatten().cpu())
            alts.append(alt_share.flatten().cpu())
            del d, abs_d, bits, main_idx, p_star, flip_mask, alt_idx, d_min
            del delta_ts, u, main_w, row_main, row_alt, dot_ma, nm, na, cos_sim
            del main_n, alt_n, out_n_sq, out_n, alt_share, tix
    return {
        'u': torch.cat(us),
        'd_min': torch.cat(dms),
        'cos_sim': torch.cat(coss),
        'alt_share': torch.cat(alts),
    }


def summarize(stats, name):
    u = stats['u']; cos = stats['cos_sim']; alt = stats['alt_share']; dm = stats['d_min']
    out = {'name': name, 'n': u.numel()}
    out['u_mean']  = u.mean().item()
    out['u_p50']   = u.median().item()
    out['u_p90']   = u.kthvalue(int(0.9*u.numel())).values.item()
    out['u_p99']   = u.kthvalue(int(0.99*u.numel())).values.item()
    out['u_p999']  = u.kthvalue(int(0.999*u.numel())).values.item()
    for eps in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.1):
        out[f'u<{eps:g}'] = float((u < eps).float().mean())
    out['cos_p50'] = cos.median().item() if cos.numel() > 0 else 0.0
    out['cos_p10'] = cos.kthvalue(int(0.1*cos.numel())).values.item()
    out['cos_mean'] = cos.mean().item()
    out['alt_share_mean'] = alt.mean().item()
    out['alt_share_p90']  = alt.kthvalue(int(0.9*alt.numel())).values.item()
    out['d_min_mean'] = dm.mean().item()
    out['d_min_p10']  = dm.kthvalue(int(0.1*dm.numel())).values.item()
    return out


def main():
    print(f'Loading exp710 checkpoint...')
    ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
    model = Model().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model with {sum(p.numel() for p in model.parameters()):,} params')

    # Small chunk to coexist with exp712 in VRAM
    B, T = 2, 512
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B, T, split='val', device=DEVICE)
    x, y = next(val_loader)
    print(f'Val chunk: B={B}, T={T}, total tokens = {B*T}')

    with torch.no_grad():
        logits, capture = model(x)

    all_stats = {}
    # emb_resid_lut
    s = compute_blend_stats(model.emb_resid_lut, capture['emb_resid_lut'])
    all_stats['emb_resid_lut'] = s

    for li, layer in enumerate(model.layers):
        for name in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(layer, name)
            inp = capture[f'L{li}.{name}']
            s = compute_blend_stats(mod, inp)
            all_stats[f'L{li}.{name}'] = s

    print()
    print(f'{"module":<22s} {"n":>9s} {"u_mean":>7s} {"u_p50":>7s} {"u_p90":>7s} {"u_p99":>7s} {"u<1e-3":>7s} {"u<1e-2":>7s} {"u<.05":>7s} {"u<.1":>7s} {"cos_p50":>7s} {"alt_sh_mean":>11s}')
    summaries = []
    for name, s in all_stats.items():
        d = summarize(s, name)
        summaries.append(d)
        print(f'{d["name"]:<22s} {d["n"]:>9d} {d["u_mean"]:>7.4f} {d["u_p50"]:>7.4f} {d["u_p90"]:>7.4f} {d["u_p99"]:>7.4f} {d["u<0.001"]:>7.3f} {d["u<0.01"]:>7.3f} {d["u<0.05"]:>7.3f} {d["u<0.1"]:>7.3f} {d["cos_p50"]:>7.3f} {d["alt_share_mean"]:>11.4f}')

    # Aggregate by module type
    print()
    print('===== Aggregated by module type (across all layers) =====')
    by_type = {}
    for name, s in all_stats.items():
        kind = name.split('.')[-1]  # qk_lut, v_lut, out_proj, residual_lut, emb_resid_lut
        by_type.setdefault(kind, []).append(s)
    for kind, slist in by_type.items():
        u = torch.cat([s['u'] for s in slist])
        cos = torch.cat([s['cos_sim'] for s in slist])
        alt = torch.cat([s['alt_share'] for s in slist])
        print(f'\n[{kind}]  n_total={u.numel():,}')
        print(f'  u: mean={u.mean():.4f}  p50={u.median():.4f}  p90={u.kthvalue(int(0.9*u.numel())).values.item():.4f}  p99={u.kthvalue(int(0.99*u.numel())).values.item():.4f}  p999={u.kthvalue(int(0.999*u.numel())).values.item():.4f}')
        for eps in (1e-3, 1e-2, 5e-2, 0.1, 0.2):
            print(f'  u<{eps:g}: {(u < eps).float().mean():.4f}')
        print(f'  cos_sim(main,alt): p10={cos.kthvalue(int(0.1*cos.numel())).values.item():.3f}  p50={cos.median():.3f}  p90={cos.kthvalue(int(0.9*cos.numel())).values.item():.3f}')
        print(f'  alt_share (||u*row_alt|| / ||out_table||): mean={alt.mean():.4f}  p50={alt.median():.4f}  p90={alt.kthvalue(int(0.9*alt.numel())).values.item():.4f}')

    # Save raw + summary
    summary_path = os.path.join(EXP_DIR, 'blend_summary.json')
    json.dump({'n_total_tokens': B*T, 'modules': summaries}, open(summary_path,'w'), indent=2)
    print(f'\nSaved summary -> {summary_path}')


if __name__ == '__main__':
    main()
