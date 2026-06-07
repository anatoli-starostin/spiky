"""exp696 — fork of exp693 (1.4379 @ 97.5M) with an emb_resid_lut.

Single addition: a 7th contribution to the D-stream, computed directly from
MeanAbsNorm(tok_emb_E(tokens)) — gives the unembedder a direct view of the
bare embedding without going through the LUTBlock stack.

Architecture (vs exp693):
  + emb_resid_lut: TinyMHLut(NAP=5, tph=256, n_heads=1, n_out=D=384), applied to
    the embedding pre-blocks; result initializes x_resid in the D-stream.
  - LUTBlocks: unchanged (x_lut/E-stream is untouched).

D-stream becomes 7 contributions: 1 from bare embedding + 6 from per-layer
residual_lut. Alternative to exp695's pre_lut which mutated the E-stream.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


# ---- Asymmetric mode: 2-row forward + (NAP+1)-row Hamming-1 ball backward ----
# Forward output is the (main + Hamming-1-alt) blend identical to exp724.
# Backward distributes input gradient via (NAP+1)-row softmax over the ball
# (main + all NAP single-bit-flip neighbors). Weight gradient is 2-row scatter
# (consistent with forward). T_sel/T_soft gradients flow through input-grad path.

class _AsymFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, anchor_a_long, anchor_b_long, powers,
                log_T_soft, log_T_sel, n_heads, tph, table_dim, n_outputs):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_tables = anchor_a_long.shape[0]
        NAP = anchor_a_long.shape[1]

        # 2-row forward (same algebra as _hybrid_smooth_lut_fwd_body)
        d = x[:, anchor_a_long] - x[:, anchor_b_long]      # [B, n_tables, NAP]
        abs_d = d.abs()
        bits = (d > 0).to(torch.int64)
        powers_view = powers.view(1, 1, -1)
        main_idx = (bits * powers_view).sum(dim=-1)        # [B, n_tables]
        p_star = abs_d.argmin(dim=-1)                      # [B, n_tables]
        flip_mask = powers.to(main_idx.dtype)[p_star]
        alt_idx = main_idx ^ flip_mask                     # [B, n_tables]
        d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
        delta_ts = 2.0 * d_min / (T_soft + d_min)
        u = torch.sigmoid(-delta_ts / T_sel)
        main_w = 1.0 - u

        table_offset = torch.arange(n_tables, device=weights.device,
                                     dtype=main_idx.dtype) * table_dim
        weights_flat = weights.view(n_tables * table_dim, n_outputs)
        main_flat = (main_idx + table_offset.view(1, -1)).reshape(-1)
        alt_flat  = (alt_idx  + table_offset.view(1, -1)).reshape(-1)
        row_main = F.embedding(main_flat, weights_flat).view(B, n_tables, n_outputs)
        row_alt  = F.embedding(alt_flat,  weights_flat).view(B, n_tables, n_outputs)
        blended = main_w.unsqueeze(-1) * row_main + u.unsqueeze(-1) * row_alt
        out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2)         # [B, n_heads, n_out]

        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                               powers, log_T_soft, log_T_sel,
                               main_idx, alt_idx, d, abs_d)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.table_dim = table_dim
        ctx.n_outputs = n_outputs
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, powers,
         log_T_soft, log_T_sel, main_idx, alt_idx, d, abs_d) = ctx.saved_tensors
        n_heads, tph, table_dim, n_outputs = ctx.n_heads, ctx.tph, ctx.table_dim, ctx.n_outputs
        B = x.shape[0]
        n_tables = anchor_a_long.shape[0]
        NAP = anchor_a_long.shape[1]
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()

        # ---- (NAP+1)-row Hamming-1 ball softmax for INPUT grad -----------------
        # Per-anchor "absolute soft score": |p[i]| = |d[i]| / (T_soft + |d[i]|)
        abs_p = abs_d / (T_soft + abs_d)                                 # [B, n_tables, NAP]

        # ts[main]   = sum_i |p[i]|
        # ts[alt_k]  = ts[main] - 2*|p[k]|
        ts_main = abs_p.sum(dim=-1, keepdim=True)                        # [B, n_tables, 1]
        ts_alt  = ts_main - 2.0 * abs_p                                  # [B, n_tables, NAP]
        ts = torch.cat([ts_main, ts_alt], dim=-1)                        # [B, n_tables, NAP+1]
        z = ts / T_sel
        sel_soft = F.softmax(z, dim=-1)                                  # [B, n_tables, NAP+1]

        table_offset = torch.arange(n_tables, device=weights.device,
                                     dtype=main_idx.dtype) * table_dim
        weights_flat = weights.view(n_tables * table_dim, n_outputs)

        # Compute Z[b,t,r] = sum_o W[ball_idx[r]][o] * grad_out[b, h(t), o]
        # row-by-row to avoid materialising [B, n_tables, NAP+1, n_outputs].
        Z = torch.empty(B, n_tables, NAP + 1, dtype=z.dtype, device=z.device)
        # Reshape grad_out so multiplication broadcasts cleanly: [B, n_heads, 1, n_outputs].
        grad_out_b = grad_out.unsqueeze(2)                                # [B, n_heads, 1, n_outputs]
        for r in range(NAP + 1):
            if r == 0:
                idx_r = main_idx                                          # [B, n_tables]
            else:
                idx_r = main_idx ^ powers.to(main_idx.dtype)[r - 1]
            flat_r = (idx_r + table_offset.view(1, -1)).reshape(-1)
            rows_r = F.embedding(flat_r, weights_flat).view(B, n_heads, tph, n_outputs)
            Z_r_view = (rows_r * grad_out_b).sum(dim=-1)                  # [B, n_heads, tph]
            Z[..., r] = Z_r_view.reshape(B, n_tables)
            del rows_r, Z_r_view

        sum_term = (Z * sel_soft).sum(dim=-1, keepdim=True)
        d_z = sel_soft * (Z - sum_term)
        d_ts = d_z / T_sel
        d_ts_main = d_ts[..., 0]
        d_ts_alt  = d_ts[..., 1:]
        d_abs_p = d_ts_main.unsqueeze(-1) - 2.0 * d_ts_alt
        denom = T_soft + abs_d
        d_abs_d = d_abs_p * T_soft / (denom * denom)
        sign_d = d.sign()
        d_d = d_abs_d * sign_d

        grad_x = torch.zeros_like(x)
        idx_a = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        idx_b = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        d_d_flat = d_d.reshape(B, -1)
        grad_x.scatter_add_(1, idx_a,  d_d_flat)
        grad_x.scatter_add_(1, idx_b, -d_d_flat)

        grad_log_T_sel = -(d_z * z).sum()
        d_abs_p_via_Tsoft = -abs_d / (denom * denom)
        grad_T_soft = (d_abs_p * d_abs_p_via_Tsoft).sum()
        grad_log_T_soft = T_soft * grad_T_soft

        # ---- Weight grad: 2-row scatter (consistent with forward output) -------
        # main_w_view, u_view: [B, n_heads, tph] so we can broadcast against grad_out.
        d_min = abs_d.gather(-1, abs_d.argmin(-1).unsqueeze(-1)).squeeze(-1)
        delta_ts_fwd = 2.0 * d_min / (T_soft + d_min)
        u_fwd_flat = torch.sigmoid(-delta_ts_fwd / T_sel)                 # [B, n_tables]
        main_w_flat = 1.0 - u_fwd_flat
        u_view     = u_fwd_flat.view(B, n_heads, tph, 1)
        main_w_view = main_w_flat.view(B, n_heads, tph, 1)

        # grad_main_scale[b, h, t, o] = main_w * grad_out[b, h, o]
        grad_main_scale = (main_w_view * grad_out_b).reshape(B * n_tables, n_outputs)
        grad_alt_scale  = (u_view      * grad_out_b).reshape(B * n_tables, n_outputs)
        flat_main = (main_idx + table_offset.view(1, -1)).reshape(-1)
        flat_alt  = (alt_idx  + table_offset.view(1, -1)).reshape(-1)
        grad_w_flat = torch.zeros(n_tables * table_dim, n_outputs,
                                   dtype=weights.dtype, device=weights.device)
        grad_w_flat.index_add_(0, flat_main, grad_main_scale)
        grad_w_flat.index_add_(0, flat_alt,  grad_alt_scale)
        grad_weights = grad_w_flat.view(n_tables, table_dim, n_outputs)

        return (grad_x, grad_weights, None, None, None,
                grad_log_T_soft, grad_log_T_sel, None, None, None, None)


def _asym_lut_call(lut: TinyMultiHeadLut, x: torch.Tensor) -> torch.Tensor:
    """Run the 2-fwd / (NAP+1)-bwd asymmetric path on a configured TinyMultiHeadLut."""
    n_tables = lut.soft_anchor_a_long.shape[0]
    table_dim = lut.weights.shape[1]
    n_outputs = lut.weights.shape[2]
    return _AsymFn.apply(
        x, lut.weights, lut.soft_anchor_a_long, lut.soft_anchor_b_long,
        lut.soft_powers, lut.log_soft_score_temp, lut.log_select_temp,
        lut.n_heads, lut.tables_per_head, table_dim, n_outputs,
    )

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
D           = cfg['residual_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
DEVICE_BS   = cfg['device_batch_size']
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']
_ROPE_BASE  = cfg.get('rope_base', 10000.0)
_NOISE_EPS  = cfg.get('argmax_noise_eps', 0.0)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories ------------------------------------------------------------
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode=cfg.get('backward_mode', 'soft'),
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS,
)

def _make_qk(seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_v(seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_out(seed_offset):
    return TinyMultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_residual_lut(seed_offset):
    """Per-layer residual_lut: E -> D, accumulated into the D-stream."""
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_emb_resid_lut(seed_offset):
    """Embedding-level residual_lut: E -> D, written directly to the D-stream.

    7th contribution to x_resid; bypasses the LUTBlock stack.
    """
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )


# --- RoPE on (q, k) -----------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        ))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut       = _make_qk(layer_idx)
        self.v_lut        = _make_v(200 + layer_idx)
        self.out_proj     = _make_out(400 + layer_idx)
        self.residual_lut = _make_residual_lut(600 + layer_idx)

        self.q_norm  = nn.LayerNorm(d_qk)
        self.k_norm  = nn.LayerNorm(d_qk)
        self.ln_pre  = MeanAbsNorm(E)
        self.ln_resid = MeanAbsNorm(E)   # pre-norm before residual_lut

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        x_pre = self.ln_pre(x_flat)

        qk_out = _asym_lut_call(self.qk_lut, x_pre)
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = _asym_lut_call(self.v_lut, x_pre)
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = _asym_lut_call(self.out_proj, out_in).squeeze(1)

        x_lut_next_flat = x_flat + out_e

        # Per-layer residual_lut: MeanAbsNorm(E) -> residual_lut -> D-stream contribution.
        r_in  = self.ln_resid(x_lut_next_flat)
        r_out = _asym_lut_call(self.residual_lut, r_in).squeeze(1).reshape(B, T, D)

        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut = _make_emb_resid_lut(800)
        self.ln_emb_resid  = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        # Required by nanochat.loss_eval.evaluate_bpb.
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        # Initialise D-stream with the bare-embedding residual_lut contribution.
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B * T, E))
        x_resid = _asym_lut_call(self.emb_resid_lut, x_emb_pre).squeeze(1).reshape(B, T, D)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# --- Build + optimiser --------------------------------------------------------
model = Model().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'Total params (all fp32): {n_params:,}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

lut_params     = []
tok_emb_params = []
decay_params   = []
nodecay_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 3:
        lut_params.append(p)
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp['lr'], grp['betas'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if 'exp_avg' not in st:
                    st['exp_avg'] = torch.zeros_like(p)
                m = st['exp_avg']
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                update = (m * b1 + g * (1.0 - b1)).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)

_LUT_LR  = cfg.get('lut_lr', cfg['adam_lr'])
_LUT_OPT = cfg.get('lut_optimizer', 'adamw')

adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)

if _LUT_OPT == 'lion':
    lut_optimizer = Lion([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                         lr=_LUT_LR, betas=tuple(cfg.get('lut_betas', (0.9, 0.99))))
else:
    lut_optimizer = torch.optim.AdamW(
        [dict(params=lut_params, lr=_LUT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])

all_optimizers = [optimizer, lut_optimizer]
for o in all_optimizers:
    for g in o.param_groups:
        g['initial_lr'] = g['lr']
print(
    f'LUT optimizer = {_LUT_OPT} (lut_lr={_LUT_LR}) | '
    f'lut={sum(p.numel() for p in lut_params):,} | '
    f'decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'tok_emb={sum(p.numel() for p in tok_emb_params):,} | '
    f'nodecay_other={sum(p.numel() for p in nodecay_params):,} | non-LUT on AdamW'
)

print(f'D=residual_dim={D}, E=embedding_dim={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'qk_lut       TinyMHLut(soft, LION): in_nap={cfg["qkv_input_nap"]} tph={cfg["qkv_tph"]} n_out=2*d_qk={2*d_qk}')
print(f'v_lut        TinyMHLut(soft): in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v}')
print(f'out_proj     TinyMHLut(soft): in_nap={cfg["out_input_nap"]} tph={cfg["out_tph"]} n_out=E={E}')
print(f'residual_lut TinyMHLut(soft) [per-layer, x{N_LAYERS}]: in_nap={cfg["residual_input_nap"]} tph={cfg["residual_tph"]} n_out=D={D}; MeanAbsNorm(E) before each')
print(f'UNTIED unembedder Linear(D={D}, V={VOCAB_SIZE}); tok_emb_E at E={E}; ln_final(D); RoPE base={_ROPE_BASE}')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(blk, slut_name, None)
            if mod is not None and getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_path = os.path.join(EXP_DIR, 'temperatures.csv')
temp_f = open(temp_path, 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step'] + [name for name, _ in temp_specs])
print(f'Tracking {len(temp_specs)} learnable temperatures in temperatures.csv')

# --- Per-parameter weight-delta tracking --------------------------------------
_PARAM_SNAPSHOT = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
_weight_csv_path = os.path.join(EXP_DIR, 'weight_deltas.csv')
_weight_csv_f = open(_weight_csv_path, 'w', newline='')
_weight_csv_w = csv.writer(_weight_csv_f)
_weight_csv_w.writerow(['step', 'param_name', 'weight_norm', 'delta_norm', 'rel_delta'])
print(f'Tracking weight deltas of {len(_PARAM_SNAPSHOT)} parameters in weight_deltas.csv')

def _log_weight_deltas(step_):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            w = p.detach()
            w_norm = float(w.norm())
            prev = _PARAM_SNAPSHOT.get(n)
            if prev is None or prev.shape != w.shape:
                _PARAM_SNAPSHOT[n] = w.clone()
                continue
            d_norm = float((w - prev).norm())
            rel = (d_norm / w_norm) if w_norm > 0 else 0.0
            _weight_csv_w.writerow([step_, n, f'{w_norm:.6e}', f'{d_norm:.6e}', f'{rel:.6e}'])
            _PARAM_SNAPSHOT[n] = w.clone()
        _weight_csv_f.flush()


# --- Training loop ------------------------------------------------------------
tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb'])

train_losses_logged, val_bpbs, val_steps = [], [], []
ema = None
best_bpb = float('inf')
t0 = time.time()

temp_w.writerow([0] + [f'{getter():.6f}' for _, getter in temp_specs])
temp_f.flush()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step)
    for o in all_optimizers:
        for g in o.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale

    for o in all_optimizers:
        o.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    for o in all_optimizers:
        o.step()

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')

    if step in (1, 5) and DEVICE == 'cuda':
        print(f'[MEM] step {step} alloc_peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB '
              f'reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
        if bpb < best_bpb:
            best_bpb = bpb
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema)
        val_bpbs.append(bpb)
        val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}'])
        csv_f.flush()
        temp_w.writerow([step] + [f'{getter():.6f}' for _, getter in temp_specs])
        temp_f.flush()
        _log_weight_deltas(step)
        model.train()

csv_f.close()
temp_f.close()
_weight_csv_f.close()
elapsed = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy loss', title='Training Loss')
ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, label='val bpb', color='red')
ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120)
plt.close(fig)

summary = dict(
    exp_name=cfg['exp_name'],
    best_val_bpb=best_bpb,
    final_val_bpb=val_bpbs[-1] if val_bpbs else float('nan'),
    n_params=n_params,
    training_time_hours=round(elapsed / 3600, 3),
)
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

ckpt_path = os.path.join(EXP_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lut_optimizer_state_dict': lut_optimizer.state_dict(),
    'config': cfg,
    'step': N_STEPS,
    'best_val_bpb': best_bpb,
    'final_val_bpb': val_bpbs[-1] if val_bpbs else float('nan'),
}, ckpt_path)
print(f'saved checkpoint -> {ckpt_path}')

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
