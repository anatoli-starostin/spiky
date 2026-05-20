"""nanochat_exps/exp407_clean_v_nap6_bs16 — cleaned-up fork of exp392.

LUT-LM with RoPE, dual-stream residual, untied unembedder.

Per-layer block:
  - qkv_lut       TinyMHLut(soft, NAP=6, tph=16)  -> n_out = 2*d_qk + d_v
                  Q, K come exclusively from qkv_lut. Last d_v outputs are
                  ADDED to v_lut's output (shallow shared-anchor v branch).
  - v_lut         TinyMHLut(soft, NAP=6, tph=128) -> d_v
                  Trade vs exp392: NAP=8 tph=32 -> NAP=6 tph=128 (param-matched).
  - q_norm, k_norm LayerNorm(d_qk)
  - RoPE on (q, k) before SDPA
  - SDPA causal
  - out_proj      TinyMHLut(soft, NAP=6, tph=512) -> E
                  Plain TinyMHLut (no multi-nap wrapper, single-component case
                  is functionally identical).
  - ln_e          LayerNorm(E)  -- post-norm on E-stream output
  - residual_lut  TinyMHLut(soft, NAP=6, tph=64)  -> D

Model loop:
  x_lut  starts from tok_emb_E.
  x_resid starts at zeros, accumulates residual_lut output per layer.
  Final ln_final(D), then untied unembedder Linear(D, V).

All LUTs use backward_mode='soft' with learnable temperatures and bf16.
"""
import sys, os, json, math, time, csv, re
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
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories (all TinyMHLut soft) ---------------------------------------
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS,
)

def _make_qkv_joint(layer_idx, seed_offset):
    kwargs = dict(_TINY_SOFT_KWARGS)
    kwargs['initial_weights_noise'] = cfg.get('qkv_lut_init_std',
                                              cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=2 * d_qk + d_v,
        n_anchor_pairs=cfg['qkv_input_nap'],
        tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **kwargs,
    )

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'],
        tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_out(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=H * d_v,
        n_heads=1,
        n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'],
        tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )

def _make_residual_lut(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=1,
        n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'],
        tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
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


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qkv_lut      = _make_qkv_joint(layer_idx, layer_idx)
        self.v_lut        = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj     = _make_out(layer_idx, 400 + layer_idx)
        self.residual_lut = _make_residual_lut(layer_idx, 600 + layer_idx)

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        # Pre-norm BEFORE qkv/v LUTs (GPT-2 style); E-stream residual after.
        self.ln_pre  = nn.LayerNorm(E)
        # Post-norm on the residual-updated E-stream, feeds residual_lut.
        self.ln_post = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        # Pre-norm BEFORE qkv/v LUTs (GPT-2 style pre-norm).
        x_pre = self.ln_pre(x_flat)

        qkv_out = self.qkv_lut(x_pre)                   # [B*T, H, 2*d_qk + d_v]
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
        v_branch = qkv_out[..., 2 * d_qk:]              # [B*T, H, d_v]
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_lut_out = self.v_lut(x_pre)                   # [B*T, H, d_v]
        v_vec = v_lut_out + v_branch                    # additive shallow v branch
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)       # [B*T, E]

        # E-stream residual: identity-skip around the attention/LUT block.
        x_lut_next_flat = x_flat + out_e                # [B*T, E]
        x_lut_next = x_lut_next_flat.reshape(B, T, E)

        # Post-norm fed into residual_lut (the D-stream contribution).
        r_in   = self.ln_post(x_lut_next_flat)          # [B*T, E]
        r_out  = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next, r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut   = self.tok_emb_E(tokens)
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

_LUT_LR = cfg.get('lut_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=lut_params,     lr=_LUT_LR,        betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
print(
    f'optimizer groups: '
    f'lut={sum(p.numel() for p in lut_params):,} (wd=0, lr={_LUT_LR}) | '
    f'decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'tok_emb={sum(p.numel() for p in tok_emb_params):,} (wd=0) | '
    f'nodecay_other={sum(p.numel() for p in nodecay_params):,} (wd=0)'
)
optimizer = torch.optim.AdamW(adam_groups)
for g in optimizer.param_groups:
    g['initial_lr'] = g['lr']

print(f'D=residual_dim={D}, E=embedding_dim={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'qkv_lut      TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["qkv_input_nap"]} tph={cfg["qkv_tph"]} n_out=2*d_qk+d_v={2*d_qk+d_v}  [q,k from this; last d_v added to v]')
print(f'v_lut        TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v}')
print(f'out_proj     TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["out_input_nap"]} tph={cfg["out_tph"]} n_out=E={E}')
print(f'residual_lut TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["residual_input_nap"]} tph={cfg["residual_tph"]} n_out=D={D}  [E -> D]')
print(f'UNTIED unembedder Linear(D={D}, V={VOCAB_SIZE}); tok_emb_E at E={E}; ln_final(D) before unembed; RoPE base={_ROPE_BASE}')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(blk, slut_name)
            if getattr(mod, 'learnable_temps', False):
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


# --- Effective (real) Adam learning-rate tracking -----------------------------
# Adam's per-parameter step is  lr * m_hat / (sqrt(v_hat) + eps), where the
# factor f = m_hat/(sqrt(v_hat)+eps) is in roughly [-1, 1]: |f|~1 when the
# gradient is consistent (full nominal lr is realized), |f|->0 when the gradient
# is noisy / the param is rarely hit (the realized step is damped). We log, per
# logical module group:
#   nominal_lr     = group base lr * cosine/warmup scale
#   adam_factor_rms= rms over all elements of |m_hat/(sqrt(v_hat)+eps)|  (in [0,1])
#   eff_step_rms   = nominal_lr * adam_factor_rms   (the REAL per-step displacement RMS)
def _param_group_name(n):
    if 'qkv_lut.weights' in n:      return 'qkv_lut'
    if 'v_lut.weights' in n:        return 'v_lut'
    if 'out_proj.weights' in n:     return 'out_proj'
    if 'residual_lut.weights' in n: return 'residual_lut'
    if n.startswith('unembedder'):  return 'unembedder'
    if n.startswith('tok_emb_E'):   return 'tok_emb'
    if 'log_soft_score_temp' in n or 'log_select_temp' in n: return 'temps'
    if '_norm.' in n or n.startswith('ln_') or '.ln_' in n:  return 'norms'
    return 'other'

_LAYER_RE = re.compile(r'layers\.(\d+)\.')
def _param_layer(n):
    m = _LAYER_RE.search(n)
    return int(m.group(1)) if m else -1   # -1 = global (unembedder, tok_emb, ln_final)

_NAME_BY_PARAM = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
_LUT_GROUP_NAMES = {'qkv_lut', 'v_lut', 'out_proj', 'residual_lut'}
_eff_csv_path = os.path.join(EXP_DIR, 'effective_lr.csv')
_eff_csv_f = open(_eff_csv_path, 'w', newline='')
_eff_csv_w = csv.writer(_eff_csv_f)
_eff_csv_w.writerow(['step', 'layer', 'group', 'nominal_lr', 'adam_factor_rms', 'eff_step_rms', 'n_params'])
print(f'Tracking effective Adam lr per (layer, module group) in effective_lr.csv')

def _log_effective_lr(step_):
    # accumulate sum(f^2) and element count per (layer, logical group)
    sq = {}; cnt = {}
    with torch.no_grad():
        for grp in optimizer.param_groups:
            b1, b2 = grp['betas']; eps = grp['eps']
            for p in grp['params']:
                st = optimizer.state.get(p)
                if not st or 'exp_avg' not in st:
                    continue
                t = float(st['step']) if not torch.is_tensor(st['step']) else float(st['step'].item())
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                m_hat = st['exp_avg'] / bc1
                v_hat = st['exp_avg_sq'] / bc2
                f = (m_hat / (v_hat.sqrt() + eps))
                name = _NAME_BY_PARAM[id(p)]
                key = (_param_layer(name), _param_group_name(name))
                sq[key]  = sq.get(key, 0.0)  + float((f * f).sum())
                cnt[key] = cnt.get(key, 0)   + f.numel()
    scale = get_lr_scale(step_)
    for key in sorted(sq):
        layer, g = key
        base = _LUT_LR if g in _LUT_GROUP_NAMES else cfg['adam_lr']
        nominal = base * scale
        factor_rms = (sq[key] / cnt[key]) ** 0.5
        _eff_csv_w.writerow([step_, layer, g, f'{nominal:.6e}', f'{factor_rms:.6e}',
                             f'{nominal * factor_rms:.6e}', cnt[key]])
    _eff_csv_f.flush()


# --- LUT effective-lr variance decomposition ----------------------------------
# For each LUT module the weight tensor is [n_tables, K=2^NAP, n_outputs], and the
# Adam step magnitude |f| = |m_hat/(sqrt(v_hat)+eps)| is per-element. We report the
# spread of the REAL lr at three NESTED scopes (CV = std/mean, scale-free):
#   inside_entry : spread across the n_outputs of a single row  (avg over t,k)
#   inside_table : spread across all K*n_outputs of a single table (avg over t)
#   inside_lut   : spread across the whole module's elements
# and the exact nested ANOVA decomposition of total variance (sums to 1):
#   f_within_entry  : E_{t,k}[Var_o]                 (variance inside rows)
#   f_between_row   : E_t[Var_k(E_o)]                (hot vs cold rows in a table)
#   f_between_table : Var_t(E_{k,o})                 (table-to-table)
_lutvar_csv_path = os.path.join(EXP_DIR, 'lut_lr_variance.csv')
_lutvar_csv_f = open(_lutvar_csv_path, 'w', newline='')
_lutvar_csv_w = csv.writer(_lutvar_csv_f)
_lutvar_csv_w.writerow(['step', 'layer', 'module', 'mean_abs_f',
                        'cv_inside_entry', 'cv_inside_table', 'cv_inside_lut',
                        'frac_within_entry', 'frac_between_row', 'frac_between_table'])
print('Tracking LUT effective-lr variance decomposition in lut_lr_variance.csv')

def _log_lut_lr_variance(step_):
    with torch.no_grad():
        for grp in optimizer.param_groups:
            b1, b2 = grp['betas']; eps = grp['eps']
            for p in grp['params']:
                name = _NAME_BY_PARAM[id(p)]
                if '.weights' not in name or p.ndim != 3:
                    continue
                st = optimizer.state.get(p)
                if not st or 'exp_avg' not in st:
                    continue
                t = float(st['step']) if not torch.is_tensor(st['step']) else float(st['step'].item())
                m_hat = st['exp_avg'] / (1.0 - b1 ** t)
                v_hat = st['exp_avg_sq'] / (1.0 - b2 ** t)
                f = (m_hat / (v_hat.sqrt() + eps)).abs().float()   # [T, K, O]
                mod = _param_group_name(name)
                mean = float(f.mean())
                # nested CVs (std/mean) at each scope
                cv_entry = float(f.std(dim=2, unbiased=False).mean()) / (mean + 1e-12)
                T = f.shape[0]
                cv_table = float(f.reshape(T, -1).std(dim=1, unbiased=False).mean()) / (mean + 1e-12)
                cv_lut   = float(f.std(unbiased=False)) / (mean + 1e-12)
                # exact ANOVA decomposition (uniform group sizes -> additive)
                v_within_entry  = float(f.var(dim=2, unbiased=False).mean())
                v_between_row   = float(f.mean(dim=2).var(dim=1, unbiased=False).mean())
                v_between_table = float(f.mean(dim=(1, 2)).var(unbiased=False))
                tot = v_within_entry + v_between_row + v_between_table + 1e-30
                _lutvar_csv_w.writerow([step_, _param_layer(name), mod, f'{mean:.6e}',
                                        f'{cv_entry:.4f}', f'{cv_table:.4f}', f'{cv_lut:.4f}',
                                        f'{v_within_entry/tot:.4f}', f'{v_between_row/tot:.4f}',
                                        f'{v_between_table/tot:.4f}'])
        _lutvar_csv_f.flush()


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
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    optimizer.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    optimizer.step()

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')

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
        _log_effective_lr(step)
        _log_lut_lr_variance(step)
        model.train()

csv_f.close()
temp_f.close()
_weight_csv_f.close()
_eff_csv_f.close()
_lutvar_csv_f.close()
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

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
