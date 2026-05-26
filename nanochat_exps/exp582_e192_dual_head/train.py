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

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut, MatmulMultiHeadLut
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

# exp508: qk_lut is the exp493 SOFT MatmulMultiHeadLut(gate_mode='softmax'), now
# PURE q,k (n_out=2*d_qk, NO additive v-branch). All other modules stay TinyMHLut(soft argmax).
_MATMUL_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    gate_mode=cfg.get('qkv_gate_mode', 'softmax'),
    use_bias=False,
)

def _make_qk(layer_idx, seed_offset):
    # exp511: HARD argmax TinyMHLut (soft backward) on LION, like the other 3 modules.
    # nap=2, tph=1024 = param-matched reshape of exp508's nap=6, tph=64 (2^nap*tph = 4096).
    return TinyMultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=2 * d_qk,                    # q,k only -- no v-branch
        n_anchor_pairs=cfg['qkv_input_nap'],
        tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        **_TINY_SOFT_KWARGS,
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


class MeanAbsNorm(nn.Module):
    """exp475: cheapest magnitude norm = x / (mean(|x|) + eps). L1 mean-absolute,
    NO centering, NO affine, NO square/sqrt (so NOT root-mean-square -- this is
    deliberately not RMSNorm). Completes the 2x2 (L1/L2 x center/no-center):
    exp472 std (L2+center), exp474 RMS (L2 no-center), exp473 MAD (L1+center),
    this = L1 no-center. For the LUT mu cancels in differences -> per-token scalar
    mean|x| absorbed by temperature T; expect ~= exp453. Most matmul-free-friendly
    (only abs/mean/divide)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_lut       = _make_qk(layer_idx, layer_idx)
        self.v_lut        = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj     = _make_out(layer_idx, 400 + layer_idx)
        # exp571: residual_lut REMOVED — no D-stream at all.

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        # Pre-norm only (no ln_post since residual_lut is gone).
        self.ln_pre  = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        # Pre-norm BEFORE qkv/v LUTs (GPT-2 style pre-norm).
        x_pre = self.ln_pre(x_flat)

        qk_out = self.qk_lut(x_pre)                     # [B*T, H, 2*d_qk] -- q,k only
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = self.v_lut(x_pre)                       # [B*T, H, d_v] -- no additive v-branch
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)       # [B*T, E]

        # E-stream residual: identity-skip around the attention/LUT block.
        x_lut_next_flat = x_flat + out_e                # [B*T, E]
        x_lut_next = x_lut_next_flat.reshape(B, T, E)
        return x_lut_next


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        # Tied dot head on E-stream (exp574 recipe).
        self.ln_final = nn.LayerNorm(E)
        self.ln_emb = nn.LayerNorm(E)
        self.ln_emb.weight.data.fill_(0.1)
        # exp582: ONE final residual_lut after layer 6 (NOT per-layer), projecting
        # E -> D, plus a Linear(D, V) unembedder. Output logits are the SUM of
        # (tied dot on E-stream) + (Linear on the D-stream after final_residual_lut).
        # Tests whether a single final LUT projection + linear head can recover
        # what the full per-layer residual_lut + Linear head pair (exp567) does.
        self.ln_pre_final = MeanAbsNorm(E)
        self.final_residual_lut = TinyMultiHeadLut(
            input_dim=E,
            n_heads=1,
            n_outputs=D,
            n_anchor_pairs=cfg['residual_input_nap'],
            tables_per_head=cfg['residual_tph'],
            random_seed=cfg['random_seed'] + 800,
            device=DEVICE,
            **_TINY_SOFT_KWARGS,
        )
        self.ln_final_d = nn.LayerNorm(D)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut = layer(x_lut, self.rope.cos, self.rope.sin)
        # Tied dot head on E-stream (exp574 recipe).
        x_normed   = self.ln_final(x_lut)
        emb_normed = self.ln_emb(self.tok_emb_E.weight)
        logits_tied = x_normed @ emb_normed.t()
        # exp582: parallel Linear head from one final residual_lut projection E -> D.
        x_pre_final = self.ln_pre_final(x_lut.reshape(B * T, E))         # [B*T, E]
        x_d         = self.final_residual_lut(x_pre_final).squeeze(1)     # [B*T, D]
        x_d         = self.ln_final_d(x_d).reshape(B, T, D)
        logits_lin  = self.unembedder(x_d)                                # [B, T, V]
        logits = logits_tied + logits_lin
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
        # exp511: ALL LUT modules (incl. qk_lut now hardened to argmax TinyMHLut) are
        # SPARSE-gradient -> LION, like exp475. No more soft-qk-on-AdamW split.
        lut_params.append(p)
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)

# ---- LUT-specific optimizers (sign-based) ------------------------------------
class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum. update = -lr*sign(b1*m + (1-b1)*g); m = b2*m + (1-b2)*g.
    Single state tensor; full-magnitude (unit) sign step per element."""
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

class SignSGD(torch.optim.Optimizer):
    """Sign-SGD with momentum (Signum). buf = mu*buf + (1-mu)*g; update = -lr*sign(buf).
    mu=0 -> pure signSGD (sign of raw gradient)."""
    def __init__(self, params, lr=2e-4, momentum=0.9, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, mu, wd = grp['lr'], grp['momentum'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad if wd == 0 else p.grad.add(p, alpha=wd)
                if mu > 0:
                    st = self.state[p]
                    if 'momentum_buffer' not in st:
                        st['momentum_buffer'] = torch.zeros_like(p)
                    buf = st['momentum_buffer']
                    buf.mul_(mu).add_(g, alpha=1.0 - mu)
                    d = buf.sign()
                else:
                    d = g.sign()
                p.add_(d, alpha=-lr)

_LUT_LR  = cfg.get('lut_lr', cfg['adam_lr'])
_LUT_OPT = cfg.get('lut_optimizer', 'adamw')   # 'adamw' | 'lion' | 'signsgd'

# Non-LUT params always on AdamW (unchanged from exp428).
adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)

# LUT params on the chosen optimizer (separate so we can swap it independently).
if _LUT_OPT == 'lion':
    lut_optimizer = Lion([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                         lr=_LUT_LR, betas=tuple(cfg.get('lut_betas', (0.9, 0.99))))
elif _LUT_OPT == 'signsgd':
    lut_optimizer = SignSGD([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                            lr=_LUT_LR, momentum=cfg.get('lut_momentum', 0.9))
else:  # 'adamw' — reproduces exp428 exactly
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
print(f'qk_lut       TinyMHLut(soft argmax, LION) [exp511 q,k only, HARD]: in_nap={cfg["qkv_input_nap"]} tph={cfg["qkv_tph"]} n_out=2*d_qk={2*d_qk}  [no v-branch]')
print(f'v_lut        TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} d_v={d_v}')
print(f'out_proj     TinyMHLut(soft, noise_eps={_NOISE_EPS}): in_nap={cfg["out_input_nap"]} tph={cfg["out_tph"]} n_out=E={E}')
print(f'final_residual_lut (exp582, ONE instance, post-layer-6): TinyMHLut(soft, in_nap={cfg["residual_input_nap"]} tph={cfg["residual_tph"]}, E={E}->D={D})')
print(f'DUAL head (exp582): logits = tied_dot(ln_final(x_lut), ln_emb(tok_emb_E)) + Linear(D={D}, V={VOCAB_SIZE})(ln_final_d(final_residual_lut(x_lut))); ln_emb.gamma INIT=0.1; RoPE base={_ROPE_BASE}')


# --- Temperature tracking -----------------------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qk_lut', 'v_lut', 'out_proj'):
            mod = getattr(blk, slut_name)
            if getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
    # exp582: also track the single post-layer-6 final_residual_lut.
    mod = model.final_residual_lut
    if getattr(mod, 'learnable_temps', False):
        specs.append(('final_residual_lut.T_soft',
                      (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
        specs.append(('final_residual_lut.T_sel',
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

# ---- save checkpoint (weights + optimizer state + config) --------------------
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
