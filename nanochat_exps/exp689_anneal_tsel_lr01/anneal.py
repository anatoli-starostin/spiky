"""exp689 — anneal T_sel on exp684 checkpoint for 1000 steps at 1% of peak LR.

Sibling of exp688 (0.1% LR -> only 2 mb gain over naive collapse; weights
basically didn't move at 3e-7 LR). This run uses lr_scale = 0.01 (10x exp688)
so the weights can actually adapt to the collapsing forward while T_sel
anneals log-linearly from learned ~0.5 down to 0.001.

  - lr_scale = 0.01 (1% of peak), constant for the whole anneal phase
  - log-linear anneal of every log_select_temp, ~0.5 -> 0.001 over 1000 steps
  - log_select_temp is excluded from both optimizers (driven externally)
  - log_soft_score_temp stays learnable (AdamW updates it at 1% peak LR)
  - Forward = hybrid_smooth (2-row blend); blend -> single-row as T_sel -> 0.

End evals (same as exp688):
  - SOFT eval: hybrid_smooth fwd at final low T_sel (~= hard already).
  - HARD eval: monkey-patched _soft_lut_fwd_body_einsum (pure argmax).
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
sys.path.insert(0, '/home/starost/spiky/src')

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.tiny_multi_head_lut import (
    TinyMultiHeadLut, _soft_lut_fwd_body_einsum,
)
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = '/home/starost/spiky/nanochat_exps/exp684_dual_stream'
with open(os.path.join(SRC_DIR, 'config.json')) as f:
    cfg = json.load(f)

# --- anneal-phase hyperparams ------------------------------------------------
ANNEAL_STEPS = 1000
ANNEAL_LR_SCALE = 0.01      # 1% of peak (10x exp688)
ANNEAL_T_SEL_FLOOR = 0.001  # target T_sel at the end of the anneal
EVAL_EVERY = 100
EVAL_STEPS_FINAL = 50

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'] + 1)  # different sampling seed from train

CONTEXT_SIZE = cfg['context_size']
E    = cfg['embedding_dim']
D    = cfg['residual_dim']
H    = cfg['n_heads']
d_qk = cfg['d_qk']
d_v  = cfg['d_v']
N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']
TOTAL_BS  = cfg['total_batch_size']
_ROPE_BASE = cfg.get('rope_base', 10000.0)
_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
print(f'Loading tokenizer from {os.path.join(BASE_DIR, "tokenizer")}')
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab: {VOCAB_SIZE}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE,
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE,
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories (kwargs identical to exp684 train) -------------------------
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

def _make_qk(layer_idx):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_v(layer_idx):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + 200 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_out(layer_idx):
    return TinyMultiHeadLut(input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + 400 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_residual_lut(layer_idx):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + 600 + layer_idx, device=DEVICE, **_TINY_SOFT_KWARGS)


# --- RoPE / norms / block / model (mirror exp684) -----------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


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
        self.v_lut        = _make_v(layer_idx)
        self.out_proj     = _make_out(layer_idx)
        self.residual_lut = _make_residual_lut(layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E)
        self.ln_resid = MeanAbsNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre)
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_vec = self.v_lut(x_pre)
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)
        x_lut_next_flat = x_flat + out_e
        r_in = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_resid = torch.zeros(B, T, D, device=x_lut.device, dtype=x_lut.dtype)
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


# --- Build + load checkpoint --------------------------------------------------
print('Building model...')
model = Model().to(DEVICE)
ckpt_path = os.path.join(SRC_DIR, 'checkpoint.pt')
print(f'Loading exp684 checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'  loaded step={ckpt["step"]}, final_val_bpb(soft hybrid)={ckpt["final_val_bpb"]:.4f}')

n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params/1e6:.2f}M')


# --- Capture initial log_select_temp per LUT ----------------------------------
tsel_params = []
tsel_names  = []
tsel_init_log = []
for li, blk in enumerate(model.layers):
    for slut_name in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
        mod = getattr(blk, slut_name, None)
        if mod is None or not getattr(mod, 'learnable_temps', False):
            continue
        p = mod.log_select_temp
        tsel_params.append(p)
        tsel_names.append(f'L{li}.{slut_name}.log_select_temp')
        tsel_init_log.append(float(p.detach().item()))
        p.requires_grad_(False)  # we drive externally
print(f'Captured {len(tsel_params)} log_select_temp params (initial T_sel range: '
      f'{math.exp(min(tsel_init_log)):.4f}..{math.exp(max(tsel_init_log)):.4f})')

LOG_FLOOR = math.log(ANNEAL_T_SEL_FLOOR)


def set_tsel_at_step(step):
    """Log-linear anneal: log_T(t) = log_T_init + (log_floor - log_T_init) * t/N."""
    progress = step / ANNEAL_STEPS
    progress = min(max(progress, 0.0), 1.0)
    with torch.no_grad():
        for p, log_init in zip(tsel_params, tsel_init_log):
            new_log = log_init + (LOG_FLOOR - log_init) * progress
            p.fill_(new_log)


# --- Optimisers (excluding log_select_temp) -----------------------------------
lut_params, tok_emb_params, decay_params, nodecay_params = [], [], [], []
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
    dict(params=decay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
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

print(f'LUT optimizer = {_LUT_OPT} (peak lr={_LUT_LR}) | non-LUT AdamW peak lr={cfg["adam_lr"]} | anneal lr_scale={ANNEAL_LR_SCALE}')
print(f'lut_params={sum(p.numel() for p in lut_params):,} | decay={sum(p.numel() for p in decay_params):,} | tok_emb={sum(p.numel() for p in tok_emb_params):,} | nodecay_other={sum(p.numel() for p in nodecay_params):,}')


# --- Tracking -----------------------------------------------------------------
metrics_path = os.path.join(EXP_DIR, 'metrics.csv')
metrics_f = open(metrics_path, 'w', newline='')
metrics_w = csv.writer(metrics_f)
metrics_w.writerow(['step', 'train_loss', 'val_bpb', 'T_sel_mean', 'T_sel_min', 'T_sel_max'])

tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'tokens/micro-batch={tokens_per_step:,} | grad_accum={grad_accum} | effective batch={grad_accum*tokens_per_step:,}')

# Apply T_sel at step 0
set_tsel_at_step(0)
t_vals = [math.exp(p.item()) for p in tsel_params]
print(f'Step 0: T_sel mean={sum(t_vals)/len(t_vals):.4f}, min={min(t_vals):.4f}, max={max(t_vals):.4f}')

# --- Eval baseline (re-eval exp684 at hybrid_smooth with current T_sel) -------
print('\n--- Baseline (re-eval exp684 weights, T_sel unchanged) ---')
model.eval()
vl = val_loader_factory()
t0 = time.time()
bpb_baseline = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (hybrid_smooth, T_sel @ trained values) = {bpb_baseline:.4f}   [{time.time()-t0:.1f}s]')

# --- Anneal loop --------------------------------------------------------------
model.train()
ema = None
t0_train = time.time()
for step in range(1, ANNEAL_STEPS + 1):
    # set T_sel for THIS step (use step / N progress at the start of the step)
    set_tsel_at_step(step)

    for o in all_optimizers:
        for g in o.param_groups:
            g['lr'] = g['initial_lr'] * ANNEAL_LR_SCALE

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

    if step % 50 == 0 or step == 1:
        t_vals = [math.exp(p.item()) for p in tsel_params]
        print(f'step {step:4d} | loss={ema:.4f} | T_sel mean={sum(t_vals)/len(t_vals):.5f} '
              f'min={min(t_vals):.5f} max={max(t_vals):.5f} | lr={ANNEAL_LR_SCALE*cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == ANNEAL_STEPS:
        model.eval()
        vl = val_loader_factory()
        bpb = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
        t_vals = [math.exp(p.item()) for p in tsel_params]
        tmean = sum(t_vals) / len(t_vals)
        tmin, tmax = min(t_vals), max(t_vals)
        print(f'[VAL] step {step}: bpb={bpb:.4f} | T_sel mean={tmean:.5f} ({tmin:.5f}..{tmax:.5f})')
        metrics_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}', f'{tmean:.6f}', f'{tmin:.6f}', f'{tmax:.6f}'])
        metrics_f.flush()
        model.train()

metrics_f.close()
train_time = time.time() - t0_train
print(f'\nAnneal training time: {train_time:.1f}s')


# --- Final evals: SOFT (T_sel ~ 0.001) and HARD (single-row) ------------------
print(f'\n=== Final eval (EVAL_STEPS={EVAL_STEPS_FINAL}) ===')

print('\n--- SOFT eval (hybrid_smooth fwd, T_sel ~ 0.001 -> ~hard) ---')
model.eval()
vl = val_loader_factory()
t0 = time.time()
bpb_soft = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (soft, annealed) = {bpb_soft:.4f}   [{time.time()-t0:.1f}s]')


def hard_forward_factory(mod):
    """Pure single-row argmax forward via _soft_lut_fwd_body_einsum."""
    def hard_forward(x):
        if x.dim() != 2 or x.shape[1] != mod.input_dim:
            raise ValueError(f'x shape must be [B,{mod.input_dim}], got {tuple(x.shape)}')
        T_soft = mod.log_soft_score_temp.exp()
        autocast_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                        if mod.use_bf16 and x.is_cuda
                        else torch.amp.autocast('cpu', enabled=False))
        with autocast_ctx:
            out, _ = _soft_lut_fwd_body_einsum(
                x, mod.weights, mod.soft_anchor_a_long, mod.soft_anchor_b_long,
                mod.soft_bit_matrix, T_soft,
                mod.n_heads, mod.tables_per_head, mod.table_dim,
            )
        return out
    return hard_forward

print('\n--- HARD eval (argmax single row, coeff=1.0) ---')
n_patched = 0
for m in model.modules():
    if isinstance(m, TinyMultiHeadLut):
        m.forward = hard_forward_factory(m)
        n_patched += 1
print(f'  patched {n_patched} TinyMHLut modules')
vl = val_loader_factory()
t0 = time.time()
bpb_hard = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (hard)           = {bpb_hard:.4f}   [{time.time()-t0:.1f}s]')


# --- Report -------------------------------------------------------------------
print('\n========================================================')
print(f'  exp684 final (train-mode, hybrid_smooth)   : 1.4262 bpb')
print(f'  exp684 reread @ EVAL_STEPS={EVAL_STEPS_FINAL}             : {bpb_baseline:.4f} bpb')
print(f'  exp684 collapsed to hard (no retraining)   : 1.4529 bpb  (validate_hard.py earlier)')
print(f'  exp687 trained-from-scratch hard           : 1.4643 bpb')
print(f'  --- ANNEAL FROM exp684 (this run) ---')
print(f'  SOFT eval (hybrid_smooth, T_sel ~ 0.001)   : {bpb_soft:.4f} bpb')
print(f'  HARD eval (argmax single row, coeff=1.0)   : {bpb_hard:.4f} bpb')
print(f'  delta (hard - soft annealed)               : {bpb_hard - bpb_soft:+.4f}')
print(f'  delta (hard - baseline-collapse 1.4529)    : {bpb_hard - 1.4529:+.4f}')
print(f'  delta (hard - exp687 1.4643)               : {bpb_hard - 1.4643:+.4f}')
print('========================================================')

summary = dict(
    exp_name='exp689_anneal_tsel_lr01',
    baseline_bpb=float(bpb_baseline),
    final_soft_bpb=float(bpb_soft),
    final_hard_bpb=float(bpb_hard),
    delta_hard_vs_soft=float(bpb_hard - bpb_soft),
    delta_hard_vs_exp684_collapse=float(bpb_hard - 1.4529),
    delta_hard_vs_exp687_scratch=float(bpb_hard - 1.4643),
    anneal_steps=ANNEAL_STEPS,
    anneal_lr_scale=ANNEAL_LR_SCALE,
    anneal_t_sel_floor=ANNEAL_T_SEL_FLOOR,
    train_time_seconds=train_time,
    n_params=n_params,
)
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'wrote {os.path.join(EXP_DIR, "summary.json")}')

ckpt_save = os.path.join(EXP_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'anneal_steps': ANNEAL_STEPS,
    'final_soft_bpb': float(bpb_soft),
    'final_hard_bpb': float(bpb_hard),
}, ckpt_save)
print(f'saved checkpoint -> {ckpt_save}')

print('\n=== DONE ===')
