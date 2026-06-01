"""exp691 — take exp684 weights, switch to hard forward at step 0, 1K steps at 1% LR.

No T_sel anneal. Just instantly swap `backward_mode='hybrid_smooth'` for
`backward_mode='soft'` (which gives the same single-row argmax forward as the
hard-eval kernel) and finetune for 1K steps. Direct test of: does the anneal
trajectory matter, or can the optimizer adapt straight from a hard cliff?

  - lr_scale = 0.01 (1% of peak — exp689's sweet spot)
  - forward: _soft_lut_fwd_body_einsum (single-row argmax, coeff=1.0)
  - backward: soft K-row attribution to x, hard index_add to W
  - log_soft_score_temp + log_select_temp stay learnable
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F

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

FINETUNE_STEPS = 1000
LR_SCALE = 0.01
EVAL_EVERY = 100
EVAL_STEPS_FINAL = 50

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'] + 2)

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


BASE_DIR = get_base_dir()
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


# KEY CHANGE vs exp684/688/689/690: backward_mode 'hybrid_smooth' -> 'soft'.
# 'soft' forward = single-row argmax (hard); 'soft' backward = K-row soft attr.
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',                              # <-- the switch
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS,
)

def _make_qk(li):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + li, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_v(li):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + 200 + li, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_out(li):
    return TinyMultiHeadLut(input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + 400 + li, device=DEVICE, **_TINY_SOFT_KWARGS)
def _make_residual_lut(li):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + 600 + li, device=DEVICE, **_TINY_SOFT_KWARGS)


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
    def __init__(self, li):
        super().__init__()
        self.qk_lut       = _make_qk(li)
        self.v_lut        = _make_v(li)
        self.out_proj     = _make_out(li)
        self.residual_lut = _make_residual_lut(li)
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


print('Building model with backward_mode=soft (hard fwd + soft bwd)...')
model = Model().to(DEVICE)
ckpt_path = os.path.join(SRC_DIR, 'checkpoint.pt')
print(f'Loading exp684 checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
# state_dict keys match: log_soft_score_temp and log_select_temp exist in both
# hybrid_smooth and soft modes (same parameter names). T_soft is used in the
# soft forward; T_sel only affects the backward (no anneal here).
model.load_state_dict(ckpt['model_state_dict'])
n_params = sum(p.numel() for p in model.parameters())
print(f'  loaded step={ckpt["step"]}, exp684 final_val_bpb(hybrid_smooth)={ckpt["final_val_bpb"]:.4f}')
print(f'Params: {n_params/1e6:.2f}M')


# --- Optimisers (no log_select_temp freezing here — let everything train) -----
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

print(f'LUT={_LUT_OPT} (peak lr={_LUT_LR}) | non-LUT AdamW peak lr={cfg["adam_lr"]} | LR scale={LR_SCALE}')
print(f'lut={sum(p.numel() for p in lut_params):,} | decay={sum(p.numel() for p in decay_params):,} | tok_emb={sum(p.numel() for p in tok_emb_params):,} | nodecay_other={sum(p.numel() for p in nodecay_params):,}')


metrics_path = os.path.join(EXP_DIR, 'metrics.csv')
metrics_f = open(metrics_path, 'w', newline='')
metrics_w = csv.writer(metrics_f)
metrics_w.writerow(['step', 'train_loss', 'val_bpb'])

tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'tokens/micro-batch={tokens_per_step:,} | grad_accum={grad_accum} | effective batch={grad_accum*tokens_per_step:,}')


# --- Eval baseline: hard forward, exp684 weights, no finetune ---
print('\n--- Baseline (exp684 weights, hard fwd via soft backward_mode, no finetune) ---')
model.eval()
vl = val_loader_factory()
t0 = time.time()
bpb_baseline = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (step=0, hard) = {bpb_baseline:.4f}   [{time.time()-t0:.1f}s]')


model.train()
ema = None
t0_train = time.time()
for step in range(1, FINETUNE_STEPS + 1):
    for o in all_optimizers:
        for g in o.param_groups:
            g['lr'] = g['initial_lr'] * LR_SCALE
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
        print(f'step {step:4d} | loss={ema:.4f} | lr={LR_SCALE*cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == FINETUNE_STEPS:
        model.eval()
        vl = val_loader_factory()
        bpb = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        metrics_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}'])
        metrics_f.flush()
        model.train()

metrics_f.close()
train_time = time.time() - t0_train
print(f'\nFinetune time: {train_time:.1f}s')


# --- Final eval (soft mode already = hard; eval with patched fwd just to confirm) ---
print(f'\n=== Final eval (EVAL_STEPS={EVAL_STEPS_FINAL}) ===')
model.eval()
vl = val_loader_factory()
t0 = time.time()
bpb_final_softmode = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (soft backward_mode, hard fwd) = {bpb_final_softmode:.4f}   [{time.time()-t0:.1f}s]')


def hard_forward_factory(mod):
    def hard_forward(x):
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

n_patched = 0
for m in model.modules():
    if isinstance(m, TinyMultiHeadLut):
        m.forward = hard_forward_factory(m)
        n_patched += 1
print(f'  patched {n_patched} TinyMHLut modules to explicit hard forward')
vl = val_loader_factory()
t0 = time.time()
bpb_final_explicit = evaluate_bpb(model, vl, EVAL_STEPS_FINAL, token_bytes)
print(f'  val_bpb (explicit hard fwd)            = {bpb_final_explicit:.4f}   [{time.time()-t0:.1f}s]')


print('\n========================================================')
print(f'  exp684 baseline (hybrid_smooth 2-row blend) : 1.4190')
print(f'  exp684 naive collapse to hard               : 1.4529')
print(f'  exp687 trained from scratch (8K, hard)      : 1.4643')
print(f'  exp688 anneal lr=0.1% (1K)                  : 1.4505')
print(f'  exp689 anneal lr=1%   (1K)                  : 1.4479 [prior best]')
print(f'  exp690 anneal lr=10%  (1K)                  : 1.4528')
print(f'  --- this run ---')
print(f'  step 0 baseline (hard, no finetune)         : {bpb_baseline:.4f}')
print(f'  final soft-mode (hard fwd, soft bwd)        : {bpb_final_softmode:.4f}')
print(f'  final explicit hard fwd                     : {bpb_final_explicit:.4f}')
print(f'  delta (final - baseline)                    : {bpb_final_explicit - bpb_baseline:+.4f}')
print(f'  delta (final - exp689 1.4479)               : {bpb_final_explicit - 1.4479:+.4f}')
print('========================================================')

summary = dict(
    exp_name='exp691_finetune_hard',
    baseline_bpb=float(bpb_baseline),
    final_softmode_bpb=float(bpb_final_softmode),
    final_explicit_hard_bpb=float(bpb_final_explicit),
    delta_vs_baseline=float(bpb_final_explicit - bpb_baseline),
    delta_vs_exp689=float(bpb_final_explicit - 1.4479),
    finetune_steps=FINETUNE_STEPS,
    lr_scale=LR_SCALE,
    train_time_seconds=train_time,
    n_params=n_params,
)
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'wrote {os.path.join(EXP_DIR, "summary.json")}')

torch.save({
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'finetune_steps': FINETUNE_STEPS,
    'final_softmode_bpb': float(bpb_final_softmode),
    'final_explicit_hard_bpb': float(bpb_final_explicit),
}, os.path.join(EXP_DIR, 'checkpoint.pt'))
print('saved checkpoint')

print('\n=== DONE ===')
