"""Bench exp761 per-step wallclock at the real config and 49152 tok/step (eff bs=96).

Uses the exact train.py model build + optimizer (so per-step wall reflects the real run).
Skips the dataloader by feeding random tokens of shape (DEVICE_BS, CONTEXT_SIZE).
"""
import os, sys, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E = cfg['embedding_dim']
D = cfg['residual_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']
TOTAL_BS = cfg['total_batch_size']
VOCAB_SIZE = 32768
_ROPE_BASE = cfg['rope_base']

tokens_per_micro = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_micro)
print(f'device_bs={DEVICE_BS}  context={CONTEXT_SIZE}  tokens/micro={tokens_per_micro:,}')
print(f'eff bs={grad_accum * DEVICE_BS}  tokens/step={grad_accum * tokens_per_micro:,}  grad_accum={grad_accum}')

_WEIGHT_DTYPE = {'fp32': torch.float32, 'bf16': torch.bfloat16}[cfg['weight_dtype']]
_FAST_KWARGS = dict(
    forward_mode=cfg['forward_mode'],
    backward_mode=cfg['backward_mode'],
    weight_dtype=_WEIGHT_DTYPE,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg['mhlut_init_std'],
    soft_score_temp=cfg['soft_score_temp'],
    select_temp=cfg['select_temp'],
    learnable_temps=cfg['soft_learnable_temps'],
    use_bf16=cfg['soft_use_bf16'],
)

def _make_qk(s):
    return FastMultiHeadLUT(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                            n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_FAST_KWARGS)
def _make_v(s):
    return FastMultiHeadLUT(input_dim=E, n_heads=H, n_outputs=d_v,
                            n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_FAST_KWARGS)
def _make_out(s):
    return FastMultiHeadLUT(input_dim=H * d_v, n_heads=1, n_outputs=E,
                            n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_FAST_KWARGS)
def _make_residual_lut(s):
    return FastMultiHeadLUT(input_dim=E, n_heads=1, n_outputs=D,
                            n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_FAST_KWARGS)
def _make_emb_resid_lut(s):
    return FastMultiHeadLUT(input_dim=E, n_heads=1, n_outputs=D,
                            n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
                            random_seed=cfg['random_seed'] + s, device=DEVICE, **_FAST_KWARGS)

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
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)

class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)

class LUTBlock(nn.Module):
    def __init__(self, li):
        super().__init__()
        self.qk_lut = _make_qk(li)
        self.v_lut = _make_v(200 + li)
        self.out_proj = _make_out(400 + li)
        self.residual_lut = _make_residual_lut(600 + li)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E)
        self.ln_resid = MeanAbsNorm(E)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        x_pre = self.ln_pre(x_flat)
        qk_out = self.qk_lut(x_pre).float()
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_vec = self.v_lut(x_pre).float()
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e = self.out_proj(out_in).squeeze(1).float()
        x_lut_next = x_flat + out_e
        r_in = self.ln_resid(x_lut_next)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D).float()
        return x_lut_next.reshape(B, T, E), r_out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.emb_resid_lut = _make_emb_resid_lut(800)
        self.ln_emb_resid = MeanAbsNorm(E)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x_lut = self.tok_emb_E(tokens)
        x_emb_pre = self.ln_emb_resid(x_lut.reshape(B * T, E))
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D).float()
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits

model = Model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
lut_params = [p for p in model.parameters() if p.ndim >= 3]
non_lut = [p for p in model.parameters() if p.ndim < 3]
print(f'\nTotal params: {n_params:,}  ({n_params/1e6:.2f} M)')
print(f'  LUT (ndim>=3): {sum(p.numel() for p in lut_params):,}  ({sum(p.numel() for p in lut_params)/1e6:.2f} M)')
print(f'  Non-LUT:       {sum(p.numel() for p in non_lut):,}  ({sum(p.numel() for p in non_lut)/1e6:.2f} M)')

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
                st = self.state[p]
                is_low = p.dtype != torch.float32
                if 'exp_avg' not in st:
                    st['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    if is_low:
                        st['master'] = p.detach().to(torch.float32).clone()
                m = st['exp_avg']
                g_f = p.grad if p.grad.dtype == torch.float32 else p.grad.to(torch.float32)
                if is_low:
                    master = st['master']
                    if wd != 0:
                        master.mul_(1.0 - lr * wd)
                    update = (m * b1 + g_f * (1.0 - b1)).sign_()
                    master.add_(update, alpha=-lr)
                    m.mul_(b2).add_(g_f, alpha=1.0 - b2)
                    p.data.copy_(master)
                else:
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    update = (m * b1 + g_f * (1.0 - b1)).sign_()
                    p.add_(update, alpha=-lr)
                    m.mul_(b2).add_(g_f, alpha=1.0 - b2)

decay_params, nodecay_params, tok_emb_params = [], [], []
for name, p in model.named_parameters():
    if p.ndim >= 3:
        continue
    if name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)

adam_groups = [
    dict(params=decay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg['weight_decay']),
    dict(params=tok_emb_params + nodecay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)
lut_optimizer = Lion([dict(params=lut_params, lr=cfg['lut_lr'], weight_decay=0.0)],
                     lr=cfg['lut_lr'], betas=tuple(cfg['lut_betas']))
all_optimizers = [optimizer, lut_optimizer]

# Synthetic data (no dataloader)
torch.manual_seed(0)
xy_cached = []
for _ in range(grad_accum + 2):
    tokens = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, CONTEXT_SIZE), device=DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, CONTEXT_SIZE), device=DEVICE)
    xy_cached.append((tokens, targets))

def one_step():
    for o in all_optimizers:
        o.zero_grad()
    accum = 0.0
    for k in range(grad_accum):
        x, y = xy_cached[k]
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum += loss.item() / grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in all_optimizers:
        o.step()
    return accum

# Warmup
print(f'\nwarmup (5 steps)...')
for i in range(5):
    L = one_step()
torch.cuda.synchronize()
print(f'  last loss = {L:.4f}')
peak_warmup = torch.cuda.max_memory_allocated() / 1e9
print(f'  peak alloc (warmup) = {peak_warmup:.2f} GB')

# Measure
N_BENCH = 20
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
for i in range(N_BENCH):
    one_step()
torch.cuda.synchronize()
dt = time.time() - t0
sec_per_step = dt / N_BENCH
peak_run = torch.cuda.max_memory_allocated() / 1e9
print(f'\n=== benchmark over {N_BENCH} steps ===')
print(f'  sec/step           = {sec_per_step:.3f}')
print(f'  tokens/step        = {grad_accum * tokens_per_micro:,}')
print(f'  microseconds/token = {sec_per_step * 1e6 / (grad_accum * tokens_per_micro):.1f}')
print(f'  peak alloc (run)   = {peak_run:.2f} GB')
print()
print(f'projected 24K steps  = {sec_per_step * 24000 / 3600:.2f} h')
print(f'projected 16K steps  = {sec_per_step * 16000 / 3600:.2f} h')
