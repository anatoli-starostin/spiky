"""exp545: BitReadout head — the unembedder as a popcount.

exp513 backbone EXACTLY (residual LUTs -> x_resid D=384), but the Linear head is
replaced by BitReadoutHead: x_resid -> P anchor-sign bits p; each vocab token a
learned ±1 code b_v; logit(v) = p · b_v (binary dot, rank P). Forward is the exact
hard popcount (STE: soft-sign backward to x, straight-through sign to the code
latent), so logged bpb = the deployable matmul-free model.

This is richer than the ScoreLUT (exp542-544): a full P-bit DOT instead of m scalar
gathers (which capped ~+0.2 vs Linear). vs exp513 Linear 1.4825 (10-step)/1.4656
(100-step). Backbone identical; only the head changed (Linear 12.6M matmul ->
P-bit popcount, codes 4.2MB at deploy).
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
from spiky.lutorch.mixture_bit_readout import MixtureBitReadout

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']; E = cfg['embedding_dim']; D = cfg['residual_dim']
H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']; N_LAYERS = cfg['num_layers']
DEVICE_BS = cfg['device_batch_size']; TOTAL_BS = cfg['total_batch_size']
N_STEPS = cfg['n_steps']; EVAL_EVERY = cfg['eval_every']; EVAL_STEPS = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']; _ROPE_BASE = cfg.get('rope_base', 10000.0)
_NOISE_EPS = cfg.get('argmax_noise_eps', 0.0)
MIX_T = cfg['mix_n_tables']; MIX_B = cfg.get('mix_bits', 15); BIT_SIGN_TEMP = cfg.get('bit_sign_temp', 0.5)

BASE_DIR = get_base_dir()
tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)

_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32, anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001), backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5), select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True), use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS)

def _mk(input_dim, n_heads, n_outputs, nap, tph, so):
    return TinyMultiHeadLut(input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=nap, tables_per_head=tph, random_seed=cfg['random_seed'] + so,
        device=DEVICE, **_TINY_SOFT_KWARGS)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1); return torch.cat([-b, a], dim=-1)
def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

class MeanAbsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, x): return x / (x.abs().mean(dim=-1, keepdim=True) + self.eps)

class LUTBlock(nn.Module):
    def __init__(self, li):
        super().__init__()
        self.qk_lut = _mk(E, H, 2 * d_qk, cfg['qkv_input_nap'], cfg['qkv_tph'], li)
        self.v_lut = _mk(E, H, d_v, cfg['v_input_nap'], cfg['v_tph'], 200 + li)
        self.out_proj = _mk(H * d_v, 1, E, cfg['out_input_nap'], cfg['out_tph'], 400 + li)
        self.residual_lut = _mk(E, 1, D, cfg['residual_input_nap'], cfg['residual_tph'], 600 + li)
        self.q_norm = nn.LayerNorm(d_qk); self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre = MeanAbsNorm(E); self.ln_post = MeanAbsNorm(E)
    def forward(self, x, cos, sin):
        B, T, _ = x.shape; xf = x.reshape(B * T, E); xp = self.ln_pre(xf)
        qk = self.qk_lut(xp)
        q = self.q_norm(qk[..., :d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = self.k_norm(qk[..., d_qk:2 * d_qk]).reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v = self.v_lut(xp).reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        oe = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)).squeeze(1)
        xn = xf + oe
        r = self.residual_lut(self.ln_post(xn)).squeeze(1).reshape(B, T, D)
        return xn.reshape(B, T, E), r

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.head = MixtureBitReadout(D, VOCAB_SIZE, MIX_T, bits_per_table=MIX_B,
                                      sign_temp=BIT_SIGN_TEMP, device=DEVICE, seed=cfg['random_seed'])
        self.rope = RotaryEmbedding(d_qk, CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)
    def get_device(self): return self.tok_emb_E.weight.device
    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin); x_resid = x_resid + r
        logits = self.head(self.ln_final(x_resid).reshape(B * T, D))
        if targets is not None:
            return F.cross_entropy(logits, targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits

model = Model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Total train params: {n_params:,} | MixtureBitReadout T={MIX_T} x {MIX_B} bits | {model.head}')
print(f'head latent (train) {model.head.latent.numel()/1e6:.1f}M -> deploy codes {MIX_T*VOCAB_SIZE*MIX_B/8/1e6:.2f}MB (popcount MoS) vs Linear 12.6M matmul')

def get_lr_scale(step):
    n = N_STEPS; w = int(WARMUP_FRAC * n)
    if step < w: return step / max(w, 1)
    p = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

lut_params, head_params, tok_emb_params, decay_params, nodecay_params = [], [], [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    if name.startswith('head.'): head_params.append(p)
    elif p.ndim >= 3: lut_params.append(p)
    elif name.startswith('tok_emb_E.'): tok_emb_params.append(p)
    elif p.ndim == 2: decay_params.append(p)
    else: nodecay_params.append(p)

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp['lr'], grp['betas'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None: continue
                g = p.grad; st = self.state[p]
                if 'exp_avg' not in st: st['exp_avg'] = torch.zeros_like(p)
                m = st['exp_avg']
                if wd != 0: p.mul_(1.0 - lr * wd)
                p.add_((m * b1 + g * (1.0 - b1)).sign_(), alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)

_LUT_LR = cfg.get('lut_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=decay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params + head_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)
lut_optimizer = Lion([dict(params=lut_params, lr=_LUT_LR, weight_decay=0.0)],
                     lr=_LUT_LR, betas=tuple(cfg.get('lut_betas', (0.9, 0.95))))
all_optimizers = [optimizer, lut_optimizer]
for o in all_optimizers:
    for grp in o.param_groups: grp['initial_lr'] = grp['lr']
print(f'lut(LION)={sum(p.numel() for p in lut_params):,} | head(AdamW)={sum(p.numel() for p in head_params):,}')

tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb'])
val_bpbs, val_steps, train_losses = [], [], []
ema = None; best_bpb = float('inf'); t0 = time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step)
    for o in all_optimizers:
        for grp in o.param_groups: grp['lr'] = grp['initial_lr'] * lr_scale
    for o in all_optimizers: o.zero_grad()
    accum = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum += loss.item() / grad_accum
    for o in all_optimizers: o.step()
    ema = accum if ema is None else 0.99 * ema + 0.01 * accum
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | ce_ema={ema:.4f} | lr={lr_scale*cfg["adam_lr"]:.2e}')
    if step in (1, 5) and DEVICE == 'cuda':
        print(f'[MEM] step {step} alloc_peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        val_bpbs.append(bpb); val_steps.append(step); train_losses.append(ema)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}']); csv_f.flush(); model.train()

csv_f.close(); elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses); ax1.set(xlabel='step', ylabel='ce', title='train ce'); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, color='red'); ax2.axhline(1.4825, ls='--', color='gray', label='exp513 Linear')
ax2.set(xlabel='step', ylabel='bpb', title='val bpb'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close(fig)
summary = dict(exp_name=cfg['exp_name'], best_val_bpb=best_bpb,
               final_val_bpb=val_bpbs[-1] if val_bpbs else float('nan'),
               n_params=n_params, training_time_hours=round(elapsed / 3600, 3))
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f: json.dump(summary, f, indent=2)
torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'step': N_STEPS, 'best_val_bpb': best_bpb}, os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
