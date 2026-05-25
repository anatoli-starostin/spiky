"""nanochat_exps/exp535_soft_anchor_pairs — fork of exp534.

Directed-gradient trainable anchors = SMART DATA-DRIVEN PAIR SAMPLING.

All 4 LUT modules use SoftAnchorPairMHLUT
(src/spiky/lutorch/trainable_anchors_multi_head_lut.py): each anchor pair is two
LEARNED soft convex blends compared by a sign bit:
    a = softmax(alpha / tau) @ x        # soft anchor "+"
    b = softmax(beta  / tau) @ x        # soft anchor "-"
    bit = sign(a - b)                    # explicit PAIR comparison on the forward
Because the softmax is IN the forward (not a discarded backward surrogate like the
argmax-STE in exp533/534), a and b move continuously with alpha/beta -> the
boundary a=b rotates smoothly -> the anchor gradient is DIRECTED (no teleport, no
re-fit tax). softmax(alpha)-softmax(beta) is zero-sum so it stays a shift-invariant
difference -> a genuine learned PAIR.

anchor_tau is ANNEALED 1.0 -> 0.02 (log-linear over the first anneal_frac of
training, held after) so each softmax collapses to ~one-hot and the FINAL model
makes HARD pair decisions. Each eval reports BOTH the soft model at the current tau
(val_bpb, what's optimized) AND the exact hard-argmax model (val_bpb_hard,
mod.hard=True -> cheap coordinate gather, the deployable model). A final hard eval
is reported at the end.

Replaces the fixed CANONICAL_FULL_COVERAGE pair-sampling policy with a learned one.
Same naps/tphs/optimizer/LR/steps as exp530; per-block gradient checkpointing.
Optimizer: LUT table weights (ndim==3) -> LION; anchor_logits (ndim==4) -> AdamW
(anchor_lr); everything else -> AdamW. ~130M params.
vs exp530 1.4731 (fixed coverage), exp534 (undirected ~+0.026), exp513 1.4825.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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

from spiky.lutorch.trainable_anchors_multi_head_lut import SoftAnchorPairMHLUT

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

TAU_START   = cfg.get('anchor_tau_start', 1.0)
TAU_MID     = cfg.get('anchor_tau_mid', 0.02)
TAU_END     = cfg.get('anchor_tau_end', 1e-3)
ANNEAL_FRAC = cfg.get('anchor_anneal_frac', 0.3)


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


# --- LUT factories (all SoftAnchorPairMHLUT) ----------------------------------
_SA_KWARGS = dict(
    sign_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    anchor_init_std=cfg.get('anchor_init_std', 1.0),
    anchor_tau_init=TAU_START,
    weights_init_std=cfg.get('mhlut_init_std', 0.001),
)

def _make_qk(layer_idx, seed_offset):
    return SoftAnchorPairMHLUT(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_SA_KWARGS,
    )

def _make_v(layer_idx, seed_offset):
    return SoftAnchorPairMHLUT(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_SA_KWARGS,
    )

def _make_out(layer_idx, seed_offset):
    return SoftAnchorPairMHLUT(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_SA_KWARGS,
    )

def _make_residual_lut(layer_idx, seed_offset):
    return SoftAnchorPairMHLUT(
        input_dim=E, n_heads=1, n_outputs=D,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_SA_KWARGS,
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
    """L1 mean-absolute norm x / (mean(|x|) + eps); no centering, no affine."""
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
        self.residual_lut = _make_residual_lut(layer_idx, 600 + layer_idx)

        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre  = MeanAbsNorm(E)
        self.ln_post = MeanAbsNorm(E)

    def _body(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)

        x_pre = self.ln_pre(x_flat)

        qk_out = self.qk_lut(x_pre)                     # [B*T, H, 2*d_qk]
        q_vec = self.q_norm(qk_out[..., :d_qk])
        k_vec = self.k_norm(qk_out[..., d_qk:2 * d_qk])
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = self.v_lut(x_pre)                       # [B*T, H, d_v]
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)       # [B*T, E]

        x_lut_next_flat = x_flat + out_e                # [B*T, E]
        x_lut_next = x_lut_next_flat.reshape(B, T, E)

        r_in   = self.ln_post(x_lut_next_flat)          # [B*T, E]
        r_out  = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next, r_out

    def forward(self, x, cos, sin):
        if self.training:
            return checkpoint(self._body, x, cos, sin, use_reentrant=False)
        return self._body(x, cos, sin)


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

# All SoftAnchorPairMHLUT modules (for tau scheduling + hard-eval toggling).
sa_modules = [m for m in model.modules() if isinstance(m, SoftAnchorPairMHLUT)]

n_params = sum(p.numel() for p in model.parameters())
print(f'Total params (all fp32): {n_params:,}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

def anchor_tau_at(step):
    """Two-phase log-linear anneal:
      phase 1 (step <= ANNEAL_FRAC*N): TAU_START -> TAU_MID  (fast: sharpen bits
              during peak LR so the soft model can learn good pairs);
      phase 2 (step >  ANNEAL_FRAC*N): TAU_MID  -> TAU_END   (continue toward ~0
              so the soft-trained model becomes the EXACT hard model -> the
              soft<->hard residual vanishes by the end).
    """
    s1 = max(1, int(ANNEAL_FRAC * N_STEPS))
    if step <= s1:
        p = step / s1
        return math.exp((1.0 - p) * math.log(TAU_START) + p * math.log(TAU_MID))
    p = (step - s1) / max(1, N_STEPS - s1)
    return math.exp((1.0 - p) * math.log(TAU_MID) + p * math.log(TAU_END))

def set_anchor_tau(step):
    tau = anchor_tau_at(step)
    for m in sa_modules:
        m.anchor_tau.fill_(tau)
    return tau

def set_hard(flag):
    for m in sa_modules:
        m.hard = flag

lut_params     = []   # LUT table weights (ndim==3) -> LION
anchor_params  = []   # anchor selection logits (ndim==4) -> AdamW (anchor_lr)
tok_emb_params = []
decay_params   = []
nodecay_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.endswith('anchor_logits'):
        anchor_params.append(p)
    elif p.ndim >= 3:
        lut_params.append(p)
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)

class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum. update = -lr*sign(b1*m + (1-b1)*g); m = b2*m + (1-b2)*g."""
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

_LUT_LR    = cfg.get('lut_lr', cfg['adam_lr'])
_ANCHOR_LR = cfg.get('anchor_lr', cfg['adam_lr'])
_LUT_OPT   = cfg.get('lut_optimizer', 'lion')

adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=tok_emb_params + nodecay_params,
         lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    dict(params=anchor_params,  lr=_ANCHOR_LR, betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
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
    f'LUT optimizer = {_LUT_OPT} (lut_lr={_LUT_LR}) | anchor_lr={_ANCHOR_LR} (AdamW) | '
    f'lut_weights={sum(p.numel() for p in lut_params):,} | '
    f'anchor_logits={sum(p.numel() for p in anchor_params):,} | '
    f'decay(unembed)={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay", 0.0)}) | '
    f'tok_emb={sum(p.numel() for p in tok_emb_params):,} | '
    f'nodecay_other={sum(p.numel() for p in nodecay_params):,}'
)
print(f'D=residual_dim={D}, E={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'SoftAnchorPairMHLUT: sign(softmax(a/tau).x - softmax(b/tau).x); tau {TAU_START} ->[{ANNEAL_FRAC}] {TAU_MID} -> {TAU_END} (two-phase anneal to ~0)')
print(f'qk in_nap={cfg["qkv_input_nap"]} tph={cfg["qkv_tph"]} | v in_nap={cfg["v_input_nap"]} tph={cfg["v_tph"]} | '
      f'out in_nap={cfg["out_input_nap"]} tph={cfg["out_tph"]} | res in_nap={cfg["residual_input_nap"]} tph={cfg["residual_tph"]}')
print(f'anchor_init_std={cfg.get("anchor_init_std",1.0)} | UNTIED unembedder Linear(D,V); RoPE base={_ROPE_BASE} | per-block ckpt ON | dual eval (soft@tau + hard)')


# --- Temperature + anchor-stats tracking --------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for nm in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(blk, nm)
            specs.append((f'L{li}.{nm}.T_sign', (lambda m=mod: float(m.log_temps.detach()[0].exp()))))
            specs.append((f'L{li}.{nm}.T_sel',  (lambda m=mod: float(m.log_temps.detach()[1].exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_f = open(os.path.join(EXP_DIR, 'temperatures.csv'), 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step', 'anchor_tau'] + [name for name, _ in temp_specs])
print(f'Tracking {len(temp_specs)} learnable temps + anchor_tau in temperatures.csv')

def _anchor_argmax_snapshot():
    snap = {}
    for li, blk in enumerate(model.layers):
        for nm in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
            snap[f'L{li}.{nm}'] = getattr(blk, nm).anchor_logits.detach().argmax(dim=-1).clone()
    return snap

_ANCHOR_INIT = _anchor_argmax_snapshot()
_anchor_csv_f = open(os.path.join(EXP_DIR, 'anchor_stats.csv'), 'w', newline='')
_anchor_csv_w = csv.writer(_anchor_csv_f)
_anchor_csv_w.writerow(['step', 'module', 'frac_flipped_vs_init', 'mean_max_softmax_w'])
print(f'Tracking anchor flips + hardness of {len(_ANCHOR_INIT)} modules in anchor_stats.csv')

def _log_anchor_stats(step_, tau):
    with torch.no_grad():
        for li, blk in enumerate(model.layers):
            for nm in ('qk_lut', 'v_lut', 'out_proj', 'residual_lut'):
                key = f'L{li}.{nm}'
                al = getattr(blk, nm).anchor_logits.detach()
                cur = al.argmax(dim=-1)
                frac = float((cur != _ANCHOR_INIT[key]).float().mean())
                hardness = float(F.softmax(al / tau, dim=-1).max(dim=-1).values.mean())
                _anchor_csv_w.writerow([step_, key, f'{frac:.6f}', f'{hardness:.6f}'])
        _anchor_csv_f.flush()


# --- Eval helper (soft @ current tau, or exact hard) --------------------------
def eval_bpb(hard):
    set_hard(hard)
    model.eval()
    bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    model.train()
    set_hard(False)
    return bpb


# --- Training loop ------------------------------------------------------------
tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb', 'val_bpb_hard', 'anchor_tau'])

train_losses_logged, val_bpbs, val_bpbs_hard, val_steps = [], [], [], []
ema = None
best_bpb = float('inf')
best_bpb_hard = float('inf')
t0 = time.time()

temp_w.writerow([0, f'{TAU_START:.6f}'] + [f'{getter():.6f}' for _, getter in temp_specs])
temp_f.flush()

model.train()
for step in range(1, N_STEPS + 1):
    tau = set_anchor_tau(step)
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
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e} | tau={tau:.4f}')

    if step in (1, 5) and DEVICE == 'cuda':
        print(f'[MEM] step {step} alloc_peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB '
              f'reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        bpb = eval_bpb(hard=False)
        bpb_hard = eval_bpb(hard=True)
        best_bpb = min(best_bpb, bpb)
        best_bpb_hard = min(best_bpb_hard, bpb_hard)
        print(f'[VAL] step {step}: bpb={bpb:.4f} | bpb_hard={bpb_hard:.4f} | tau={tau:.4f}')
        train_losses_logged.append(ema)
        val_bpbs.append(bpb); val_bpbs_hard.append(bpb_hard); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}', f'{bpb_hard:.6f}', f'{tau:.6f}'])
        csv_f.flush()
        temp_w.writerow([step, f'{tau:.6f}'] + [f'{getter():.6f}' for _, getter in temp_specs])
        temp_f.flush()
        _log_anchor_stats(step, tau)

csv_f.close(); temp_f.close(); _anchor_csv_f.close()
elapsed = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, label='val bpb (soft@tau)', color='red')
ax2.plot(val_steps, val_bpbs_hard, label='val bpb (hard)', color='purple', linestyle='--')
ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120)
plt.close(fig)

summary = dict(
    exp_name=cfg['exp_name'],
    best_val_bpb=best_bpb,
    best_val_bpb_hard=best_bpb_hard,
    final_val_bpb=val_bpbs[-1] if val_bpbs else float('nan'),
    final_val_bpb_hard=val_bpbs_hard[-1] if val_bpbs_hard else float('nan'),
    n_params=n_params,
    training_time_hours=round(elapsed / 3600, 3),
)
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

set_hard(True)  # deployable model makes hard pair decisions
ckpt_path = os.path.join(EXP_DIR, 'checkpoint.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lut_optimizer_state_dict': lut_optimizer.state_dict(),
    'config': cfg, 'step': N_STEPS,
    'best_val_bpb': best_bpb, 'best_val_bpb_hard': best_bpb_hard,
}, ckpt_path)
print(f'saved checkpoint -> {ckpt_path}')

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
