"""LUTGPT: a small transformer whose attention projections (Q / K / V / out)
and per-layer residuals are TinyMultiHeadLut tables instead of dense
matmuls. The other matmuls a transformer needs are still here: the
token-embedding gather, scaled_dot_product_attention itself, and the final
unembedder Linear.

Architecture (E = embedding_dim, D = residual_dim, H = n_heads):

  tokens -> tok_emb_E [V, E]
            |
            MeanAbsNorm(E) -> emb_resid_lut (NAP=5, tph=256) -> D  -.
            |                                                       |
            |  per layer x N_LAYERS:                                |
            |    MeanAbsNorm(E)                                     |
            |    qk_lut       (NAP=4, tph=256, n_outputs=2*d_qk)    |
            |    v_lut        (NAP=6, tph=256, n_outputs=d_v)       |
            |    scaled-dot-product attention (RoPE on q,k)         |
            |    out_proj     (NAP=6, tph=512, n_outputs=E)         |
            |    residual_lut (NAP=5, tph=256, n_outputs=D)  -------+
            |    x_lut += out_proj(attn)                            |
            |                                                       v
            v                                                    x_resid
            (E-stream, never mutated by D-stream)                 (D-stream)
                                                                  ln_final
                                                                  unembedder -> logits

Two-phase training:
  - phase A (steps 1..bs_switch_step): forward_mode='hybrid_smooth', smaller
    batch. The forward blends each LUT's top-2 rows so gradients are dense.
  - phase B (steps bs_switch_step+1..N): forward_mode='hard', larger batch.
    Forward is the discrete sign-pack lookup. Backward stays soft in both
    phases (see paper/tinymhl_hybrid_smooth.tex).

Requires nanochat (https://github.com/karpathy/nanochat) for the tokenizer,
data loader and bpb eval — point NANOCHAT_ROOT at a local checkout before
running.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT')
if not NANOCHAT_ROOT:
    raise RuntimeError(
        "Set NANOCHAT_ROOT to a nanochat checkout "
        "(https://github.com/karpathy/nanochat) — the tokenizer, data "
        "loader, and bpb eval are pulled from there."
    )
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


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')

# Phase A dataloader (initial).
train_loader_a = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
# Phase B dataloader (post-switch).
DEVICE_BS_B = cfg.get('device_batch_size_b', DEVICE_BS)
TOTAL_BS_B  = cfg.get('total_batch_size_b', TOTAL_BS)
if DEVICE_BS_B != DEVICE_BS:
    train_loader_b = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS_B, CONTEXT_SIZE, split='train', device=DEVICE)
else:
    train_loader_b = train_loader_a
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories ------------------------------------------------------------
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    forward_mode=cfg.get('forward_mode', 'hard'),
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
)

def _make_qk(seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
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

        # Per-layer residual_lut: MeanAbsNorm(E) -> residual_lut -> D-stream contribution.
        r_in  = self.ln_resid(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)

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
        x_resid = self.emb_resid_lut(x_emb_pre).squeeze(1).reshape(B, T, D)
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
print(f'qk_lut       TinyMHLut(soft, LION): in_nap={cfg["qk_input_nap"]} tph={cfg["qk_tph"]} n_out=2*d_qk={2*d_qk}')
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
tokens_per_step_a = DEVICE_BS * CONTEXT_SIZE
grad_accum_a = max(1, TOTAL_BS // tokens_per_step_a)
tokens_per_step_b = DEVICE_BS_B * CONTEXT_SIZE
grad_accum_b = max(1, TOTAL_BS_B // tokens_per_step_b)
print(f'Phase A: bs={DEVICE_BS} tokens/microbatch={tokens_per_step_a:,} grad_accum={grad_accum_a} effective={grad_accum_a*tokens_per_step_a:,}')
print(f'Phase B: bs={DEVICE_BS_B} tokens/microbatch={tokens_per_step_b:,} grad_accum={grad_accum_b} effective={grad_accum_b*tokens_per_step_b:,}')

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

HARD_SWITCH_STEP = cfg.get('hard_switch_step', None)
BS_SWITCH_STEP   = cfg.get('bs_switch_step', None)

def _flip_to_hard():
    """Phase-B forward switch: top-2 hybrid_smooth -> single-row hard forward.

    The soft backward surrogate (input + temp gradients) is identical in both
    forward modes; only the forward path and the weight-gradient scatter change.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, TinyMultiHeadLut):
            mod.forward_mode = 'hard'
            n += 1
    print(f'[HARD SWITCH] flipped forward_mode hybrid_smooth -> hard on {n} LUT modules')

train_loader = train_loader_a
grad_accum   = grad_accum_a

model.train()
for step in range(1, N_STEPS + 1):
    if HARD_SWITCH_STEP is not None and step == HARD_SWITCH_STEP:
        _flip_to_hard()
    if BS_SWITCH_STEP is not None and step == BS_SWITCH_STEP:
        train_loader = train_loader_b
        grad_accum   = grad_accum_b
        print(f'[BS SWITCH] step {step}: dataloader -> bs={DEVICE_BS_B} grad_accum={grad_accum}')
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
