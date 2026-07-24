"""exp011_hyperplane_mhl_ffn_nap6_tph256_stack2_ln_resid — exp007 (MinimalGPT +
RoPE, untied head; dense MLP FFN replaced by a single HyperplaneMultiHeadLUT) with
the FFN generalized to a TWO-sublayer stack of HyperplaneMHLs:
    h = lut1(x)                    # first HyperplaneMHL, 384->384
    h = h + lut2(LayerNorm(h))     # residual around the second, LN as pre-norm
Both LUTs keep exp007 geometry (NAP6, tph256, hard). The outer block residual
x + mlp(ln2(x)) is unchanged. This is the only architectural change vs exp007.

The ONLY architectural change vs exp001 is the FFN: `x + mlp(ln2(x))` becomes
`x + hyperplane_mhl(ln2(x))`, where the FFN is a HyperplaneMultiHeadLUT
(n_heads=1, n_outputs=d_model, hard forward, always-soft backward). All other
model/training hyperparameters match exp001 exactly.

Dtypes / optimizer (matching the latest FastMHL recipe, plus the new hyperplanes):
  - LUT table weights: bf16 storage, optimized by LION (lr 2e-4, betas (0.9,0.95),
    wd 0), grad_clip 1.0 — exactly the FastMHL bf16 recipe.
  - hyperplane weights w + bias b: fp32 storage, optimized by ADAM, NO weight
    decay (routed by parameter IDENTITY, not ndim — both the LUT weights and w
    are 3-D, only the LUT goes to Lion).
  - temperatures: fp32, Adam, no wd.
  - rest of the model (attn, tok_emb, unembed head, LayerNorms): exp001's verbatim
    AdamW rule (ndim>=2 -> wd 0.1, else no wd).
Both optimizers share the exp001 warmup(0.1)+cosine(->0.1x) schedule.

Outputs (alongside this script):
  metrics.csv     step, train_loss, val_bpb (rows at every eval_every step)
  summary.json    final model size + best/final val_bpb + training time
  loss.png        training loss + validation BPB curves
  checkpoint.pt   final model state_dict
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Make nanochat + spiky importable ----------------------------------------
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
# spiky is normally pip-installed (editable) in the venv; add src as a fallback.
_SPIKY_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
if os.path.isdir(_SPIKY_SRC) and _SPIKY_SRC not in sys.path:
    sys.path.insert(0, _SPIKY_SRC)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPUTE_DTYPE = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
}[cfg.get('compute_dtype', 'bf16')]
_DTYPE = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}

torch.manual_seed(cfg['random_seed'])

DEPTH       = cfg['depth']
N_EMBD      = cfg['n_embd']
N_HEAD      = cfg['n_head']
SEQ_LEN     = cfg['seq_len']
DEVICE_BS   = cfg['device_batch_size']
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
LR          = cfg['lr']
WD          = cfg['weight_decay']
WARMUP_FRAC = cfg['lr_warmup_fraction']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
GRAD_CLIP   = cfg.get('grad_clip', 1.0)
LUT_LR      = cfg['lut_lr']
LUT_BETAS   = tuple(cfg.get('lut_betas', (0.9, 0.95)))
LUT_DTYPE   = _DTYPE[cfg.get('hml_weight_dtype', 'bf16')]
HP_DTYPE    = _DTYPE[cfg.get('hml_hyperplane_dtype', 'fp32')]

# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')
assert VOCAB_SIZE == cfg['tokenizer_vocab_size'], \
    f"tokenizer vocab {VOCAB_SIZE} != cfg vocab {cfg['tokenizer_vocab_size']}"

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- RoPE (rotary position embedding) ----------------------------------------
class RotaryEmbedding(nn.Module):
    """Standard half-rotation RoPE buffers. cos/sin shape: [seq_len, head_dim]."""
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


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: [B, H, T, D]. cos, sin: [T, D]. Returns rotated q, k."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# --- MinimalGPT (vanilla GPT-2 style + RoPE) ---------------------------------
class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class HyperplaneFFN(nn.Module):
    """FFN = TWO stacked single-head HyperplaneMultiHeadLUT with an intermediate
    pre-LayerNorm and an internal residual around the second sublayer:

        h = lut1(x)                      # first HyperplaneMHL, 384->384
        h = h + lut2(LayerNorm(h))       # residual around the second, LN as pre-norm

    Each LUT has n_heads=1 & n_outputs=C, so it maps d_model->d_model with no
    reshape/projection; the intermediate LayerNorm(C) sits on the d_model-wide
    intermediate representation. The outer block residual (x + mlp(ln2(x))) is
    unchanged. Distinct, reproducible per-layer init seeds keep no two LUTs
    identical: lut1 = seed + 2*layer_idx + 1, lut2 = seed + 2*layer_idx + 2.
    """
    def __init__(self, n_embd, layer_idx):
        super().__init__()
        assert cfg['hml_n_heads'] * cfg['hml_n_outputs'] == n_embd, \
            "hml_n_heads * hml_n_outputs must equal d_model"
        self.n_embd = n_embd
        self.lut1 = self._make_lut(n_embd, cfg['random_seed'] + 2 * layer_idx + 1)
        self.norm = nn.LayerNorm(n_embd)           # pre-norm on the second sublayer
        self.lut2 = self._make_lut(n_embd, cfg['random_seed'] + 2 * layer_idx + 2)

    @staticmethod
    def _make_lut(n_embd, seed):
        return HyperplaneMultiHeadLUT(
            input_dim=n_embd,
            n_heads=cfg['hml_n_heads'],
            n_outputs=cfg['hml_n_outputs'],
            n_anchor_pairs=cfg['hml_nap'],
            tables_per_head=cfg['hml_tph'],
            forward_mode=cfg['hml_forward_mode'],
            weight_dtype=LUT_DTYPE,
            hyperplane_dtype=HP_DTYPE,
            use_bf16=cfg.get('hml_use_bf16', True),
            hyperplane_init=cfg['hml_hyperplane_init'],
            initial_weights_noise=cfg['hml_init_noise'],
            learnable_temps=cfg['hml_learnable_temps'],
            soft_score_temp=cfg.get('hml_soft_score_temp', 0.5),
            select_temp=cfg.get('hml_select_temp', 0.5),
            random_seed=seed,                       # distinct, reproducible per LUT
            device=DEVICE,
        )

    def _apply_lut(self, lut, z):
        B, T, C = z.shape
        y = lut(z.reshape(B * T, C))               # [B*T, 1, C]
        return y.reshape(B, T, C).float()          # squeeze head + bf16->fp32

    def forward(self, x):
        h = self._apply_lut(self.lut1, x)                       # first HyperplaneMHL
        h = h + self._apply_lut(self.lut2, self.norm(h))        # residual around the second
        return h


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = HyperplaneFFN(n_embd, layer_idx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        head_dim = n_embd // n_head
        self.rope = RotaryEmbedding(head_dim, max_seq_len=seq_len)
        self.blocks  = nn.ModuleList(
            [MinimalBlock(n_embd, n_head, li) for li in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        # UNTIED: head and tok_emb are separate matrices. Apply normal(0,0.02)
        # init to Linear/Embedding only; HyperplaneMHL keeps its own init.
        self.apply(self._init_weights)
        # Zero-init attn output projections so the attention branch starts as
        # identity. The FFN (HyperplaneMHL) is NOT zero-inited: initial_weights_
        # noise=0.001 perturbs both its LUT and hyperplane weights (deliberate,
        # per the approved plan — no longer an exact-identity init).
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# --- Lion optimizer (verbatim from examples/lutgpt/train.py) ------------------
class Lion(torch.optim.Optimizer):
    """Lion optimizer with optional fp32 master copy for low-precision params.

    Standard Lion: m <- b2*m + (1-b2)*g; update <- sign(b1*m + (1-b1)*g);
    p <- p*(1-lr*wd) - lr*update. For bf16 params, applies the sign-step to an
    fp32 master copy + fp32 momentum, then casts back to the bf16 param so the
    forward still reads bf16 from HBM (avoids the +5-9mb bpb rounding drift).
    """
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


def setup_optimizers(model, lr, lut_lr, weight_decay):
    """Hybrid optimizer, routed by parameter IDENTITY (not ndim):
      - LUT table weights (HyperplaneMHL.weights)                -> Lion (wd 0)
      - hyperplane_weight, hyperplane_bias, temperatures         -> Adam, no wd
      - everything else (attn/emb/head 2-D -> wd; LN 1-D -> no wd)-> AdamW (exp001)
    """
    lut_params, hp_params, seen = [], [], set()
    for blk in model.blocks:
        # BOTH stacked LUTs must be routed: table weights -> Lion, hyperplane
        # w/b + temps -> Adam (no wd). Missing lut2 would silently drop its
        # params into AdamW-with-wd (wrong).
        for lut in (blk.mlp.lut1, blk.mlp.lut2):
            lut_params.append(lut.weights)
            hp_params.append(lut.hyperplane_weight)
            hp_params.append(lut.hyperplane_bias)
            specials = [lut.weights, lut.hyperplane_weight, lut.hyperplane_bias]
            if lut.learnable_temps:
                hp_params.append(lut.log_soft_score_temp)
                hp_params.append(lut.log_select_temp)
                specials += [lut.log_soft_score_temp, lut.log_select_temp]
            seen.update(id(p) for p in specials)

    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    decay   = [p for p in rest if p.ndim >= 2]
    nodecay = [p for p in rest if p.ndim < 2]

    adam = torch.optim.AdamW([
        dict(params=decay,     lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        dict(params=nodecay,   lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(params=hp_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ])
    lion = Lion([dict(params=lut_params, lr=lut_lr, weight_decay=0.0)],
                lr=lut_lr, betas=LUT_BETAS)
    opts = [adam, lion]
    for o in opts:
        for g in o.param_groups:
            g['initial_lr'] = g['lr']
    n_lut = sum(p.numel() for p in lut_params)
    n_hp  = sum(p.numel() for p in hp_params)
    print(f'Hybrid optimizer | Lion(LUT weights)={n_lut:,} lr={lut_lr} betas={LUT_BETAS} wd=0 | '
          f'Adam(hyperplane w/b/temps, NO wd)={n_hp:,} lr={lr} | '
          f'AdamW(rest): decay={sum(p.numel() for p in decay):,} (wd={weight_decay}), '
          f'nodecay={sum(p.numel() for p in nodecay):,}')
    return opts


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# --- Build + train ------------------------------------------------------------
model = MinimalGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT + stacked(2) HyperplaneMHL FFN (+ intermediate LN, internal residual): depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN}')
print(f'Params: {total_params:,}')

optimizers = setup_optimizers(model, lr=LR, lut_lr=LUT_LR, weight_decay=WD)
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb'])

train_losses_logged = []
val_bpbs = []
val_steps = []
ema = None
best_bpb = float('inf')
t0 = time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, N_STEPS, WARMUP_FRAC)
    for o in optimizers:
        for g in o.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale

    for o in optimizers:
        o.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    if GRAD_CLIP is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    for o in optimizers:
        o.step()

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * LR:.2e}')

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
        model.train()

csv_f.close()
elapsed = time.time() - t0

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy loss', title='Training Loss')
ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb')
ax2.set(xlabel='step', ylabel='bits per byte', title='Validation BPB')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_bpb': best_bpb,
    'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
    'total_params': total_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
