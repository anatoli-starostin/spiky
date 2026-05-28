"""nanochat_exps/exp319_minimal_gpt_rope — exp001 with RoPE instead of learned
absolute positional embeddings. Single trade: drop nn.Embedding(seq_len, n_embd)
and the additive position term; insert rotary-position embedding on q and k
inside MinimalAttention. Standard half-rotation RoPE form (base=10000).
All other hyperparameters (depth, n_embd, n_head, lr, wd, batch, steps,
weight-tying, zero-init residual projections) match exp001 exactly.

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

# --- Make nanochat importable -------------------------------------------------
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
COMPUTE_DTYPE = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
}[cfg.get('compute_dtype', 'bf16')]

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
    """Standard half-rotation RoPE buffers. cos/sin shape: [seq_len, head_dim].

    The first head_dim//2 components share frequencies with the second half
    (cat([freqs, freqs])); paired with rotate_half this implements the
    classic q' = q*cos + rotate_half(q)*sin formulation used in llama/falcon.
    """
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        ))                                                          # [head_dim/2]
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                            # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                     # [T, head_dim]
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


class LUTFFN(nn.Module):
    """Replaces vanilla Linear(n,4n)+GELU+Linear(4n,n) with one TinyMultiHeadLut.
    H=1, NAP=6, tph=1024, n_out=n_embd. LayerNorm is applied upstream by the block."""
    def __init__(self, n_embd, layer_idx):
        super().__init__()
        self.lut = TinyMultiHeadLut(
            input_dim=n_embd,
            n_heads=1,
            n_outputs=n_embd,
            n_anchor_pairs=cfg['lut_input_nap'],
            tables_per_head=cfg['lut_tph'],
            random_seed=cfg['random_seed'] + 500 + layer_idx,
            device=DEVICE,
            weight_dtype=torch.float32,
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
            backward_mode='soft',
            soft_score_temp=cfg.get('soft_score_temp', 0.5),
            select_temp=cfg.get('select_temp', 0.5),
            learnable_temps=cfg.get('soft_learnable_temps', True),
            use_bf16=cfg.get('soft_use_bf16', True),
            argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
        )

    def forward(self, x):
        # x: [B, T, n_embd] -> flatten -> LUT -> [B*T, 1, n_embd] -> reshape.
        B, T, C = x.shape
        out = self.lut(x.reshape(B * T, C)).squeeze(1)
        return out.reshape(B, T, C)


class MinimalBlock(nn.Module):
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = LUTFFN(n_embd, layer_idx)

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
        self.blocks  = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # Zero-init attention output projection so attn residual branch starts as identity.
        # (LUT FFN has its own small-init via mhlut_init_std; no zero-init needed.)
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


class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum (Lion). Sign-based with single momentum buffer."""
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, (b1, b2), wd = grp['lr'], grp['betas'], grp['weight_decay']
            for p in grp['params']:
                if p.grad is None: continue
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


def setup_optimizers(model, lr, lut_lr, weight_decay, lut_betas):
    # LUT params (ndim>=3, from TinyMultiHeadLut weight tensors) go to Lion.
    # Other 2D weights (Linears, embeddings) go to AdamW with weight decay.
    # 1D (LN gamma/beta, learnable temps) go to AdamW without weight decay.
    lut_params = [p for p in model.parameters() if p.ndim >= 3]
    decay      = [p for p in model.parameters() if p.ndim == 2]
    nodecay    = [p for p in model.parameters() if p.ndim < 2]
    adam_groups = [
        dict(params=decay,   lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ]
    adam_opt = torch.optim.AdamW(adam_groups)
    lion_opt = Lion([dict(params=lut_params, lr=lut_lr, weight_decay=0.0)],
                    lr=lut_lr, betas=tuple(lut_betas))
    for opt in (adam_opt, lion_opt):
        for g in opt.param_groups:
            g['initial_lr'] = g['lr']
    n_lut = sum(p.numel() for p in lut_params)
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in nodecay)
    print(f'LUT optimizer = Lion (lut_lr={lut_lr}, betas={lut_betas}) | '
          f'lut={n_lut:,} | adam_decay={n_decay:,} | adam_nodecay={n_nodecay:,}')
    return [adam_opt, lion_opt]


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
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN}')
print(f'Params: {total_params:,}')

optimizers = setup_optimizers(model, lr=LR, lut_lr=cfg['lut_lr'],
                              weight_decay=WD, lut_betas=cfg.get('lut_betas', (0.9, 0.95)))
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
    for opt in optimizers:
        for g in opt.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale

    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for opt in optimizers:
        opt.step()

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
