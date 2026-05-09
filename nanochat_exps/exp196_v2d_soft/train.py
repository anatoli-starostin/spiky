"""nanochat_exps/exp196_v2d_soft.

Fork of exp194 (untied head, linear V2D). Single change: V2D switched from
linear_mode=True (raw x_a - x_b) to linear_mode=False so the smooth_mode
soft path runs: (x_a - x_b) / (t + |x_a - x_b|), bounded in (-1, 1).

Launch:
    PYTHONPATH=/home/starost/nanochat \\
      /home/starost/spiky/.venv/bin/python -u \\
      nanochat_exps/exp196_v2d_soft/train.py \\
      > nanochat_exps/exp196_v2d_soft/stdout.log 2>&1 &
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

from spiky.lutorch.ranking_tools import VectorToDominance

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH         = cfg['depth']
E             = cfg['n_embd']
H             = cfg['n_head']
SEQ_LEN       = cfg['seq_len']
DEVICE_BS     = cfg['device_batch_size']
TOTAL_BS      = cfg['total_batch_size']
N_STEPS       = cfg['n_steps']
LR            = cfg['lr']
WD            = cfg['weight_decay']
WARMUP_FRAC   = cfg['lr_warmup_fraction']
EVAL_EVERY    = cfg['eval_every']
EVAL_STEPS    = cfg['eval_steps']

D             = cfg['v2d_n_pairs']               # NlogN per-head pair count
OUT_HIDDEN    = cfg.get('out_mlp_hidden', 4 * E)

assert E % H == 0, f'n_embd ({E}) must be divisible by n_head ({H})'
E_H = E // H

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


# --- Architecture -------------------------------------------------------------
class V2DSDPABlock(nn.Module):
    """Pre-norm: LN -> V2D -> per-head Q/K/V -> SDPA -> MLP out_proj -> residual."""
    def __init__(self, layer_idx: int):
        super().__init__()
        self.ln = nn.LayerNorm(E)
        # V2D with per-head random pair subsets.
        self.v2d = VectorToDominance(
            d_head=E,
            n_pairs=D,
            n_heads=H,
            smooth_mode=True,
            linear_mode=cfg.get('v2d_linear_mode', False),
            temperature=cfg.get('v2d_temperature', 0.1),
            random_seed=cfg['random_seed'] + 1000 + layer_idx,
        )
        # Per-head Q, K, V: weights [H, D, E_h], no bias.
        qkv_std = cfg.get('qkv_init_std', 0.005)
        gen = torch.Generator().manual_seed(cfg['random_seed'] + 2000 + layer_idx)
        self.Wq = nn.Parameter(torch.randn(H, D, E_H, generator=gen) * qkv_std)
        self.Wk = nn.Parameter(torch.randn(H, D, E_H, generator=gen) * qkv_std)
        self.Wv = nn.Parameter(torch.randn(H, D, E_H, generator=gen) * qkv_std)
        # out_proj: standard 2-layer MLP (Linear -> GELU -> Linear), no bias.
        # Zero-init the final projection so each block starts as the identity
        # residual (block contributes 0). fc1 gets std=0.02; fc2 starts at 0
        # and picks up signal on the first step.
        out_hidden = cfg.get('out_mlp_hidden', 4 * E)
        self.out_fc1 = nn.Linear(E, out_hidden, bias=False)
        self.out_fc2 = nn.Linear(out_hidden, E, bias=False)
        nn.init.normal_(self.out_fc1.weight, std=0.02)
        nn.init.zeros_(self.out_fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        B, T, _ = x.shape
        x_norm = self.ln(x)                                    # pre-norm
        # Per-head V2D: input must have explicit head dim. Same E-dim feeds
        # all heads; each head has its own pair subset.
        x_h = x_norm.unsqueeze(-2).expand(B, T, H, E)          # [B, T, H, E]
        dom = self.v2d(x_h)                                    # [B, T, H, D]
        # Q, K, V: [B, T, H, E_h] via per-head einsum.
        q = torch.einsum('bthd,hde->bthe', dom, self.Wq)
        k = torch.einsum('bthd,hde->bthe', dom, self.Wk)
        v = torch.einsum('bthd,hde->bthe', dom, self.Wv)
        # SDPA expects [B, H, T, E_h].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, E_h]
        attn = attn.transpose(1, 2).contiguous().view(B, T, E)          # [B, T, E]
        # out_proj MLP: E -> 4E -> GELU -> E.
        out = self.out_fc2(F.gelu(self.out_fc1(attn)))
        return x + out                                                  # residual


class V2DSDPATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, E)
        self.pos_emb = nn.Embedding(SEQ_LEN, E)
        self.blocks  = nn.ModuleList([V2DSDPABlock(i) for i in range(DEPTH)])
        self.ln_f    = nn.LayerNorm(E)
        self.head    = nn.Linear(E, VOCAB_SIZE, bias=False)
        # Untied head: head.weight is its own [VOCAB, E] tensor (no sharing
        # with tok_emb). Init both at std=0.02.
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight,    std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)              # [B, T, E]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


def setup_optimizer(model, lr, weight_decay):
    """Standard AdamW: decay on ndim>=2 (Linears, embeddings, Wq/Wk/Wv),
       no decay on ndim<2 (LayerNorm gains/biases)."""
    decay   = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    print(f'AdamW groups: decay={sum(p.numel() for p in decay):,} '
          f'nodecay={sum(p.numel() for p in nodecay):,}')
    groups = [
        dict(params=decay,   lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ]
    opt = torch.optim.AdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# --- Build + train ------------------------------------------------------------
model = V2DSDPATransformer().to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f'V2DSDPATransformer: depth={DEPTH}, E={E}, H={H}, E_H={E_H}, D={D}, seq_len={SEQ_LEN}')
print(f'out_proj MLP: hidden={OUT_HIDDEN} (zero-init fc2)')
print(f'Params: {total_params:,}')

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
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
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

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
