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
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

# exp033: the plain fixed-anchor-pair FastMultiHeadLUT (NOT HyperplaneMHL) added as a
# second, parallel FFN path.  spiky is installed editable in this venv (importable from
# any cwd).
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

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

# exp033 hybrid-FFN (path B = FastMHL) capacity knobs, read from config.json.
LUT_NAP        = cfg['lut_n_anchor_pairs']    # NAP: K = 2^NAP rows per table
LUT_TPH        = cfg['lut_tables_per_head']   # tables summed per head
LUT_HEADS      = cfg['lut_n_heads']           # output heads
LUT_NOUT       = cfg['lut_n_outputs']         # per-head outputs (heads*nout must == n_embd)
LUT_FORWARD    = cfg['lut_forward_mode']      # "hard" (deployable) or "hybrid_smooth"
LUT_USE_BF16   = cfg['lut_use_bf16']          # False -> LUT path runs fp32, matching the model
LUT_INIT_NOISE = cfg['lut_init_weights_noise']
LUT_BASE_SEED  = cfg['lut_base_seed']         # per-layer anchor/init seed = base + layer_idx
assert LUT_HEADS * LUT_NOUT == N_EMBD, \
    f"lut_n_heads*lut_n_outputs ({LUT_HEADS}*{LUT_NOUT}) must equal n_embd ({N_EMBD})"

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


class MinimalBlock(nn.Module):
    """exp033 HYBRID block: the vanilla dense FFN is AUGMENTED (not replaced) with a
    second, parallel FastMultiHeadLUT path on the same LayerNorm'd input. The two
    outputs are SUMMED into the residual:

        h = ln2(x);  x = x + mlp(h) + fastmhl(h)

    Everything else (attention sub-block, LayerNorms, residual structure) is byte-identical
    to exp032/exp002. Gradients flow through BOTH paths simultaneously (mlp is dense; the
    FastMHL surrogate backward emits grads for its LUT tables and, densely, back to h).
    At init the LUT tables are ~zero (init_weights_noise=1e-3), so the block ≈ the vanilla
    FFN and the LUT path grows only if it helps.
    """
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )
        # Path B: plain fixed-anchor-pair FastMHL. Per-layer seed so anchor pairs and
        # table inits are decorrelated across depth. Built on CPU here; the model-level
        # .to(DEVICE) moves its Parameter + buffers onto the GPU.
        self.lut = FastMultiHeadLut(
            input_dim=n_embd, n_heads=LUT_HEADS, n_outputs=LUT_NOUT,
            n_anchor_pairs=LUT_NAP, tables_per_head=LUT_TPH,
            forward_mode=LUT_FORWARD, use_bf16=LUT_USE_BF16,
            initial_weights_noise=LUT_INIT_NOISE,
            random_seed=LUT_BASE_SEED + layer_idx,
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        ffn_out = self.mlp(h)                                   # dense path A: [B, T, C]
        B, T, C = h.shape
        lut_out = self.lut(h.reshape(B * T, C))                # [B*T, LUT_HEADS, LUT_NOUT]
        lut_out = lut_out.reshape(B, T, C).to(ffn_out.dtype)   # heads*nout == C
        x = x + ffn_out + lut_out                              # parallel-sum into residual
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
        # UNTIED (exp476): head and tok_emb are separate matrices (each gets its
        # own normal(0,0.02) init via _init_weights). exp328 tied them here.
        self.apply(self._init_weights)
        # Zero-init output projections so residual branches start as identity.
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp[-1].weight)

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


def setup_optimizer(model, lr, weight_decay):
    # exp033: the ONLY deviation from exp032's optimizer is that the new FastMHL LUT-table
    # weights (ndim=3) are routed to the NO-weight-decay group -- the project's standing
    # lesson is that LUT tables take no weight decay (near-zero init; wd would fight the
    # sparse table gradient). Every dense-path parameter is grouped EXACTLY as in exp032
    # (2-D -> wd 0.1, 1-D -> wd 0). Same AdamW, same betas, same schedule.
    lut_param_ids = {id(p) for m in model.modules() if isinstance(m, FastMultiHeadLut)
                     for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in lut_param_ids or p.ndim < 2:
            nodecay.append(p)
        else:
            decay.append(p)
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
model = MinimalGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN}')
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
