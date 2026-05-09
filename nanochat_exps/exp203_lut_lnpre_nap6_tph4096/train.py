"""nanochat_exps/exp200_v2d_lut_perlayer_pos.

Fork of exp199 (LUT tph=4096, smooth=False, n_alt=1). Single change: replace
the input-only positional embedding with a per-layer positional embedding.
Each block has its own learned pos_emb [SEQ_LEN, E] which is added to the
residual stream at the start of the block (before LN), so the LUT and
attention path see fresh position info at every layer.

  per block:  x = x + pos_emb_layer
              x = x + scatter(lut(SDPA(V2D(LN(x)))))

Adds DEPTH * SEQ_LEN * E = 6 * 512 * 384 = 1.18M new params (small vs LUTs).
LUT settings (nap=6, tph=4096, n_sparse=32, n_alt=1, smooth=False) unchanged.

Launch:
    PYTHONPATH=/home/starost/nanochat \\
      /home/starost/spiky/.venv/bin/python -u \\
      nanochat_exps/exp200_v2d_lut_perlayer_pos/train.py \\
      > nanochat_exps/exp200_v2d_lut_perlayer_pos/stdout.log 2>&1 &
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

from spiky.lutorch.ranking_tools import VectorToDominance, SparseScatter
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

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
LUT_NAP       = cfg.get('lut_input_nap', 6)
LUT_TPH       = cfg.get('lut_tph', 384)
LUT_N_SPARSE  = cfg.get('lut_n_sparse', 32)
LUT_N_ALT     = cfg.get('lut_n_alternatives', 3)
LUT_INIT_STD  = cfg.get('lut_init_std', 0.001)

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
    """Single sub-layer block:
       LN -> V2D -> per-head Q/K/V -> SDPA -> LUT(+SparseScatter) -> residual.

    The LUT replaces both the attention out_proj AND the FFN: a single
    MultiHeadLut consumes the post-SDPA E-dim vector and emits per-table
    sparse outputs, which SparseScatter sums into a dense E-dim residual."""
    def __init__(self, layer_idx: int):
        super().__init__()
        # Per-layer positional embedding: added to the residual stream at
        # the start of every block so each layer sees fresh position info.
        pos_gen = torch.Generator().manual_seed(cfg['random_seed'] + 5000 + layer_idx)
        self.pos_emb = nn.Parameter(torch.randn(SEQ_LEN, E, generator=pos_gen) * 0.02)
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
        # Single LUT replacing out_proj + FFN: MultiHeadLut + SparseScatter.
        # Input: post-SDPA [B*T, E]. Per-table outputs of size LUT_N_SPARSE
        # are scattered into the final E-dim residual.
        self.lut = MultiHeadLut(
            input_dim=E,
            n_heads=1,
            n_outputs=LUT_N_SPARSE,
            n_anchor_pairs=LUT_NAP,
            tables_per_head=LUT_TPH,
            n_alternatives=LUT_N_ALT,
            smooth_mode=cfg.get('lut_smooth_mode', False),
            anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            random_seed=cfg['random_seed'] + 3000 + layer_idx,
            initial_weights_noise=LUT_INIT_STD,
            return_per_table_outputs=True,
            device=torch.device(DEVICE),
        )
        self.scatter = SparseScatter(
            n_heads=1,
            tables_per_head=LUT_TPH,
            n_sparse_outputs=LUT_N_SPARSE,
            n_outputs=E,
            seed=cfg['random_seed'] + 4000 + layer_idx,
            device=torch.device(DEVICE),
        )
        # LayerNorm before LUT: stabilizes the input distribution that the
        # LUT's pair-comparison anchors are computed against.
        self.ln_lut = nn.LayerNorm(E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        B, T, _ = x.shape
        x = x + self.pos_emb[:T]                               # per-layer pos
        x_norm = self.ln(x)                                    # pre-norm
        x_h = x_norm.unsqueeze(-2).expand(B, T, H, E)          # [B, T, H, E]
        dom = self.v2d(x_h)                                    # [B, T, H, D]
        q = torch.einsum('bthd,hde->bthe', dom, self.Wq)
        k = torch.einsum('bthd,hde->bthe', dom, self.Wk)
        v = torch.einsum('bthd,hde->bthe', dom, self.Wv)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, E_h]
        attn = attn.transpose(1, 2).contiguous().view(B, T, E)          # [B, T, E]
        # LayerNorm before LUT (out_proj) so pair comparisons see a stable
        # input distribution.
        attn_ln = self.ln_lut(attn)
        per_table = self.lut(attn_ln.view(B * T, E))           # [B*T, 1, tph, n_sparse]
        scattered = self.scatter(per_table)                    # [B*T, 1, E]
        out = scattered.view(B, T, E)
        return x + out                                          # single residual


class V2DSDPATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, E)
        # No input-level positional embedding; each block has its own pos_emb.
        self.blocks  = nn.ModuleList([V2DSDPABlock(i) for i in range(DEPTH)])
        self.ln_f    = nn.LayerNorm(E)
        self.head    = nn.Linear(E, VOCAB_SIZE, bias=False)
        # Untied head: head.weight is its own [VOCAB, E] tensor (no sharing
        # with tok_emb). Init at std=0.02.
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight,    std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        x = self.tok_emb(idx)                                  # [B, T, E]
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
    """AdamW with three groups:
       - decay: ndim>=2 weights that are NOT LUT cell weights.
       - lut_nodecay: MultiHeadLut.projection.weights (rare-cell preservation,
         per exp191 finding).
       - nodecay: ndim<2 (LayerNorm gain/bias).
    """
    lut_weight_ids = set()
    for m in model.modules():
        if isinstance(m, MultiHeadLut):
            lut_weight_ids.add(id(m.projection.weights))
    decay       = [p for p in model.parameters()
                   if p.ndim >= 2 and id(p) not in lut_weight_ids]
    lut_nodecay = [p for p in model.parameters() if id(p) in lut_weight_ids]
    nodecay     = [p for p in model.parameters() if p.ndim < 2]
    print(f'AdamW groups: decay={sum(p.numel() for p in decay):,} '
          f'lut_nodecay={sum(p.numel() for p in lut_nodecay):,} '
          f'nodecay={sum(p.numel() for p in nodecay):,}')
    groups = [
        dict(params=decay,       lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
        dict(params=lut_nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(params=nodecay,     lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
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
print(f'LUT: nap={LUT_NAP} (2^nap={2**LUT_NAP} cells), tph={LUT_TPH}, '
      f'n_sparse={LUT_N_SPARSE}, n_alt={LUT_N_ALT}, '
      f'smooth={cfg.get("lut_smooth_mode", False)}')
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
