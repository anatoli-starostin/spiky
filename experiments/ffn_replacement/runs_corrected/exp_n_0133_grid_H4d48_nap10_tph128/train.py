# =============================================================================
# TRAINER CORRECTED POST-HOC on 2026-09-07 -- READ THIS BEFORE USING ANY NUMBER
# =============================================================================
# This file is NOT what produced this run's recorded metrics.csv / summary.json.
#
# As actually run, it evaluated with the OLD batch-coupled protocol: the val
# loader was built at the TRAINING device_batch_size and scored eval_steps=10,
# so this run's in-run bpb was measured on device_batch_size x 512 x 10 tokens
# -- a window that differed between runs. That is the bug this branch exists to
# fix (see ../../FIXED_EVAL.md).
#
# The eval path here has since been rewritten to delegate to the shared
# tools/fixed_eval.py (bs48 x 100, leading 12 rows skipped, clone-then-score,
# one eval set feeding both the training curve and the final summary), so that
# runs_corrected/ contains only corrected-eval trainers and forking any file in
# this tree cannot silently reintroduce the coupling.
#
# CONSEQUENCES, stated plainly:
#   * This run's ORIGINALLY-REPORTED in-run bpb (in metrics.csv and
#     summary.json) came from the old eval and is NOT what this file would now
#     produce. Those files are deliberately left untouched as the record.
#   * The AUTHORITATIVE number for this run is in corrected_score.json, which
#     was produced by re-scoring its checkpoint under the fixed protocol.
#   * Re-running this file would produce a corrected curve that does NOT match
#     this folder's metrics.csv. That is expected, not a bug.
#
# The file exactly as it ran is in git history at commit
#   f1dd5e9d71be04942a85760191d24d29256f4408
# (`git show f1dd5e9d:experiments/ffn_replacement/runs_corrected/exp_n_0133_grid_H4d48_nap10_tph128/train.py`).
# =============================================================================
"""Flexible FFN-slot sweep trainer (shared across the exp043+ CompressionMHL sweep).

MinimalGPT + RoPE, config-driven. The FFN slot of every block is one of:
  * ffn_type="dense"        -> the vanilla 384->1536->384 GELU MLP (baselines).
  * ffn_type="compression"  -> x = x + gamma*Linear(384->384)(h) + CompressionMultiHeadLUT(h),
                               h = ln2(x); gamma in {0,1}; CompressionMHL params in config.
Unembedder is tied (lm_head.weight = tok_emb.weight) when tie_unembedder=True, else untied.

Outputs alongside: metrics.csv, summary.json, loss.png, checkpoint.pt.
"""
import sys, os, json, math, time, csv
# Live-log fix: force line-buffered stdout so `step N |` / [VAL] lines reach
# run.log immediately even when stdout is redirected to a file (block-buffered by
# default). Belt-and-suspenders with `python -u` / PYTHONUNBUFFERED=1 at launch.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
# CORRECTED EVAL (2026-09-07): the shared, batch-size-independent eval set. Replaces
# `from nanochat.loss_eval import evaluate_bpb`, which this file used with a val loader
# built at the training device_batch_size. Delegating to tools/fixed_eval.py rather than
# inlining the window keeps every trainer in this tree on one definition.
def _find_tools():
    _d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(_d, 'tools', 'fixed_eval.py')):
            return os.path.join(_d, 'tools')
        _d = os.path.dirname(_d)
    raise RuntimeError('could not locate ffn_replacement/tools/ (fixed_eval.py)')


sys.path.insert(0, _find_tools())
from fixed_eval import evaluate_bpb_fixed, eval_config

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut       # optimizer isinstance route
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY = cfg['eval_every']
# `cfg['eval_steps']` is the LEGACY key and is deliberately NOT read: eval_config ignores
# it, so no old config can re-couple the eval window to the training batch size.
EVAL = eval_config(cfg)   # fixed eval: bs48 x 100, skip 12 (NOT a function of DEVICE_BS)

FFN_TYPE = cfg.get('ffn_type', 'compression')          # "dense" | "compression"
GAMMA    = int(cfg.get('gamma', 0))                    # 0/1: parallel Linear(384->384) path
TIE      = bool(cfg.get('tie_unembedder', False))

# CompressionMHL knobs (only used when ffn_type == "compression")
LUT_IN     = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))    # -1 => no compress
LUT_OUT    = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))   # -1 => no decompress
LUT_NAP    = cfg.get('lut_n_anchor_pairs')
LUT_TPH    = cfg.get('lut_tables_per_head')
LUT_HEADS  = cfg.get('lut_n_heads', 1)
LUT_JOINT  = cfg.get('lut_joint_head_compression', False)
LUT_FWD    = cfg.get('lut_forward_mode', 'hard')
LUT_BF16   = cfg.get('lut_use_bf16', False)
LUT_NOISE  = cfg.get('lut_init_weights_noise', 1e-3)
LUT_LEARNTEMP = bool(cfg.get('lut_learnable_temps', False))   # learnable T_soft/T_sel per head
LUT_SEED   = cfg.get('lut_base_seed', 1000)

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')
assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
# (removed 2026-09-07) the val loader was built here at DEVICE_BS -- the bug itself.
# evaluate_bpb_fixed builds its own loader at the fixed eval batch size.
token_bytes = get_token_bytes(device=DEVICE)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
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
    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = FFN_TYPE
        if FFN_TYPE == 'dense':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd, bias=False))
        else:
            self.lin = nn.Linear(n_embd, n_embd, bias=True) if GAMMA == 1 else None
            self.ffn = CompressionMultiHeadLUT(
                input_dim=n_embd, output_dim=n_embd,
                inner_in_dim=LUT_IN, inner_out_dim=LUT_OUT,
                nap=LUT_NAP, tph=LUT_TPH, n_heads=LUT_HEADS,
                joint_head_compression=LUT_JOINT, forward_mode=LUT_FWD,
                use_bf16=LUT_BF16, initial_weights_noise=LUT_NOISE,
                learnable_temps=LUT_LEARNTEMP,
                random_seed=LUT_SEED + layer_idx)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        out = self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)
        if self.lin is not None:
            out = out + self.lin(h)
        return x + out


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            if FFN_TYPE == 'dense':
                nn.init.zeros_(block.mlp[-1].weight)
            else:
                if block.ffn.has_decompress:
                    nn.init.zeros_(block.ffn.decompress.weight)   # FFN-slot output proj
                if block.lin is not None:
                    nn.init.zeros_(block.lin.weight)              # parallel Linear (gamma=1)
        if TIE:
            self.head.weight = self.tok_emb.weight                # weight sharing

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def get_device(self):
        return self.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


def setup_optimizer(model, lr, weight_decay):
    lut_ids = {id(p) for m in model.modules() if isinstance(m, FastMultiHeadLut)
               for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
    groups = [dict(params=decay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay),
              dict(params=nodecay, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    n_lut = sum(m.weights.numel() for m in model.modules() if isinstance(m, FastMultiHeadLut))
    print(f"AdamW (0033 grouping) | decay(2-D weights)={sum(p.numel() for p in decay):,} wd={weight_decay} | "
          f"nodecay(LUT tables+temps+1-D)={sum(p.numel() for p in nodecay):,} wd=0 | lr={lr} betas=(0.9, 0.95) "
          f"eps=1e-8 [LUT tables={n_lut:,} in nodecay]")
    return opt


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


model = MinimalGPT(vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN} | ffn={FFN_TYPE} tie={TIE} gamma={GAMMA}')
print(f'Params: {total_params:,}')

if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb'])
train_losses_logged, val_bpbs, val_steps = [], [], []
ema, best_bpb, t0 = None, float('inf'), time.time()

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
        bpb = evaluate_bpb_fixed(model, tokenizer, token_bytes, SEQ_LEN, DEVICE, **EVAL)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}']); csv_f.flush()
        model.train()

csv_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)'); ax1.set(xlabel='step', ylabel='ce loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb'); ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'total_params': total_params, 'training_time_hours': round(elapsed / 3600, 3),
           'eval_protocol': {'eval_batch_size': EVAL['eval_batch_size'],
                             'eval_steps': EVAL['eval_steps'],
                             'skip_rows': EVAL['skip_rows'],
                             'batch_size_independent': True}}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
