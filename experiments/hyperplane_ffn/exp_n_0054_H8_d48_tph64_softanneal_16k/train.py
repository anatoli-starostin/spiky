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
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut       # optimizer isinstance route
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
# experiment-local soft-forward + annealed-temp LUT (replaces the per-head FastMHL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soft_anneal_lut import SoftAnnealLut, anneal_temp

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']

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
LUT_BATCHED = bool(cfg.get('lut_batched_multi_head_input', False))   # True -> batched forward (shared per-head temp)
LUT_FWD    = cfg.get('lut_forward_mode', 'hard')
LUT_BF16   = cfg.get('lut_use_bf16', False)
LUT_NOISE  = cfg.get('lut_init_weights_noise', 1e-3)
LUT_LEARNTEMP = bool(cfg.get('lut_learnable_temps', False))   # learnable T_soft/T_sel per head
LUT_SEED   = cfg.get('lut_base_seed', 1000)

# Soft-anneal knobs: temps decay exponentially start -> floor over all n_steps.
SOFT_START = float(cfg.get('soft_anneal_temp_start', 0.5))
SOFT_FLOOR = float(cfg.get('soft_anneal_temp_floor', 0.02))
# Adaptive handoff: once mean top-1 softmax mass >= this, switch soft -> real FastMHL hard path.
HANDOFF_THRESH = float(cfg.get('handoff_top1_threshold', 0.85))

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')
assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
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
                joint_head_compression=LUT_JOINT,
                batched_multi_head_input=LUT_BATCHED,   # batched forward path (control vs exp_g_0006 loop path)
                forward_mode=LUT_FWD,
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


def set_hard_mode(luts, hard):
    prev = [l.hard for l in luts]
    for l in luts:
        l.hard = hard
    return prev


def restore_modes(luts, prev):
    for l, h in zip(luts, prev):
        l.hard = h


@torch.no_grad()
def model_soft_hard_gap(model, luts, x):
    """End-to-end relative L2 gap between soft-mode and hard-mode logits on batch x."""
    was = model.training
    model.eval()
    prev = set_hard_mode(luts, False); ls = model(x)
    set_hard_mode(luts, True);         lh = model(x)
    restore_modes(luts, prev)
    if was:
        model.train()
    num = (ls - lh).pow(2).sum().sqrt()
    den = lh.pow(2).sum().sqrt() + 1e-8
    return float((num / den).item())


model = MinimalGPT(vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN).to(DEVICE)

# --- Substitute SoftAnnealLut for each per-head FastMHL in every CompressionMHL FFN slot.
# The SoftAnnealLut copies each FastMHL's tables/anchors/bit_matrix (init bit-identical to the
# exp_n_0052 hard control) but runs the differentiable soft weighted-sum forward with annealed
# temps. Requires the independent per-head loop path (lut_batched_multi_head_input=false) so
# block.ffn.luts is a ModuleList we can swap into. ---
soft_luts = []
for block in model.blocks:
    if getattr(block, 'ffn_type', None) != 'compression':
        continue
    ffn = block.ffn
    assert hasattr(ffn, 'luts'), (
        "soft-anneal needs the per-head loop path; set lut_batched_multi_head_input=false")
    for h in range(len(ffn.luts)):
        sal = SoftAnnealLut(ffn.luts[h], temp_init=SOFT_START).to(DEVICE)
        ffn.luts[h] = sal           # nn.ModuleList __setitem__ re-registers the module
        soft_luts.append(sal)
print(f'soft-anneal: swapped {len(soft_luts)} per-head FastMHL -> SoftAnnealLut '
      f'(temps anneal {SOFT_START} -> {SOFT_FLOOR}, exp decay over n_steps)')

total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN} | ffn={FFN_TYPE} tie={TIE} gamma={GAMMA}')
print(f'Params: {total_params:,}')

if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)
if os.environ.get('SMOKE_STEPS'):          # inert for the real run; short sanity only
    N_STEPS = int(os.environ['SMOKE_STEPS'])
    EVAL_EVERY = max(1, N_STEPS)
    print(f'*** SMOKE_STEPS={N_STEPS}: short sanity run ***')

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb', 'temp'])
train_losses_logged, val_bpbs, val_steps = [], [], []
ema, best_bpb, t0 = None, float('inf'), time.time()
handed_off, handoff_info, mean_top1 = False, {}, 0.0

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, N_STEPS, WARMUP_FRAC)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale
    # anneal the soft temperatures (same value for both)
    t_cur = anneal_temp(step, N_STEPS, SOFT_START, SOFT_FLOOR)
    for sal in soft_luts:
        sal.set_temps(t_cur, t_cur)
    # ADAPTIVE HANDOFF: once the soft blend is sharp enough (mean top-1 softmax mass
    # >= threshold, i.e. soft output ~= hard output), switch every slot to the real
    # FastMHL hard-forward/soft-backward path and continue training from there.
    if (not handed_off) and mean_top1 >= HANDOFF_THRESH:
        xp, _ = next(train_loader)
        gap = model_soft_hard_gap(model, soft_luts, xp)      # end-to-end soft-vs-hard logit gap
        for sal in soft_luts:
            sal.set_hard(seed_fmhl_temps_to=t_cur)           # continuity: seed learnable temps to t_cur
        handed_off = True
        handoff_info = {'step': step, 'mean_top1': round(mean_top1, 4),
                        'temp': round(t_cur, 5), 'soft_hard_logit_gap': round(gap, 5)}
        print(f'*** HANDOFF at step {step}: mean_top1={mean_top1:.4f} temp={t_cur:.5f} '
              f'soft->hard logit_gap={gap:.5f} -> real FastMHL STE path ***')
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if not handed_off:
        mean_top1 = sum(s.last_top1 for s in soft_luts) / len(soft_luts)
    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss
    mode = 'hard' if handed_off else 'soft'
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | {mode} temp={t_cur:.4f} top1={mean_top1:.4f} | lr={lr_scale * LR:.2e}')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        # TRUE hard-eval bpb: always evaluate through the real FastMHL hard path.
        prev = set_hard_mode(soft_luts, True)
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        model.train()
        restore_modes(soft_luts, prev)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: hard_bpb={bpb:.4f} | {mode} temp={t_cur:.4f} top1={mean_top1:.4f}')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}', f'{t_cur:.6f}']); csv_f.flush()

csv_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)'); ax1.set(xlabel='step', ylabel='ce loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb'); ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'total_params': total_params,
           'soft_anneal_temp_start': SOFT_START, 'soft_anneal_temp_floor': SOFT_FLOOR,
           'handoff_top1_threshold': HANDOFF_THRESH, 'n_soft_luts': len(soft_luts),
           'handed_off': handed_off, 'handoff': handoff_info,
           'training_time_hours': round(elapsed / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
