"""exp_n_0056 — LUT-representability-constrained dense training.

Train a vanilla dense transformer from scratch on the LM objective, but constrain
each block's FFN to stay reproducible by a LUT of fixed (exp_n_0052) capacity. In
parallel, co-train one CompressionMHL per block:

  per block b, per step (x_b = ln2(x) fed to the FFN; ffn_b(x_b) = dense FFN out):
    loss_lut_b = MSE(lut_b(x_b), ffn_b(x_b).detach())    # trains the LUT only (FFN detached)
    loss_reg_b = MSE(ffn_b(x_b), lut_b(x_b).detach())    # pulls FFN/upstream toward the LUT (LUT detached)
    total = CE_LM + Σ_b loss_lut_b + λ·Σ_b loss_reg_b

The dense FFN gives a smooth differentiable optimization path while being regularized
to stay LUT-friendly. λ ramps 0 → λ_reg_target over training. The LUT-side losses are
computed on a random SUBSAMPLE of `lut_batch_tokens` token-vectors per block per step
(dense CE stays at the full device_bs 48 × 512) to keep the STE-surrogate compute light.
LUT hyperparameters are reproduced 1:1 from exp_n_0052. Shared modules untouched.

Metrics: dense-own val_bpb (real FFNs) and swap-in val_bpb (all 6 FFNs replaced by
their co-trained LUTs) each eval, plus per-block FFN<->LUT MSE.
"""
import sys, os, json, math, time, csv
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, TOTAL_BS, N_STEPS = cfg['device_batch_size'], cfg['total_batch_size'], cfg['n_steps']
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS, TIE = cfg['eval_every'], cfg['eval_steps'], bool(cfg['tie_unembedder'])
LAM_TARGET, LAM_RAMP = float(cfg['lambda_reg_target']), float(cfg['lambda_ramp_frac'])
LAMBDA_ANNEAL = bool(cfg.get('lambda_anneal', True))   # False -> constant lambda=target from step 0
LUT_BATCH = int(cfg['lut_batch_tokens'])
CKPT_EVERY = int(cfg.get('ckpt_every', 4000))
# Decoupled batch sizes (asymmetric-batch variant). Default = lut_batch_tokens (symmetric).
REG_BATCH = int(cfg.get('reg_batch_tokens', LUT_BATCH))          # tokens for the FFN->LUT regularizer
LUT_FIT_BATCH = int(cfg.get('lut_fit_batch_tokens', LUT_BATCH))  # tokens for the LUT-fitting loss
L = dict(inner_in=cfg['lut_inner_in_dim'], inner_out=cfg['lut_inner_out_dim'], nap=cfg['lut_n_anchor_pairs'],
         tph=cfg['lut_tables_per_head'], heads=cfg['lut_n_heads'], joint=cfg['lut_joint_head_compression'],
         batched=cfg['lut_batched_multi_head_input'], fwd=cfg['lut_forward_mode'], bf16=cfg['lut_use_bf16'],
         noise=cfg['lut_init_weights_noise'], seed=cfg['lut_base_seed'], learn_t=cfg['lut_learnable_temps'])

if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS']); EVAL_EVERY = max(1, N_STEPS // 2)
    print(f'*** SMOKE_STEPS={N_STEPS} ***')

BASE_DIR = get_base_dir(); tokenizer = RustBPETokenizer.from_directory(os.path.join(BASE_DIR, 'tokenizer'))
VOCAB_SIZE = tokenizer.get_vocab_size(); assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False); self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1); return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)

class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__(); self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False); self.proj = nn.Linear(n_embd, n_embd, bias=False)
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
        self.ln1 = nn.LayerNorm(n_embd); self.attn = MinimalAttention(n_embd, n_head); self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(), nn.Linear(4 * n_embd, n_embd, bias=False))
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))

class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight); nn.init.zeros_(block.mlp[-1].weight)
        if TIE:
            self.head.weight = self.tok_emb.weight
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
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=loss_reduction, ignore_index=-1)
        return logits

class LUTAdapter(nn.Module):
    def __init__(self, cmhl):
        super().__init__(); self.cmhl = cmhl
    def forward(self, h):
        B, T, C = h.shape
        return self.cmhl(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)

def make_cmhl(b):
    return CompressionMultiHeadLUT(input_dim=N_EMBD, output_dim=N_EMBD, inner_in_dim=L['inner_in'], inner_out_dim=L['inner_out'],
        nap=L['nap'], tph=L['tph'], n_heads=L['heads'], joint_head_compression=L['joint'], batched_multi_head_input=L['batched'],
        forward_mode=L['fwd'], use_bf16=L['bf16'], initial_weights_noise=L['noise'], learnable_temps=L['learn_t'], random_seed=L['seed'] + b).to(DEVICE)

def cos_lr(step, n, warm):
    w = int(warm * n)
    if step < w:
        return step / max(w, 1)
    p = (step - w) / max(n - w, 1); return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

def lam_at(step):
    if not LAMBDA_ANNEAL:
        return LAM_TARGET                                     # constant from step 0 (no ramp)
    frac = min(step / max(int(LAM_RAMP * N_STEPS), 1), 1.0)   # linear ramp 0 -> target
    return LAM_TARGET * frac

model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
luts = [make_cmhl(b) for b in range(DEPTH)]

# hooks capture per-block FFN (input x_b, output ffn_b(x_b)) WITH grad graph
caps = [None] * DEPTH
def make_hook(b):
    def hook(m, inp, out):
        caps[b] = (inp[0].reshape(-1, N_EMBD), out.reshape(-1, N_EMBD))
    return hook
for b, blk in enumerate(model.blocks):
    blk.mlp.register_forward_hook(make_hook(b))
dense_mlps = [blk.mlp for blk in model.blocks]     # keep refs to restore after swap-in eval

# optimizer over dense + LUT params (0033 grouping)
lut_ids = {id(p) for lu in luts for m in lu.modules() if isinstance(m, FastMultiHeadLut) for p in m.parameters(recurse=False)}
decay, nodecay = [], []
for p in list(model.parameters()) + [p for lu in luts for p in lu.parameters()]:
    if not p.requires_grad:
        continue
    (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
opt = torch.optim.AdamW([dict(params=decay, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WD),
                         dict(params=nodecay, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])
for g in opt.param_groups:
    g['initial_lr'] = g['lr']
dense_params = sum(p.numel() for p in model.parameters())
lut_params = sum(p.numel() for lu in luts for p in lu.parameters())
print(f'PARAM COUNTS | dense (deployable)={dense_params:,} | 6 LUTs (co-trained)={lut_params:,}')
print(f'LUT config (== exp_n_0052): H{L["heads"]}/d{L["inner_in"]}/tph{L["tph"]}/nap{L["nap"]} batched={L["batched"]} | '
      f'lambda_target={LAM_TARGET} anneal={LAMBDA_ANNEAL} ramp_frac={LAM_RAMP} | lut_fit_batch={LUT_FIT_BATCH} reg_batch={REG_BATCH} '
      f'({"symmetric" if REG_BATCH==LUT_FIT_BATCH else "ASYMMETRIC"}) | dense CE full {DEVICE_BS*SEQ_LEN})')

@torch.no_grad()
def swapin_bpb():
    for b, blk in enumerate(model.blocks):
        luts[b].eval(); blk.mlp = LUTAdapter(luts[b])
    model.eval(); v = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
    for b, blk in enumerate(model.blocks):
        blk.mlp = dense_mlps[b]                     # restore real FFN
    model.train(); return v

grad_accum = max(1, TOTAL_BS // (DEVICE_BS * SEQ_LEN))
csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'lambda', 'train_ce', 'imitation_mse', 'reg_mse'] +
               [f'mse_b{b}' for b in range(DEPTH)] + ['dense_bpb', 'swapin_bpb'])
hist = {'step': [], 'dense': [], 'swap': [], 'mse': {b: [] for b in range(DEPTH)}}
mse_ema = [None] * DEPTH
ema_ce = ema_imit = ema_reg = None

def save_ckpt(step, tag='latest'):
    """Reloadable checkpoint: dense model weights + all 6 co-trained LUT state_dicts."""
    path = os.path.join(EXP_DIR, f'checkpoint_{tag}.pt')
    torch.save({'step': step, 'model': model.state_dict(),
                'luts': [lu.state_dict() for lu in luts], 'config': cfg}, path)
    print(f'  [ckpt] saved {path} (dense + 6 LUTs, step {step})')
t0 = time.time()
model.train()
for lu in luts:
    lu.train()
for step in range(1, N_STEPS + 1):
    sc = cos_lr(step, N_STEPS, WARMUP_FRAC)
    for g in opt.param_groups:
        g['lr'] = g['initial_lr'] * sc
    lam = lam_at(step)
    opt.zero_grad(set_to_none=True)
    x, y = next(train_loader)
    ce = model(x, y)                                  # hooks fill caps[b]=(x_b, ffn_out_b) with grad
    ntok = caps[0][0].shape[0]
    sym = (REG_BATCH == LUT_FIT_BATCH)
    idx_fit = torch.randperm(ntok, device=DEVICE)[:min(LUT_FIT_BATCH, ntok)]
    idx_reg = idx_fit if sym else torch.randperm(ntok, device=DEVICE)[:min(REG_BATCH, ntok)]
    loss_lut_sum, loss_reg_sum = 0.0, 0.0
    for b in range(DEPTH):
        x_b, ffn_b = caps[b]
        # LUT-fitting loss on the (small) fit subsample -- grad -> LUT params only
        lut_fit = luts[b](x_b[idx_fit].detach())
        loss_lut = F.mse_loss(lut_fit, ffn_b[idx_fit].detach())
        # Reg loss (pulls FFN/upstream toward the LUT) -- grad -> FFN/upstream only, LUT target detached.
        if sym:
            loss_reg = F.mse_loss(ffn_b[idx_fit], lut_fit.detach())
        else:                                          # asymmetric: reg on a richer (e.g. full) token set
            with torch.no_grad():
                lut_reg = luts[b](x_b[idx_reg].detach())
            loss_reg = F.mse_loss(ffn_b[idx_reg], lut_reg)
        loss_lut_sum = loss_lut_sum + loss_lut
        loss_reg_sum = loss_reg_sum + loss_reg
        m = loss_lut.item()
        mse_ema[b] = m if mse_ema[b] is None else 0.99 * mse_ema[b] + 0.01 * m
    total = ce + loss_lut_sum + lam * loss_reg_sum
    total.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
    opt.step()
    ce_v, imit_v, reg_v = ce.item(), float(loss_lut_sum.item()), float(loss_reg_sum.item())
    ema_ce = ce_v if ema_ce is None else 0.99 * ema_ce + 0.01 * ce_v
    ema_imit = imit_v if ema_imit is None else 0.99 * ema_imit + 0.01 * imit_v
    ema_reg = reg_v if ema_reg is None else 0.99 * ema_reg + 0.01 * reg_v
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | lam={lam:.4f} | CE={ema_ce:.4f} imit={ema_imit:.4f} reg={ema_reg:.4f} | MSE ' + ' '.join(f'b{b}={mse_ema[b]:.4f}' for b in range(DEPTH)))
    if step % CKPT_EVERY == 0:
        save_ckpt(step)
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval(); d_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes); model.train()
        s_bpb = swapin_bpb()
        hist['step'].append(step); hist['dense'].append(d_bpb); hist['swap'].append(s_bpb)
        for b in range(DEPTH):
            hist['mse'][b].append(mse_ema[b])
        csv_w.writerow([step, f'{lam:.5f}', f'{ema_ce:.6f}', f'{ema_imit:.6f}', f'{ema_reg:.6f}'] +
                       [f'{mse_ema[b]:.6f}' for b in range(DEPTH)] + [f'{d_bpb:.6f}', f'{s_bpb:.6f}']); csv_f.flush()
        print(f'[VAL] step {step}: dense_bpb={d_bpb:.5f} swapin_bpb={s_bpb:.5f} (lam={lam:.4f})')
csv_f.close()
save_ckpt(N_STEPS, tag='final')      # dense + 6 LUTs, reloadable without re-distillation

# ---------------- plots ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hist['step'], hist['dense'], '-', color='tab:green', label='dense-own bpb (real FFNs)')
ax1.plot(hist['step'], hist['swap'], '-', color='tab:blue', label='swap-in bpb (LUT FFNs, deployable)')
ax1.axhline(1.196646, ls='--', color='gray', label='dense 1.19665')
ax1.axhline(1.2285517, ls='--', color='tab:red', label='e2e LUT exp_n_0052 1.2286')
ax1.set(xlabel='step', ylabel='val_bpb', title='dense-own vs swap-in bpb'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
for b in range(DEPTH):
    ax2.plot(hist['step'], hist['mse'][b], label=f'block {b} (final {mse_ema[b]:.4f})')
ax2.set(xlabel='step', ylabel='FFN<->LUT MSE', title='per-block imitation MSE'); ax2.set_yscale('log'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'constrained.png'), dpi=120); plt.close()

final_dense = hist['dense'][-1]; final_swap = hist['swap'][-1]
summary = {'exp_name': cfg['exp_name'], 'dense_own_bpb': final_dense, 'swapin_bpb': final_swap,
           'final_block_mse': {f'b{b}': round(mse_ema[b], 6) for b in range(DEPTH)},
           'swapin_delta_vs_dense_1p19665': final_swap - 1.196646,
           'swapin_delta_vs_e2e_lut_0052': final_swap - 1.2285517,
           'lambda_reg_target': LAM_TARGET, 'lut_batch_tokens': LUT_BATCH,
           'reg_batch_tokens': REG_BATCH, 'lut_fit_batch_tokens': LUT_FIT_BATCH, 'lambda_anneal': LAMBDA_ANNEAL,
           'dense_params': dense_params, 'lut_params': lut_params,
           'checkpoint': 'checkpoint_final.pt (dense model + 6 LUT state_dicts)',
           'final_imitation_mse': ema_imit, 'final_reg_mse': ema_reg,
           'training_time_hours': round((time.time() - t0) / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
