"""exp_n_0055 PHASE 2 — swap-in-then-finetune, to decompose the swap-in gap.

Phase 1 (here, reproducing the standalone distill.py): distill 6 CompressionMHLs
to imitate the frozen dense model's per-block FFNs (MSE, hard/soft STE), swap them
in, measure the RAW frozen swap-in val_bpb.

Phase 2: with the 6 distilled LUT-FFNs swapped in, FREEZE the LUTs, the input
embedding, and all LayerNorms; UNFREEZE only the attention layers + the unembedder
(output projection). Because the dense baseline ties head.weight == tok_emb.weight,
we UNTIE the head (give it its own trainable copy) so we can finetune the unembedder
while keeping the input embedding frozen — the literal reading of "freeze embeddings,
finetune unembedder". Finetune on the standard LM CE objective and log the val_bpb
recovery curve. This isolates how much of the swap-in degradation attention+unembedder
co-adaptation can repair WITHOUT touching the LUTs.

No shared-module edits.
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__)); HFF = os.path.dirname(EXP_DIR)
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS = cfg['device_batch_size']
N_STEPS, LR, WD, WARMUP_FRAC = cfg['n_steps'], cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
LOG_EVERY, EVAL_STEPS, TIE = cfg['log_every'], cfg['eval_steps'], bool(cfg['tie_unembedder'])
FT_STEPS, FT_LR = int(cfg['ft_steps']), float(cfg['ft_lr'])
FT_WARMUP, FT_EVAL_EVERY = float(cfg['ft_warmup_frac']), int(cfg['ft_eval_every'])
L = dict(inner_in=cfg['lut_inner_in_dim'], inner_out=cfg['lut_inner_out_dim'], nap=cfg['lut_n_anchor_pairs'],
         tph=cfg['lut_tables_per_head'], heads=cfg['lut_n_heads'], joint=cfg['lut_joint_head_compression'],
         batched=cfg['lut_batched_multi_head_input'], fwd=cfg['lut_forward_mode'], bf16=cfg['lut_use_bf16'],
         noise=cfg['lut_init_weights_noise'], seed=cfg['lut_base_seed'], learn_t=cfg['lut_learnable_temps'])

if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS']); LOG_EVERY = max(1, N_STEPS // 4)
    FT_STEPS = int(os.environ.get('SMOKE_FT', 120)); FT_EVAL_EVERY = max(1, FT_STEPS // 3)
    print(f'*** SMOKE: distill {N_STEPS} / finetune {FT_STEPS} ***')

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
        if TIE:
            self.head.weight = self.tok_emb.weight
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


# ---------------- load frozen dense ----------------
model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(HFF, cfg['dense_ckpt']), map_location=DEVICE), strict=False)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()
dense_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
print(f'[CHECK] dense val_bpb = {dense_bpb:.7f} (target {cfg["dense_val_bpb"]:.7f})')

# ---------------- PHASE 1: distill 6 LUTs (or load cached) ----------------
CACHE = os.path.join(EXP_DIR, 'distilled_luts.pt')
luts = [make_cmhl(b) for b in range(DEPTH)]
caps = [None] * DEPTH
def make_hook(b):
    def hook(m, inp, out):
        caps[b] = (inp[0].reshape(-1, N_EMBD).detach(), out.reshape(-1, N_EMBD).detach())
    return hook
hooks = [blk.mlp.register_forward_hook(make_hook(b)) for b, blk in enumerate(model.blocks)]

if os.path.exists(CACHE) and not os.environ.get('SMOKE_STEPS'):
    st = torch.load(CACHE, map_location=DEVICE)
    for b in range(DEPTH):
        luts[b].load_state_dict(st['luts'][b])
    final_mse = st['final_mse']; print(f'loaded cached distilled LUTs from {CACHE}')
else:
    lut_ids = {id(p) for lu in luts for m in lu.modules() if isinstance(m, FastMultiHeadLut) for p in m.parameters(recurse=False)}
    dec, ndec = [], []
    for lu in luts:
        for p in lu.parameters():
            (ndec if (id(p) in lut_ids or p.ndim < 2) else dec).append(p)
    opt = torch.optim.AdamW([dict(params=dec, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WD),
                             dict(params=ndec, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    ema = [None] * DEPTH
    for lu in luts:
        lu.train()
    for step in range(1, N_STEPS + 1):
        sc = cos_lr(step, N_STEPS, WARMUP_FRAC)
        for g in opt.param_groups:
            g['lr'] = g['initial_lr'] * sc
        with torch.no_grad():
            x, _ = next(train_loader); model(x)
        opt.zero_grad(set_to_none=True)
        for b in range(DEPTH):
            hin, hout = caps[b]
            loss = F.mse_loss(luts[b](hin), hout); loss.backward()
            ema[b] = loss.item() if ema[b] is None else 0.99 * ema[b] + 0.01 * loss.item()
        torch.nn.utils.clip_grad_norm_([p for lu in luts for p in lu.parameters()], 1.0)
        opt.step()
        if step % LOG_EVERY == 0 or step == 1 or step == N_STEPS:
            print(f'[distill] step {step:6d} MSE ' + ' '.join(f'b{b}={ema[b]:.4f}' for b in range(DEPTH)))
    final_mse = {f'b{b}': round(ema[b], 6) for b in range(DEPTH)}
    if not os.environ.get('SMOKE_STEPS'):          # never persist smoke-quality LUTs to the real cache
        torch.save({'luts': [lu.state_dict() for lu in luts], 'final_mse': final_mse}, CACHE)
        print(f'saved distilled LUTs -> {CACHE}')
for h in hooks:
    h.remove()

# ---------------- swap LUTs in, measure RAW frozen swap-in bpb ----------------
for b, blk in enumerate(model.blocks):
    luts[b].eval(); blk.mlp = LUTAdapter(luts[b])
model.eval()
raw_swap_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
print(f'\n=== PHASE-1 raw frozen swap-in val_bpb = {raw_swap_bpb:.7f} (dense {dense_bpb:.7f}, delta {raw_swap_bpb-dense_bpb:+.5f}) ===')
print('per-block distill MSE:', final_mse)

# ---------------- PHASE 2: freeze LUTs+emb+LNs, untie+finetune attn+unembedder ----------------
for p in model.parameters():
    p.requires_grad_(False)                                   # freeze everything (incl LUTs, tok_emb, LNs)
if model.head.weight is model.tok_emb.weight:                 # untie so we can train unembedder w/ frozen emb
    model.head.weight = nn.Parameter(model.tok_emb.weight.detach().clone())
    print('untied head from tok_emb for phase-2 (embedding stays frozen, unembedder trainable)')
for blk in model.blocks:
    for p in blk.attn.parameters():
        p.requires_grad_(True)                                # attention layers
model.head.weight.requires_grad_(True)                        # unembedder

# verify freeze/unfreeze
lut_grad = any(p.requires_grad for lu in luts for p in lu.parameters())
emb_grad = model.tok_emb.weight.requires_grad
ln_grad = any(p.requires_grad for b in model.blocks for p in list(b.ln1.parameters()) + list(b.ln2.parameters())) or any(p.requires_grad for p in model.ln_f.parameters())
attn_grad = all(p.requires_grad for b in model.blocks for p in b.attn.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[FREEZE CHECK] LUTs.requires_grad={lut_grad} (want False) | tok_emb.requires_grad={emb_grad} (want False) | '
      f'LNs.requires_grad={ln_grad} (want False) | attn all trainable={attn_grad} (want True) | head trainable={model.head.weight.requires_grad}')
print(f'[FREEZE CHECK] phase-2 trainable params = {n_train:,} (attn 6x{sum(p.numel() for p in model.blocks[0].attn.parameters()):,} + head {model.head.weight.numel():,})')

ft_params = [p for p in model.parameters() if p.requires_grad]
ft_opt = torch.optim.AdamW([dict(params=ft_params, lr=FT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WD)])
for g in ft_opt.param_groups:
    g['initial_lr'] = g['lr']

ft_csv = open(os.path.join(EXP_DIR, 'phase2_metrics.csv'), 'w', newline='')
ft_w = csv.writer(ft_csv); ft_w.writerow(['ft_step', 'train_ce', 'val_bpb'])
ft_steps_log, ft_bpb_log = [0], [raw_swap_bpb]      # start point = raw swap-in
ft_w.writerow([0, '', f'{raw_swap_bpb:.6f}']); ft_csv.flush()
ema_ce, best_ft, t0 = None, raw_swap_bpb, time.time()
model.train()
for step in range(1, FT_STEPS + 1):
    sc = cos_lr(step, FT_STEPS, FT_WARMUP)
    for g in ft_opt.param_groups:
        g['lr'] = g['initial_lr'] * sc
    x, y = next(train_loader)
    loss = model(x, y)
    ft_opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(ft_params, 1.0); ft_opt.step()
    ema_ce = loss.item() if ema_ce is None else 0.99 * ema_ce + 0.01 * loss.item()
    if step % FT_EVAL_EVERY == 0 or step == FT_STEPS:
        model.eval(); b = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes); model.train()
        best_ft = min(best_ft, b); ft_steps_log.append(step); ft_bpb_log.append(b)
        ft_w.writerow([step, f'{ema_ce:.6f}', f'{b:.6f}']); ft_csv.flush()
        print(f'[finetune] step {step:5d} | ce={ema_ce:.4f} | val_bpb={b:.5f} (start {raw_swap_bpb:.5f})')
ft_csv.close()
final_ft_bpb = ft_bpb_log[-1]

# ---------------- plots ----------------
plt.figure(figsize=(9, 6))
plt.plot(ft_steps_log, ft_bpb_log, 'o-', color='tab:blue', label='attn+unembed finetune (val_bpb)')
plt.axhline(raw_swap_bpb, ls=':', color='tab:red', label=f'raw swap-in {raw_swap_bpb:.4f}')
plt.axhline(cfg['dense_val_bpb'], ls='--', color='gray', label=f'dense {cfg["dense_val_bpb"]:.4f}')
plt.axhline(1.2285517, ls='--', color='tab:green', label='end-to-end LUT exp_n_0052 1.2286')
plt.xlabel('finetune step'); plt.ylabel('val_bpb'); plt.grid(True, alpha=0.3)
plt.title(f'exp_n_0055 phase-2: recover swap-in by finetuning attn+unembed\n{raw_swap_bpb:.4f} -> {final_ft_bpb:.4f} (best {best_ft:.4f})')
plt.legend(fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'phase2_recovery.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'] + '_phase2', 'dense_val_bpb': cfg['dense_val_bpb'], 'loaded_dense_val_bpb': dense_bpb,
           'phase1_raw_swapin_bpb': raw_swap_bpb, 'phase1_delta_vs_dense': raw_swap_bpb - cfg['dense_val_bpb'],
           'phase1_block_mse': final_mse, 'phase2_ft_steps': FT_STEPS, 'phase2_ft_lr': FT_LR,
           'phase2_final_bpb': final_ft_bpb, 'phase2_best_bpb': best_ft,
           'phase2_recovered_delta_vs_dense': final_ft_bpb - cfg['dense_val_bpb'],
           'phase2_vs_e2e_lut_0052': final_ft_bpb - 1.2285517,
           'phase2_trainable_params': n_train, 'training_time_hours': round((time.time() - t0) / 3600, 3)}
with open(os.path.join(EXP_DIR, 'phase2_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n=== DONE (phase 2) ==='); print(json.dumps(summary, indent=2))
