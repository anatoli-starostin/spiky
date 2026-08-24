"""exp_n_0078 — asymmetric per-step gradient accumulation to isolate LUT gradient density.

Clean test of the weak-LUT-gradients hypothesis. Each training step:
  * fetch ONE batch of A+B = 72 sequences from a single loader (guarantees 72 DISTINCT
    sequences/step, no A/B overlap), split into A (first 48) and B (last 24).
  * Pass A: full-model backward -> every param's .grad = grad of the mean-over-A loss.
  * Pass B: backward with ALL non-LUT params FROZEN (requires_grad=False) -> only the
    LUT params accumulate B's gradient. Non-LUT .grad is left exactly as A produced it.
  * Assemble the LUT gradient as the token-weighted mean over the full 72 sequences:
        LUT.grad   = W_A * gA + W_B * gB ,  W_A = 48/72 = 2/3 , W_B = 24/72 = 1/3
        nonLUT.grad = gA                                    (baseline 48-seq exposure)
  * one AdamW step.

Net effect: non-LUT params see 48 tokens/step (identical to the exp_n_0052 1x baseline),
LUT params see the mean-over-72 gradient (denser, lower-variance) with NO whole-model
bigger-batch confound. If the LUT was gradient-starved, val_bpb should move from the 1x
end-to-end-LUT ~1.2286 toward the 1.5x-batch dense-parity reference 1.196862 (exp_n_0046).

LUT params := every parameter under a `.ffn.` (the CompressionMultiHeadLUT): the
compression projection (compress.{weight,bias}), the table weights (lut_batched.weights),
the learnable temperatures (log_soft_score_temp, log_select_temp), and the decompression
projection (decompress.{weight,bias}). Anchors/powers/bit_matrix/offset are buffers, not
params, so they are correctly excluded. No shared-src edits — routing is done purely by
manipulating .grad and toggling requires_grad on the existing model.

Env:
  SMOKE=1       -> build model, print param split, exit (import/build check).
  GRADCHECK=1   -> run the A/B routing on a few steps and ASSERT the split is exact
                   against independently-recomputed A-only and B-only gradients, exit.
  SMOKE_STEPS=N -> short N-step training sanity run (eval at the end).
"""
import sys, os, json, math, time, csv
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

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS, N_STEPS = cfg['device_batch_size'], cfg['n_steps']
BATCH_A, BATCH_B = cfg['batch_a'], cfg['batch_b']
EVAL_BS = cfg.get('eval_batch_size', BATCH_A)
LR, WD, WARMUP_FRAC = cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']
assert BATCH_A + BATCH_B == DEVICE_BS, 'batch_a + batch_b must equal device_batch_size'
W_A = BATCH_A / (BATCH_A + BATCH_B)     # 2/3
W_B = BATCH_B / (BATCH_A + BATCH_B)     # 1/3

FFN_TYPE = cfg.get('ffn_type', 'compression')
GAMMA    = int(cfg.get('gamma', 0))
TIE      = bool(cfg.get('tie_unembedder', False))

LUT_IN     = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
LUT_OUT    = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
LUT_NAP    = cfg.get('lut_n_anchor_pairs')
LUT_TPH    = cfg.get('lut_tables_per_head')
LUT_HEADS  = cfg.get('lut_n_heads', 1)
LUT_JOINT  = cfg.get('lut_joint_head_compression', False)
LUT_BATCHED = bool(cfg.get('lut_batched_multi_head_input', False))
LUT_FWD    = cfg.get('lut_forward_mode', 'hard')
LUT_BF16   = cfg.get('lut_use_bf16', False)
LUT_NOISE  = cfg.get('lut_init_weights_noise', 1e-3)
LUT_LEARNTEMP = bool(cfg.get('lut_learnable_temps', False))
LUT_SEED   = cfg.get('lut_base_seed', 1000)

BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f'Vocab size: {VOCAB_SIZE}')
assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
# ONE loader at 72 seq/step; split into A(48)+B(24) per step -> 72 distinct seqs, no overlap.
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, EVAL_BS, SEQ_LEN, split='val', device=DEVICE)
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
                batched_multi_head_input=LUT_BATCHED,
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
                    nn.init.zeros_(block.ffn.decompress.weight)
                if block.lin is not None:
                    nn.init.zeros_(block.lin.weight)
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
    print(f"AdamW | decay(2-D)={sum(p.numel() for p in decay):,} wd={weight_decay} | "
          f"nodecay(LUT tables+temps+1-D)={sum(p.numel() for p in nodecay):,} wd=0 | lr={lr} "
          f"[LUT tables={n_lut:,}]")
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

# --- LUT vs non-LUT parameter split (routing target) ---
lut_params    = [p for n, p in model.named_parameters() if '.ffn.' in n and p.requires_grad]
nonlut_params = [p for n, p in model.named_parameters() if '.ffn.' not in n and p.requires_grad]
lut_names     = [n for n, p in model.named_parameters() if '.ffn.' in n and p.requires_grad]
n_lut_p    = sum(p.numel() for p in lut_params)
n_nonlut_p = sum(p.numel() for p in nonlut_params)
print(f'LUT params (denser 72-seq grad): {len(lut_params)} tensors, {n_lut_p:,} elems')
print(f'non-LUT params (baseline 48-seq grad): {len(nonlut_params)} tensors, {n_nonlut_p:,} elems')
print(f'W_A={W_A:.6f} (A={BATCH_A})  W_B={W_B:.6f} (B={BATCH_B})  | LUT.grad = W_A*gA + W_B*gB')
print('LUT param names (first block shown):')
for n in [x for x in lut_names if 'blocks.0.' in x]:
    print('   ', n)


def asym_step(xA, yA, xB, yB, optimizer):
    """One asymmetric-accumulation step. Leaves final .grad on every param:
    non-LUT.grad = gA ; LUT.grad = W_A*gA + W_B*gB. Returns (lossA, lossB) floats."""
    optimizer.zero_grad(set_to_none=True)
    # Pass A: full model, mean over A.
    lossA = model(xA, yA)
    lossA.backward()                                  # every param .grad = gA
    for p in lut_params:                              # LUT -> W_A * gA
        p.grad.mul_(W_A)
    # Pass B: only LUT accumulates. Freeze non-LUT so their .grad stays = gA.
    for p in nonlut_params:
        p.requires_grad_(False)
    lossB = model(xB, yB)
    (lossB * W_B).backward()                          # LUT += W_B * gB ; non-LUT untouched
    for p in nonlut_params:
        p.requires_grad_(True)
    return lossA.item(), lossB.item()


# ---------------- build/gradcheck/smoke gates ----------------
if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)

if os.environ.get('GRADCHECK'):
    print('\n=== GRADCHECK: verifying A/B gradient routing is exact ===')
    optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
    lut_set = {id(p) for p in lut_params}
    all_named = list(model.named_parameters())
    torch.manual_seed(123)
    for t in range(3):
        x, y = next(train_loader)
        xA, yA = x[:BATCH_A], y[:BATCH_A]
        xB, yB = x[BATCH_A:], y[BATCH_A:]
        assert x.shape[0] == DEVICE_BS, x.shape
        # Reference gA: clean backward on A alone.
        optimizer.zero_grad(set_to_none=True)
        model(xA, yA).backward()
        gA = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in all_named}
        # Reference gB: clean backward on B alone.
        optimizer.zero_grad(set_to_none=True)
        model(xB, yB).backward()
        gB = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in all_named}
        # Production routine.
        asym_step(xA, yA, xB, yB, optimizer)
        # Compare.
        max_nonlut, max_lut = 0.0, 0.0
        n_checked_nl, n_checked_l = 0, 0
        for n, p in all_named:
            if p.grad is None:
                continue
            if id(p) in lut_set:
                expect = W_A * gA[n] + W_B * gB[n]
                d = (p.grad - expect).abs().max().item()
                max_lut = max(max_lut, d); n_checked_l += 1
            else:
                expect = gA[n]                        # non-LUT must equal A-only grad
                d = (p.grad - expect).abs().max().item()
                max_nonlut = max(max_nonlut, d); n_checked_nl += 1
        # Also confirm non-LUT grad is NOT accidentally contaminated by B:
        # i.e. non-LUT expect==gA already checked; assert gB differs from gA somewhere.
        difftag = max((gB[n] - gA[n]).abs().max().item() for n, p in all_named
                      if p.grad is not None and id(p) not in lut_set)
        print(f'[step {t}] non-LUT max|grad-gA|={max_nonlut:.3e} ({n_checked_nl} tensors) | '
              f'LUT max|grad-(W_A gA+W_B gB)|={max_lut:.3e} ({n_checked_l} tensors) | '
              f'nonLUT gB-vs-gA spread={difftag:.3e}')
        assert max_nonlut < 1e-4, f'non-LUT grad routing WRONG: {max_nonlut}'
        assert max_lut < 1e-4, f'LUT grad routing WRONG: {max_lut}'
        assert difftag > 1e-8, 'sanity: A and B gradients identical?? loader not advancing'
    print('GRADCHECK PASSED: non-LUT grad = A-only; LUT grad = (2/3)gA + (1/3)gB, exactly.')
    sys.exit(0)

if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS'])
    EVAL_EVERY = max(1, N_STEPS)
    print(f'*** SMOKE_STEPS={N_STEPS}: short sanity run ***')

# ---------------- full training ----------------
optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
print(f'Per step: A={BATCH_A} seq (full-model grad) + B={BATCH_B} seq (LUT-only grad) | '
      f'{N_STEPS} steps | non-LUT sees {BATCH_A*SEQ_LEN:,} tok/step, LUT sees {DEVICE_BS*SEQ_LEN:,} tok/step')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_loss', 'val_bpb'])
train_losses_logged, val_bpbs, val_steps = [], [], []
ema, best_bpb, t0 = None, float('inf'), time.time()
max_gradnorm = 0.0

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, N_STEPS, WARMUP_FRAC)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale
    x, y = next(train_loader)
    xA, yA = x[:BATCH_A], y[:BATCH_A]
    xB, yB = x[BATCH_A:], y[BATCH_A:]
    lossA, lossB = asym_step(xA, yA, xB, yB, optimizer)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    max_gradnorm = max(max_gradnorm, float(gn))
    optimizer.step()
    step_loss = W_A * lossA + W_B * lossB
    ema = step_loss if ema is None else 0.99 * ema + 0.01 * step_loss
    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lossA={lossA:.4f} lossB={lossB:.4f} | lr={lr_scale * LR:.2e} | gnorm={gn:.2f}')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}']); csv_f.flush()
        model.train()

csv_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)'); ax1.set(xlabel='step', ylabel='ce loss', title='Training Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb')
ax2.axhline(1.2285517, ls='--', color='tab:red', label='1x LUT (0052) 1.2286')
ax2.axhline(1.196862, ls='--', color='tab:green', label='1.5x batch (0046) 1.1969')
ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'ref_1x_lut_bpb': 1.2285517, 'ref_1p5x_batch_bpb': 1.196862,
           'max_gradnorm': max_gradnorm,
           'total_params': total_params, 'training_time_hours': round(elapsed / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
