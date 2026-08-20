"""exp_n_0051 -- reconstruction-auxiliary CompressionMHL FFN slot (whole-module mirror).

Same standard 16k rung as the free-row control exp_g_0006 (H8/d48/tph64/nap6,
hard, device_bs 48, 16000 steps) so val bpb is directly comparable. Adds a
TRAINING-ONLY auxiliary: for each block's forward CompressionMHL FFN slot
(input [N,384] -> output [N,384]) we build a MIRROR CompressionMultiHeadLUT of
IDENTICAL structure/capacity (same H8/d48/tph64/nap6 hard, its own compress
Linear, its own per-head luts, its own decompress Linear, its own seed) that maps
the forward module's OUTPUT [N,384] back to a reconstruction of its INPUT [N,384].

    recon = mean over blocks of MSE(mirror(forward_output), forward_input.detach())
    total_loss = task_CE + lambda_recon * recon

The mirrors are extra trainable params during TRAINING ONLY -- they are NOT in
model.forward, so val/bpb is computed on the plain CompressionMHL, and they are
excluded from the saved checkpoint (the shipped model is the unchanged slot).
The recon gradient flows into forward_output (pushing the forward slot toward
near-lossless); forward_input is detached so the encoder isn't pushed to collapse.
No edits to the shared modules fast_multi_head_lut.py / compression_mhl.py -- the
mirror is just a second CompressionMultiHeadLUT instantiated here.
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
EVAL_EVERY, EVAL_STEPS = cfg['eval_every'], cfg['eval_steps']

FFN_TYPE = cfg.get('ffn_type', 'compression')
GAMMA    = int(cfg.get('gamma', 0))
TIE      = bool(cfg.get('tie_unembedder', False))

LUT_IN     = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
LUT_OUT    = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
LUT_NAP    = cfg.get('lut_n_anchor_pairs')
LUT_TPH    = cfg.get('lut_tables_per_head')
LUT_HEADS  = cfg.get('lut_n_heads', 1)
LUT_JOINT  = cfg.get('lut_joint_head_compression', False)
LUT_BATCHED = bool(cfg.get('lut_batched_multi_head_input', False))   # False -> loop path, bit-identical to exp_g_0006 (per-head temps)
LUT_FWD    = cfg.get('lut_forward_mode', 'hard')
LUT_BF16   = cfg.get('lut_use_bf16', False)
LUT_NOISE  = cfg.get('lut_init_weights_noise', 1e-3)
LUT_LEARNTEMP = bool(cfg.get('lut_learnable_temps', False))
LUT_SEED   = cfg.get('lut_base_seed', 1000)

# --- reconstruction auxiliary knobs ---
LAMBDA_RECON = float(cfg.get('lambda_recon', 0.1))
MIRROR_SEED  = int(cfg.get('mirror_base_seed', 5000))


def make_compression_ffn(random_seed):
    """The FFN-slot CompressionMHL. Used for the forward slot AND the mirror
    (identical structure/capacity; the mirror only differs in seed)."""
    return CompressionMultiHeadLUT(
        input_dim=N_EMBD, output_dim=N_EMBD,
        inner_in_dim=LUT_IN, inner_out_dim=LUT_OUT,
        nap=LUT_NAP, tph=LUT_TPH, n_heads=LUT_HEADS,
        joint_head_compression=LUT_JOINT,
        batched_multi_head_input=LUT_BATCHED,
        forward_mode=LUT_FWD,
        use_bf16=LUT_BF16, initial_weights_noise=LUT_NOISE,
        learnable_temps=LUT_LEARNTEMP, random_seed=random_seed)


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
            self.ffn = make_compression_ffn(LUT_SEED + layer_idx)

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
        # NOTE: mirrors are NOT referenced here -> val/bpb runs on the plain model.
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope.cos, self.rope.sin)
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


# ------------------------------------------------------------------
# Reconstruction auxiliary: one whole-module mirror CompressionMHL per block.
# Hook each forward CompressionMHL to capture its (input, output), then have the
# mirror map output -> input. Training-only; never in the CE/bpb forward path.
# ------------------------------------------------------------------
class ReconAux(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._store = {}          # id(forward_ffn) -> (fwd_input[N,384], fwd_output[N,384])
        self._enabled = False
        self._forward_ffns = []   # ordered by block
        for block in model.blocks:
            assert isinstance(block.ffn, CompressionMultiHeadLUT), "recon expects CompressionMHL FFN slot"
            block.ffn.register_forward_hook(self._make_hook())
            self._forward_ffns.append(block.ffn)
        self.mirrors = nn.ModuleList([
            make_compression_ffn(MIRROR_SEED + b) for b in range(len(self._forward_ffns))
        ])

    def _make_hook(self):
        def hook(module, inp, out):
            if self._enabled:
                self._store[id(module)] = (inp[0], out)
        return hook

    def set_enabled(self, on):
        self._enabled = bool(on)

    def compute(self):
        """MSE(mirror(fwd_output), fwd_input.detach()) averaged over blocks."""
        terms = []
        for b, ffn in enumerate(self._forward_ffns):
            fwd_in, fwd_out = self._store[id(ffn)]
            pred = self.mirrors[b](fwd_out)                 # [N,384]; grad flows into fwd_out
            terms.append(F.mse_loss(pred, fwd_in.detach()))
        self._store.clear()
        return torch.stack(terms).mean()


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
          f"eps=1e-8 [LUT tables (fwd+mirror)={n_lut:,} in nodecay]")
    return opt


def get_lr_scale(step, n_steps, warmup_frac):
    w = int(warmup_frac * n_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


model = MinimalGPT(vocab_size=VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=DEPTH, seq_len=SEQ_LEN).to(DEVICE)
# Attach the training-only reconstruction auxiliary (mirrors live under model.recon
# so .parameters()/optimizer include them, but model.forward never calls them).
model.recon = ReconAux(model).to(DEVICE)

forward_trainable = sum(p.numel() for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith('recon.'))
mirror_trainable = sum(p.numel() for p in model.recon.mirrors.parameters() if p.requires_grad)
total_trainable = forward_trainable + mirror_trainable
total_params = sum(p.numel() for p in model.parameters())
print(f'MinimalGPT: depth={DEPTH}, dim={N_EMBD}, heads={N_HEAD}, seq_len={SEQ_LEN} | ffn={FFN_TYPE} tie={TIE} gamma={GAMMA}')
print(f'PARAM COUNTS | forward trainable={forward_trainable:,} | mirror aux trainable={mirror_trainable:,} | '
      f'total trainable={total_trainable:,} (total incl. any frozen={total_params:,})')
print(f'recon: lambda={LAMBDA_RECON} mirror_seed={MIRROR_SEED} n_mirrors={len(model.recon._forward_ffns)} '
      f'(one whole-module CompressionMHL mirror per block)')

if os.environ.get('SMOKE'):
    print('SMOKE OK'); sys.exit(0)
if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS'])
    EVAL_EVERY = max(1, N_STEPS)          # single eval at the end of the smoke run
    print(f'*** SMOKE_STEPS={N_STEPS}: short run, eval_every={EVAL_EVERY} ***')

optimizer = setup_optimizer(model, lr=LR, weight_decay=WD)
tokens_per_step = DEVICE_BS * SEQ_LEN
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step', 'train_ce', 'train_recon', 'val_bpb'])
ce_logged, recon_logged, val_bpbs, val_steps = [], [], [], []
ema_ce, ema_recon, best_bpb, t0 = None, None, float('inf'), time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step, N_STEPS, WARMUP_FRAC)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale
    optimizer.zero_grad(set_to_none=True)
    accum_ce, accum_recon = 0.0, 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        model.recon.set_enabled(True)
        loss_ce = model(x, y)                     # hooks capture per-block (in, out)
        recon = model.recon.compute()
        model.recon.set_enabled(False)
        loss = loss_ce + LAMBDA_RECON * recon
        (loss / grad_accum).backward()
        accum_ce += loss_ce.item() / grad_accum
        accum_recon += recon.item() / grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    ema_ce = accum_ce if ema_ce is None else 0.99 * ema_ce + 0.01 * accum_ce
    ema_recon = accum_recon if ema_recon is None else 0.99 * ema_recon + 0.01 * accum_recon
    if step % 100 == 0 or step == 1 or os.environ.get('SMOKE_STEPS'):
        print(f'step {step:6d} | ce={ema_ce:.4f} | recon={ema_recon:.5f} '
              f'(raw {accum_recon:.5f}) | lr={lr_scale * LR:.2e}')
    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        best_bpb = min(best_bpb, bpb)
        print(f'[VAL] step {step}: bpb={bpb:.4f} | ce_ema={ema_ce:.4f} | recon_ema={ema_recon:.5f}')
        ce_logged.append(ema_ce); recon_logged.append(ema_recon)
        val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema_ce:.6f}', f'{ema_recon:.6f}', f'{bpb:.6f}']); csv_f.flush()
        model.train()

csv_f.close()
elapsed = time.time() - t0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, ce_logged, label='train CE (ema)')
ax1b = ax1.twinx(); ax1b.plot(val_steps, recon_logged, color='tab:green', label='recon (ema)')
ax1.set(xlabel='step', ylabel='ce loss', title='Train CE + recon'); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb'); ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'best_val_bpb': best_bpb,
           'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
           'final_train_ce': ce_logged[-1] if ce_logged else None,
           'final_train_recon': recon_logged[-1] if recon_logged else None,
           'lambda_recon': LAMBDA_RECON,
           'forward_trainable_params': forward_trainable,
           'mirror_aux_trainable_params': mirror_trainable,
           'total_trainable_params': total_trainable,
           'training_time_hours': round(elapsed / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
# Ship the PLAIN model only: drop the training-only mirrors from the checkpoint.
shipped = {k: v for k, v in model.state_dict().items() if not k.startswith('recon.')}
torch.save(shipped, os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
