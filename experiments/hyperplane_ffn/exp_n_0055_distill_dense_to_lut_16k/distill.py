"""exp_n_0055 — distillation hypothesis test.

Can a CompressionMHL (LUT) trained to imitate the dense FFN's real activations
reproduce the dense function? Load the frozen 1.19 tied-dense baseline (exp073),
capture each of the 6 blocks' FFN (input h=ln2(x), output mlp(h) pre-residual),
and fit ONE CompressionMultiHeadLUT per block to regress input->output with MSE
using the STANDARD hard-forward/soft-backward (STE) path. The LUT hyperparameters
are reproduced 1:1 from exp_n_0052 (batched control). After training all 6, swap
them into the frozen dense model in place of their FFNs and evaluate whole-model
val_bpb on the clean val set; compare to dense 1.196646.

Online streaming distillation: each step runs a fresh training batch through the
frozen dense model (hooks capture all 6 blocks' in/out simultaneously) and trains
each block's LUT on MSE(lut(in), out.detach()). Streams ~n_steps*B*T token-vectors
per block (stated below) rather than caching a fixed multi-GB dataset — stronger
test (no memorization) and matches the standard-rung token budget. Shared modules
untouched.
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
HFF = os.path.dirname(EXP_DIR)
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

DEPTH, N_EMBD, N_HEAD, SEQ_LEN = cfg['depth'], cfg['n_embd'], cfg['n_head'], cfg['seq_len']
DEVICE_BS = cfg['device_batch_size']
N_STEPS, LR, WD, WARMUP_FRAC = cfg['n_steps'], cfg['lr'], cfg['weight_decay'], cfg['lr_warmup_fraction']
LOG_EVERY, EVAL_STEPS = cfg['log_every'], cfg['eval_steps']
TIE = bool(cfg['tie_unembedder'])
# LUT params (1:1 from exp_n_0052)
L = dict(inner_in=cfg['lut_inner_in_dim'], inner_out=cfg['lut_inner_out_dim'], nap=cfg['lut_n_anchor_pairs'],
         tph=cfg['lut_tables_per_head'], heads=cfg['lut_n_heads'], joint=cfg['lut_joint_head_compression'],
         batched=cfg['lut_batched_multi_head_input'], fwd=cfg['lut_forward_mode'], bf16=cfg['lut_use_bf16'],
         noise=cfg['lut_init_weights_noise'], seed=cfg['lut_base_seed'], learn_t=cfg['lut_learnable_temps'])

if os.environ.get('SMOKE_STEPS'):
    N_STEPS = int(os.environ['SMOKE_STEPS'])
    LOG_EVERY = max(1, N_STEPS // 5)
    print(f'*** SMOKE_STEPS={N_STEPS} ***')

BASE_DIR = get_base_dir(); TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size(); assert VOCAB_SIZE == cfg['tokenizer_vocab_size']
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)


# ---------------- model (dense; matches exp073 / the shared exp043+ trainer) ----------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1); return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)

class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__(); self.n_head = n_head
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
        self.ln1 = nn.LayerNorm(n_embd); self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd, bias=False))
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        return x + self.mlp(h)          # self.mlp is swapped to a LUT adapter after distillation

class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, max_seq_len=seq_len)
        self.blocks = nn.ModuleList([MinimalBlock(n_embd, n_head, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
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
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction=loss_reduction, ignore_index=-1)
        return logits


class LUTAdapter(nn.Module):
    """Wrap a trained CompressionMHL so it drops into the dense block's `self.mlp`
    slot: [B,T,C] -> reshape -> CMHL -> [B,T,C]."""
    def __init__(self, cmhl):
        super().__init__(); self.cmhl = cmhl
    def forward(self, h):
        B, T, C = h.shape
        return self.cmhl(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)


def make_cmhl(block_idx):
    return CompressionMultiHeadLUT(
        input_dim=N_EMBD, output_dim=N_EMBD, inner_in_dim=L['inner_in'], inner_out_dim=L['inner_out'],
        nap=L['nap'], tph=L['tph'], n_heads=L['heads'], joint_head_compression=L['joint'],
        batched_multi_head_input=L['batched'], forward_mode=L['fwd'], use_bf16=L['bf16'],
        initial_weights_noise=L['noise'], learnable_temps=L['learn_t'], random_seed=L['seed'] + block_idx).to(DEVICE)


# ---------------- build frozen dense model + load checkpoint ----------------
model = MinimalGPT(VOCAB_SIZE, N_EMBD, N_HEAD, DEPTH, SEQ_LEN).to(DEVICE)
sd = torch.load(os.path.join(HFF, cfg['dense_ckpt']), map_location=DEVICE)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'loaded dense ckpt: missing={list(missing)} unexpected={list(unexpected)}')
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

# sanity: the loaded frozen dense model should reproduce ~1.196646
dense_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
print(f'[CHECK] loaded dense val_bpb = {dense_bpb:.7f} (target {cfg["dense_val_bpb"]:.7f}, delta {dense_bpb - cfg["dense_val_bpb"]:+.2e})')

# ---------------- hooks: capture per-block FFN (input, output) ----------------
caps = [None] * DEPTH
def make_hook(b):
    def hook(mod, inp, out):
        caps[b] = (inp[0].reshape(-1, N_EMBD).detach(), out.reshape(-1, N_EMBD).detach())
    return hook
for b, blk in enumerate(model.blocks):
    blk.mlp.register_forward_hook(make_hook(b))

# ---------------- build 6 LUTs + optimizer (0033 grouping) ----------------
luts = [make_cmhl(b) for b in range(DEPTH)]
lut_ids = {id(p) for lu in luts for m in lu.modules() if isinstance(m, FastMultiHeadLut) for p in m.parameters(recurse=False)}
decay, nodecay = [], []
for lu in luts:
    for p in lu.parameters():
        (nodecay if (id(p) in lut_ids or p.ndim < 2) else decay).append(p)
opt = torch.optim.AdamW([
    dict(params=decay, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=WD),
    dict(params=nodecay, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])
for g in opt.param_groups:
    g['initial_lr'] = g['lr']
lut_params = sum(p.numel() for lu in luts for p in lu.parameters())
per_lut = sum(p.numel() for p in luts[0].parameters())
print(f'PARAM COUNTS | dense frozen={sum(p.numel() for p in model.parameters()):,} | '
      f'per-LUT={per_lut:,} | 6 LUTs total (trainable)={lut_params:,}')
print(f'LUT config (== exp_n_0052): H{L["heads"]}/d{L["inner_in"]}/tph{L["tph"]}/nap{L["nap"]} '
      f'joint={L["joint"]} batched={L["batched"]} fwd={L["fwd"]} learnable_temps={L["learn_t"]}')

def lr_scale(step):
    w = int(WARMUP_FRAC * N_STEPS)
    if step < w:
        return step / max(w, 1)
    prog = (step - w) / max(N_STEPS - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

# ---------------- online distillation ----------------
csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['step'] + [f'mse_b{b}' for b in range(DEPTH)])
curve = {b: ([], []) for b in range(DEPTH)}
ema = [None] * DEPTH
t0 = time.time()
for lu in luts:
    lu.train()
for step in range(1, N_STEPS + 1):
    sc = lr_scale(step)
    for g in opt.param_groups:
        g['lr'] = g['initial_lr'] * sc
    with torch.no_grad():
        x, _ = next(train_loader)
        model(x)                          # frozen forward -> hooks fill caps[b]=(in,out)
    opt.zero_grad(set_to_none=True)
    step_mse = []
    for b in range(DEPTH):
        hin, hout = caps[b]
        pred = luts[b](hin.reshape(-1, N_EMBD))
        loss = F.mse_loss(pred, hout)
        loss.backward()                   # grads only into luts[b]
        step_mse.append(loss.item())
    torch.nn.utils.clip_grad_norm_([p for lu in luts for p in lu.parameters()], 1.0)
    opt.step()
    for b in range(DEPTH):
        ema[b] = step_mse[b] if ema[b] is None else 0.99 * ema[b] + 0.01 * step_mse[b]
    if step % LOG_EVERY == 0 or step == 1 or step == N_STEPS:
        for b in range(DEPTH):
            curve[b][0].append(step); curve[b][1].append(ema[b])
        csv_w.writerow([step] + [f'{ema[b]:.6f}' for b in range(DEPTH)]); csv_f.flush()
        print(f'step {step:6d} | lr={sc*LR:.2e} | MSE(ema) ' +
              ' '.join(f'b{b}={ema[b]:.4f}' for b in range(DEPTH)))
csv_f.close()
tokens_per_block = N_STEPS * DEVICE_BS * SEQ_LEN
print(f'distillation streamed ~{tokens_per_block:,} token-vectors per block ({N_STEPS} steps x {DEVICE_BS*SEQ_LEN} tok/batch)')

# ---------------- swap all 6 LUTs into the frozen dense model + eval ----------------
for b, blk in enumerate(model.blocks):
    luts[b].eval()
    blk.mlp = LUTAdapter(luts[b])
model.eval()
swap_bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
final_mse = {f'b{b}': round(ema[b], 6) for b in range(DEPTH)}
print(f'\n=== SWAP-IN whole-model val_bpb = {swap_bpb:.7f} '
      f'(dense {cfg["dense_val_bpb"]:.7f}, delta {swap_bpb - cfg["dense_val_bpb"]:+.5f}) ===')
print('final per-block distill MSE:', final_mse)

# ---------------- plot: 6 learning curves ----------------
plt.figure(figsize=(9, 6))
for b in range(DEPTH):
    plt.plot(curve[b][0], curve[b][1], label=f'block {b} (final MSE {ema[b]:.4f})')
plt.xlabel('distillation step'); plt.ylabel('MSE (ema)  in->out regression')
plt.yscale('log'); plt.grid(True, alpha=0.3, which='both'); plt.legend(fontsize=9)
plt.title(f'exp_n_0055 distill dense FFN -> CompressionMHL (per block)\nswap-in val_bpb {swap_bpb:.4f} vs dense {cfg["dense_val_bpb"]:.4f}')
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'learning_curves.png'), dpi=120); plt.close()

summary = {'exp_name': cfg['exp_name'], 'dense_val_bpb': cfg['dense_val_bpb'],
           'loaded_dense_val_bpb': dense_bpb, 'swapin_val_bpb': swap_bpb,
           'delta_vs_dense': swap_bpb - cfg['dense_val_bpb'], 'final_block_mse': final_mse,
           'lut_params_per_block': per_lut, 'lut_params_total': lut_params,
           'tokens_streamed_per_block': tokens_per_block, 'n_steps': N_STEPS,
           'lut_config': 'exp_n_0052 (H8/d48/tph64/nap6 hard batched learnable_temps)',
           'training_time_hours': round((time.time() - t0) / 3600, 3)}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n=== DONE ==='); print(json.dumps(summary, indent=2))
