"""
exp147_int8_qat — Int8 quantization-aware training.

Weight constraint: all LUT projection.weights always in {k/256000 : k=0..255}.
  - Range: [0, 255/256000] ~= [0, 0.000996]
  - Before each forward: snap weights to grid via .data assignment (STE trick).
  - After each optimizer step: clamp latent weights to [0, 255/256000].

Gradient constraint: |dw| = G/256000, G >= 0 integer. No normalization.
  - Quantize raw gradient magnitude to nearest 1/256000 step.

Initialization: uniform [0, 255/256000].
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, evaluate_model,
    count_params, compute_virtual_bandwidth, save_summary,
    BOS_ID, CONTEXT_SIZE,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import RankAttention

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

W_MIN = -128 / 256000  # -0.0005
W_MAX =  127 / 256000  #  0.000496
GRID_STEP = 1 / 256000  # weight grid spacing = gradient quantization step

# ── Model (same arch as exp135) ────────────────────────────────────────────────

def make_lut(input_dim, n_heads, n_outputs, tables_per_head, seed_offset=0):
    return MultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=cfg['n_anchor_pairs'],
        tables_per_head=tables_per_head,
        smooth_mode=cfg['smooth_mode'],
        n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'],
        calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        d, p, h = cfg['embedding_dim'], cfg['positional_dim'], cfg['num_heads']
        d_qk, d_v = cfg['d_qk'], cfg['d_v']
        tph_qkv, tph_op = cfg['qkv_tables_per_head'], cfg['out_proj_tables_per_head']
        s = layer_idx * 10
        self.q_lut    = make_lut(d + p, h, d_qk, tph_qkv, s + 0)
        self.k_lut    = make_lut(d + p, h, d_qk, tph_qkv, s + 1)
        self.v_lut    = make_lut(d + p, h, d_v,  tph_qkv, s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,  tph_op,  s + 3)
        # Per-LUT learnable scale factors (float, trained normally)
        self.q_scale  = nn.Parameter(torch.tensor(0.001))
        self.k_scale  = nn.Parameter(torch.tensor(0.001))
        self.v_scale  = nn.Parameter(torch.tensor(0.001))
        self.op_scale = nn.Parameter(torch.tensor(0.001))
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0)
        self.n_heads, self.d_qk, self.d_v, self.d = h, d_qk, d_v, d

    def lut_scale_pairs(self, exclude_out_proj=False):
        pairs = [
            (self.q_lut,    self.q_scale),
            (self.k_lut,    self.k_scale),
            (self.v_lut,    self.v_scale),
        ]
        if not exclude_out_proj:
            pairs.append((self.out_proj, self.op_scale))
        return pairs

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])
        q = self.q_lut(x_pos).permute(1,0,2).reshape(self.n_heads,B,T,self.d_qk).permute(1,0,2,3) * self.q_scale
        k = self.k_lut(x_pos).permute(1,0,2).reshape(self.n_heads,B,T,self.d_qk).permute(1,0,2,3) * self.k_scale
        v = self.v_lut(x_pos).permute(1,0,2).reshape(self.n_heads,B,T,self.d_v).permute(1,0,2,3)  * self.v_scale
        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0,2,1,3).reshape(B*T, self.n_heads*self.d_v)
        return x + self.out_proj(attn)[:,0,:].reshape(B, T, E) * self.op_scale


class LUTTransformerLinearUnemb(nn.Module):
    def __init__(self, maxlen=CONTEXT_SIZE):
        super().__init__()
        d, p = cfg['embedding_dim'], cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(d, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x)


# ── QAT helpers ────────────────────────────────────────────────────────────────

def lut_scale_pairs(model):
    """All (lut, scale) pairs except the last layer's out_proj."""
    pairs = []
    for i, layer in enumerate(model.layers):
        is_last = (i == len(model.layers) - 1)
        pairs.extend(layer.lut_scale_pairs(exclude_out_proj=is_last))
    return pairs


def init_weights_int8(model):
    """Initialize LUT weights as uniform integers in [-128, 127] (stored as float)."""
    with torch.no_grad():
        for lut, scale in lut_scale_pairs(model):
            lut.projection.weights.uniform_(-128, 127)
            lut.projection.weights.data.round_()


def quantize_weights_inplace(model):
    """Snap latent weights to nearest integer in [-128, 127] (STE)."""
    with torch.no_grad():
        for lut, _ in lut_scale_pairs(model):
            w = lut.projection.weights
            w.data.copy_(w.data.round().clamp(-128, 127))


def quantize_gradients(model):
    """Ceil |dw| to nearest integer, zero only when |dw| < 0.5."""
    for lut, _ in lut_scale_pairs(model):
        g = lut.projection.weights.grad
        if g is None:
            continue
        sign = g.sign()
        mag = g.abs()
        g_q = sign * mag.ceil()
        g_q[mag < 0.5] = 0.0
        g.copy_(g_q)


def clamp_weights(model):
    """Clamp latent weights to [-128, 127] after optimizer step."""
    with torch.no_grad():
        for lut, _ in lut_scale_pairs(model):
            lut.projection.weights.data.clamp_(-128, 127)


def count_distinct_lut(model):
    """Count distinct weight values across quantized LUT modules."""
    all_w = torch.cat([lut.projection.weights.detach().flatten() for lut, _ in lut_scale_pairs(model)])
    return all_w.unique().numel()


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# ── Run ────────────────────────────────────────────────────────────────────────

sampler = make_sampler(DEVICE, random_seed=1)
model = LUTTransformerLinearUnemb().to(DEVICE)
init_weights_int8(model)

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
bw = compute_virtual_bandwidth(model)
print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB  (ratio {bw["lut_ratio"]:.4f})')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss', 'distinct_vals'])

train_losses, val_losses, steps_log = [], [], []
best_val_loss = float('inf')
best_step = 0
ema = None
alpha = 0.01
start_time = time.time()

model.train()
for step in range(cfg['n_steps']):
    # STE: forward uses quantized weights, backward updates latent weights
    quantize_weights_inplace(model)

    x = sampler.sample_training_batch(cfg['batch_size']).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    tgt = x

    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T))

    optimizer.zero_grad()
    loss.backward()
    quantize_gradients(model)
    optimizer.step()
    scheduler.step()

    clamp_weights(model)

    lv = loss.item()
    ema = lv if ema is None else (1 - alpha) * ema + alpha * lv

    if step % 100 == 0:
        lr = scheduler.get_last_lr()[0]
        warmup_steps = int(cfg.get('lr_warmup_fraction', 0.1) * cfg['n_steps'])
        phase = 'warmup' if step < warmup_steps else 'decay'
        print(f'step {step:6d} | train_loss={ema:.4f} | lr={lr:.2e} ({phase})')

    if step % cfg['test_every'] == 0:
        val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])
        n_distinct = count_distinct_lut(model)
        print(f'[VAL] step {step}: val_loss={val_loss:.4f} | distinct_weights={n_distinct:,}')
        train_losses.append(ema)
        val_losses.append(val_loss)
        steps_log.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{val_loss:.6f}', n_distinct])
        csv_f.flush()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
        model.train()

csv_f.close()
elapsed = time.time() - start_time

plt.figure(figsize=(8, 4))
plt.plot(steps_log, train_losses, label='train')
plt.plot(steps_log, val_losses, label='val')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val_loss,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'best_step': best_step,
    'total_steps': cfg['n_steps'],
    'total_params': total_params,
    'trainable_params': trainable_params,
    'training_time_hours': round(elapsed / 3600, 3),
}
save_summary(EXP_DIR, summary)
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
