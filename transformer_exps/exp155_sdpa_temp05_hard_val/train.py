"""
exp155_sdpa_temp05_hard_val — fork of exp154.

Fixed sdpa_temperature=0.5 (no annealing).
LayerNorm only after out_proj. D=256, 50k steps, bs=32.
Adds hard validation: also evaluates with sdpa_temperature=0.01.
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
    make_sampler, evaluate_model, MetricsLogger,
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

# ── Model ─────────────────────────────────────────────────────────────────────

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
        d       = cfg['embedding_dim']
        p       = cfg['positional_dim']
        h       = cfg['num_heads']
        d_qk    = cfg['d_qk']
        d_v     = cfg['d_v']
        tph_qkv = cfg['qkv_tables_per_head']
        tph_op  = cfg['out_proj_tables_per_head']
        s = layer_idx * 10
        self.q_lut    = make_lut(d + p, h, d_qk, tph_qkv, s + 0)
        self.k_lut    = make_lut(d + p, h, d_qk, tph_qkv, s + 1)
        self.v_lut    = make_lut(d + p, h, d_v,  tph_qkv, s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,  tph_op,  s + 3)
        self.out_norm = nn.LayerNorm(d)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0, sdpa_temperature=0.5)
        self.n_heads = h
        self.d_qk = d_qk
        self.d_v = d_v
        self.d = d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos = torch.cat([x, pos], dim=-1).reshape(-1, E + pos.shape[-1])
        q = self.q_lut(x_pos).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = self.k_lut(x_pos).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = self.v_lut(x_pos).permute(1, 0, 2).reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)
        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, self.n_heads * self.d_v)
        out = self.out_proj(attn)[:, 0, :].reshape(B, T, E)
        return x + self.out_norm(out)


class LUTTransformerLinearUnemb(nn.Module):
    def __init__(self, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        p = cfg['positional_dim']
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


# ── Quantization regularization ────────────────────────────────────────────────

def quant_reg_loss(model, D):
    """L_quant = mean(sin^2(pi * w * D)) over all LUT weights."""
    total = torch.tensor(0.0, device=DEVICE)
    count = 0
    for m in model.modules():
        if isinstance(m, MultiHeadLut):
            w = m.projection.weights
            total = total + torch.sin(math.pi * w * D).pow(2).mean()
            count += 1
    return total / max(count, 1)


def get_lambda(step, n_steps, lambda_max, start_fraction):
    """Linearly anneal lambda from 0 to lambda_max after start_fraction of training."""
    start = int(start_fraction * n_steps)
    if step < start:
        return 0.0
    return lambda_max * (step - start) / max(n_steps - start, 1)


def snap_to_grid(model, D):
    """Snap all LUT weights to nearest k/D grid point (in-place)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiHeadLut):
                w = m.projection.weights
                w.copy_((w * D).round() / D)


def save_original_weights(model):
    return {id(m): m.projection.weights.clone()
            for m in model.modules() if isinstance(m, MultiHeadLut)}


def restore_weights(model, saved):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiHeadLut) and id(m) in saved:
                m.projection.weights.copy_(saved[id(m)])


def set_sdpa_temperature(model, temp):
    for m in model.modules():
        if isinstance(m, RankAttention):
            m.sdpa_temperature = temp


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

optimizer = torch.optim.Adam([
    {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
    {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
bw = compute_virtual_bandwidth(model)
print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB  (ratio {bw["lut_ratio"]:.4f})')

D = cfg['quant_reg_D']
lambda_max = cfg['quant_reg_lambda_max']
start_fraction = cfg['quant_reg_start_fraction']

# CSV logger
csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'qreg_loss', 'lambda', 'val_loss', 'val_loss_snapped', 'val_loss_hard'])

train_losses, val_losses, steps_log = [], [], []
best_val_loss = float('inf')
best_step = 0
ema = None
alpha = 0.01
ema_qreg = None
start_time = time.time()

model.train()
for step in range(cfg['n_steps']):
    x = sampler.sample_training_batch(cfg['batch_size']).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    tgt = x

    logits = model(inp)
    B, T, V = logits.shape
    ce_loss = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T))

    lam = get_lambda(step, cfg['n_steps'], lambda_max, start_fraction)
    qreg = quant_reg_loss(model, D)
    loss = ce_loss + lam * qreg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    lv = ce_loss.item()
    ema = lv if ema is None else (1 - alpha) * ema + alpha * lv
    qv = qreg.item()
    ema_qreg = qv if ema_qreg is None else (1 - alpha) * ema_qreg + alpha * qv

    if step % 100 == 0:
        lr = scheduler.get_last_lr()[0]
        warmup_steps = int(cfg.get('lr_warmup_fraction', 0.1) * cfg['n_steps'])
        phase = 'warmup' if step < warmup_steps else 'decay'
        print(f'step {step:6d} | train_loss={ema:.4f} | qreg={ema_qreg:.4f} | lambda={lam:.5f} | lr={lr:.2e} ({phase})')

    if step % cfg['test_every'] == 0:
        val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])

        orig = save_original_weights(model)
        snap_to_grid(model, D)
        val_loss_snapped = evaluate_model(model, sampler, cfg['test_batch_size'])
        restore_weights(model, orig)

        # Hard validation: evaluate with sdpa_temperature=0.01
        set_sdpa_temperature(model, 0.01)
        val_loss_hard = evaluate_model(model, sampler, cfg['test_batch_size'])
        set_sdpa_temperature(model, 0.5)

        model.train()

        print(f'[VAL] step {step}: val_loss={val_loss:.4f} | snapped={val_loss_snapped:.4f} | hard={val_loss_hard:.4f}')
        train_losses.append(ema)
        val_losses.append(val_loss)
        steps_log.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{ema_qreg:.6f}', f'{lam:.6f}', f'{val_loss:.6f}', f'{val_loss_snapped:.6f}', f'{val_loss_hard:.6f}'])
        csv_f.flush()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step

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
    'quant_reg_D': D,
    'quant_reg_lambda_max': lambda_max,
}
save_summary(EXP_DIR, summary)

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('Checkpoint saved.')

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
