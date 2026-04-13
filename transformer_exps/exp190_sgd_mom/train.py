"""
exp173_v3_softmax — V3 LUT attention scores + value LUT + softmax @ V + out_proj.

Each block:
  1. LUTAttentionV3 produces raw attention scores [B, T, T, H, 1]
  2. V-LUT (per-token) produces values [B, T, H, D_head]
  3. softmax(scores) @ V → [B, T, H, D_head]
  4. reshape → [B, T, E], apply out_proj LUT
  5. residual + LayerNorm
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
from spiky.lutorch.lut_attention import LUTAttentionV3

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
SEQ_LEN = CONTEXT_SIZE
torch.manual_seed(cfg['random_seed'])
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Model ──────────────────────────────────────────────────────────────────────

def _attn_exclusion_sets():
    if not cfg.get('use_exclusion_sets', False):
        return None
    E = cfg['embedding_dim']
    P = cfg['positional_dim']
    return [list(range(E)), list(range(E, 2 * E)), list(range(2 * E, 2 * E + P))]


def make_score_attn(seed_offset=0):
    """LUTAttentionV3: produces raw attention scores [B, T, T, H, 1]."""
    E = cfg['embedding_dim']
    P = cfg['positional_dim']
    H = cfg['n_heads']
    lut = MultiHeadLut(
        input_dim=2 * E + P,
        n_heads=H,
        n_outputs=1,
        n_anchor_pairs=cfg['attention_nap'],
        tables_per_head=cfg['attention_tph'],
        smooth_mode=cfg['smooth_mode'],
        n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'],
        calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        recompute_in_backward=cfg.get('recompute_in_backward', False),
        exclusion_sets=_attn_exclusion_sets(),
    )
    return LUTAttentionV3(
        multi_head_lut=lut,
        seq_len=SEQ_LEN,
        causal=cfg['causal'],
        include_diagonal=cfg['include_diagonal'],
    )


def make_value_lut(seed_offset=200):
    """Per-token value projection: [B*T, E] → [B*T, H, d_v]."""
    E = cfg['embedding_dim']
    H = cfg['n_heads']
    d_v = cfg['d_v']
    return MultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=d_v,
        n_anchor_pairs=cfg['value_nap'],
        tables_per_head=cfg['value_tph'],
        smooth_mode=cfg['smooth_mode'],
        n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'],
        calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        recompute_in_backward=cfg.get('recompute_in_backward', False),
    )


def make_out_proj(seed_offset=400):
    """Pointwise 1-headed LUT: input_dim=H*d_v → n_outputs=E."""
    E = cfg['embedding_dim']
    H = cfg['n_heads']
    d_v = cfg['d_v']
    return MultiHeadLut(
        input_dim=H * d_v,
        n_heads=1,
        n_outputs=E,
        n_anchor_pairs=cfg['out_proj_nap'],
        tables_per_head=cfg['out_proj_tph'],
        smooth_mode=cfg['smooth_mode'],
        n_alternatives=cfg['n_alternatives'],
        normalize_weights=cfg['normalise_weights'],
        calibrate_output=cfg['calibrate_output'],
        anchor_sampling_policy=AnchorSamplingPolicy(cfg['anchor_sampling_policy']),
        initial_weights_noise=cfg['initial_weights_noise'],
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        random_seed=cfg['random_seed'] + seed_offset,
        device=DEVICE,
        recompute_in_backward=cfg.get('recompute_in_backward', False),
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        E = cfg['embedding_dim']
        H = cfg['n_heads']
        d_v = cfg['d_v']
        self.score_attn = make_score_attn(seed_offset=layer_idx)
        self.value_lut = make_value_lut(seed_offset=200 + layer_idx)
        self.attn_norm = nn.LayerNorm(H * d_v)
        self.out_proj = make_out_proj(seed_offset=400 + layer_idx)
        self.norm = nn.LayerNorm(E)
        self.H = H
        self.d_v = d_v

    def forward(self, x, rel_pe):
        B, T, E = x.shape
        H, d_v = self.H, self.d_v

        raw_scores = self.score_attn(x, rel_pe).squeeze(-1).permute(0, 3, 1, 2)
        attn_weights = F.softmax(raw_scores, dim=-1)

        v = self.value_lut(x.reshape(B * T, E))
        v = v.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn_out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, T, H * d_v)
        attn_out = self.attn_norm(attn_out)
        proj_out = self.out_proj(attn_out.reshape(B * T, H * d_v)).squeeze(1).reshape(B, T, E)

        return x + self.norm(proj_out)


class LUTTransformerV3Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        E = cfg['embedding_dim']
        P = cfg['positional_dim']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.rel_pe = nn.Parameter(torch.randn(SEQ_LEN, P) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(cfg['num_layers'])])
        self.unembedder = nn.Linear(E, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        for layer in self.layers:
            x = layer(x, self.rel_pe)
        return self.unembedder(x)


# ── Quantization regularization ────────────────────────────────────────────────

def quant_reg_loss(model, D):
    total = torch.tensor(0.0, device=DEVICE)
    count = 0
    for m in model.modules():
        if isinstance(m, MultiHeadLut):
            w = m.projection.weights
            total = total + torch.sin(math.pi * w * D).pow(2).mean()
            count += 1
    return total / max(count, 1)


def get_lambda(step, n_steps, lambda_max, start_fraction):
    start = int(start_fraction * n_steps)
    if step < start:
        return 0.0
    return lambda_max * (step - start) / max(n_steps - start, 1)


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
model = LUTTransformerV3Softmax().to(DEVICE)

if cfg.get('optimizer') == 'SGD_momentum':
    optimizer = torch.optim.SGD([
        {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
    ], momentum=0.9)
else:
    optimizer = torch.optim.Adam([
        {'params': model.unembedder.parameters(), 'lr': cfg['lr_unembedder']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('unembedder')], 'lr': cfg['lr']},
    ])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

total_params, trainable_params = count_params(model)
bw = compute_virtual_bandwidth(model)
print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB  (ratio {bw["lut_ratio"]:.4f})')
print(f'Sequence length: {SEQ_LEN}, pairs per block: {model.layers[0].score_attn.pair_rows.shape[0]}')
print(f'Architecture: V3 scores(H={cfg["n_heads"]},tph={cfg["attention_tph"]}) + V-LUT + softmax@V + out_proj(tph={cfg["out_proj_tph"]})')

D = cfg['quant_reg_D']
lambda_max = cfg['quant_reg_lambda_max']
start_fraction = cfg['quant_reg_start_fraction']

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'qreg_loss', 'lambda', 'val_loss', 'val_loss_snapped'])

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

        print(f'[VAL] step {step}: val_loss={val_loss:.4f}')
        train_losses.append(ema)
        val_losses.append(val_loss)
        steps_log.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{ema_qreg:.6f}', f'{lam:.6f}', f'{val_loss:.6f}', ''])
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
}
save_summary(EXP_DIR, summary)

torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('Checkpoint saved.')

print('\n=== DONE ===')
print(json.dumps(summary, indent=2))
