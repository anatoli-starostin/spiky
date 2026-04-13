"""
exp144_batch_schedule_100k — Fork of exp135.

Batch size + LR schedule over 100K steps:
  Steps   0– 5K: bs=16,  LR warmup 0→1e-4 (first 1K), then hold 1e-4
  Steps  5K–15K: bs=32,  LR = 1.4e-4 (constant)
  Steps 15K–40K: bs=64,  LR = 2e-4   (constant)
  Steps 40K–100K: bs=128, LR cosine decay 2e-4→1e-6
"""
import sys, os, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, count_params, compute_virtual_bandwidth,
    evaluate_model, save_loss_plot, save_summary,
    MetricsLogger,
    CONTEXT_SIZE, BOS_ID,
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

# ── Schedule ───────────────────────────────────────────────────────────────────

def get_batch_size(step):
    if step < 5_000:  return 16
    if step < 15_000: return 32
    if step < 40_000: return 64
    return 128

def get_lr(step):
    if step < 1_000:
        return 1e-4 * (step / 1_000)
    if step < 5_000:
        return 1e-4
    if step < 15_000:
        return 1.4e-4
    if step < 40_000:
        return 2e-4
    # cosine decay 2e-4 → 1e-6 over steps 40K–100K
    progress = (step - 40_000) / (100_000 - 40_000)
    return 1e-6 + 0.5 * (2e-4 - 1e-6) * (1 + math.cos(math.pi * progress))

# ── Model ──────────────────────────────────────────────────────────────────────

def make_lut(input_dim, n_heads, n_outputs, tables_per_head, cfg, seed_offset=0):
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
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d       = cfg['embedding_dim']
        p       = cfg['positional_dim']
        h       = cfg['num_heads']
        d_qk    = cfg['d_qk']
        d_v     = cfg['d_v']
        tph_qkv = cfg['qkv_tables_per_head']
        tph_op  = cfg['out_proj_tables_per_head']
        lut_input_dim = d + p
        s = layer_idx * 10

        self.q_lut    = make_lut(lut_input_dim, h, d_qk, tph_qkv, cfg, s + 0)
        self.k_lut    = make_lut(lut_input_dim, h, d_qk, tph_qkv, cfg, s + 1)
        self.v_lut    = make_lut(lut_input_dim, h, d_v,  tph_qkv, cfg, s + 2)
        self.out_proj = make_lut(h * d_v, 1, d,          tph_op,  cfg, s + 3)
        self.rank_attn = RankAttention(d_qk, d_v, smooth_mode=False, temperature=1.0)
        self.n_heads = h
        self.d_qk = d_qk
        self.d_v = d_v
        self.d = d

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos = torch.cat([x, pos], dim=-1)
        x_pos_flat = x_pos.reshape(-1, E + pos.shape[-1])

        q = self.q_lut(x_pos_flat).permute(1, 0, 2)
        k = self.k_lut(x_pos_flat).permute(1, 0, 2)
        v = self.v_lut(x_pos_flat).permute(1, 0, 2)

        q = q.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        k = k.reshape(self.n_heads, B, T, self.d_qk).permute(1, 0, 2, 3)
        v = v.reshape(self.n_heads, B, T, self.d_v).permute(1, 0, 2, 3)

        attn = self.rank_attn(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, self.n_heads * self.d_v)

        return x + self.out_proj(attn)[:, 0, :].reshape(B, T, E)


class LUTTransformerLinearUnemb(nn.Module):

    def __init__(self, cfg, maxlen=CONTEXT_SIZE):
        super().__init__()
        d = cfg['embedding_dim']
        p = cfg['positional_dim']
        n = cfg['num_layers']
        self.token_embedder = nn.Embedding(cfg['vocab_size'], d)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, maxlen, p) * 0.1)
        self.layers = nn.ModuleList([LUTBlock(cfg, i) for i in range(n)])
        self.unembedder = nn.Linear(d, cfg['vocab_size'], bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_embedder(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unembedder(x)


# ── Training loop ──────────────────────────────────────────────────────────────

def run():
    sampler = make_sampler(DEVICE, random_seed=1)
    model = LUTTransformerLinearUnemb(cfg).to(DEVICE)
    model.train()

    n_steps = cfg['n_steps']

    total_params, trainable_params = count_params(model)
    bw = compute_virtual_bandwidth(model)
    print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
    print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB dense  (ratio {bw["lut_ratio"]:.4f})')
    print('Batch schedule: bs=16 (0-5K) → bs=32 (5K-15K) → bs=64 (15K-40K) → bs=128 (40K-100K)')
    print('LR schedule:    warmup→1e-4 (0-5K) → 1.4e-4 (5K-15K) → 2e-4 (15K-40K) → cosine 2e-4→1e-6 (40K-100K)')

    optimizer = torch.optim.Adam(model.parameters(), lr=get_lr(0))

    logger = MetricsLogger(EXP_DIR)
    train_losses, val_losses, steps_log = [], [], []
    best_val_loss = float('inf')
    best_step = 0
    ema = None
    alpha = 0.01
    start_time = time.time()
    current_bs = get_batch_size(0)

    for step in range(n_steps):
        new_bs = get_batch_size(step)
        if new_bs != current_bs:
            print(f'[SCHEDULE] step {step}: batch_size {current_bs} → {new_bs}')
            current_bs = new_bs

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = sampler.sample_training_batch(current_bs).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T), reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lv = loss.item()
        ema = lv if ema is None else (1 - alpha) * ema + alpha * lv

        if step % 100 == 0:
            print(f'step {step:6d} | train_loss={ema:.4f} | lr={lr:.2e} | bs={current_bs}')

        if step % cfg['test_every'] == 0:
            val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])
            print(f'[VAL] step {step}: val_loss={val_loss:.4f}')
            train_losses.append(ema)
            val_losses.append(val_loss)
            steps_log.append(step)
            logger.log(step, ema, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
            model.train()

    logger.close()
    elapsed = time.time() - start_time
    save_loss_plot(EXP_DIR, steps_log, train_losses, val_losses)

    summary = {
        'exp_name': cfg['exp_name'],
        'best_val_loss': best_val_loss,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_step': best_step,
        'total_steps': n_steps,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_hours': round(elapsed / 3600, 3),
        'virtual_bandwidth_MB': bw['lut_MB'],
        'dense_bandwidth_MB': bw['dense_MB'],
        'bandwidth_ratio': bw['lut_ratio'],
    }
    save_summary(EXP_DIR, summary)
    print('\n=== DONE ===')
    print(json.dumps(summary, indent=2))


run()
