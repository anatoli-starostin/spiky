"""
exp141_adam_then_eggroll_pop64_cosine — Fork of exp140.

Two-phase training:
  Phase 1: Adam, 10k steps, warmup+cosine lr schedule, lr=0.001.
  Phase 2: EGGROLL, 40k steps, cosine lr decay 0.01->0.0001, sigma=1.0, pop_size=64.
"""
import sys, os, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, count_params, compute_virtual_bandwidth,
    evaluate_model, save_loss_plot, save_summary,
    MetricsLogger, get_lr_scale,
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


# ── EGGROLL Optimizer ──────────────────────────────────────────────────────────

class EGGROLLOptimizer:

    def __init__(self, params, lr=0.01, sigma=1.0, pop_size=64, rank=1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.sigma = sigma
        self.pop_size = pop_size
        self.rank = rank
        assert pop_size % 2 == 0
        self.n_pairs = pop_size // 2

    def _sample_perturbation(self):
        perts = []
        for p in self.params:
            if p.ndim <= 1:
                noise = torch.randn_like(p)
                noise = noise / (noise.norm() / math.sqrt(noise.numel()) + 1e-8)
                perts.append(('vec', noise))
            else:
                m = p.shape[0]
                n = p.numel() // m
                a = torch.randn(m, self.rank, device=p.device, dtype=p.dtype)
                b = torch.randn(n, self.rank, device=p.device, dtype=p.dtype)
                scale = math.sqrt(m * n * self.rank)
                perts.append(('lr1', a, b, p.shape, scale))
        return perts

    def _apply(self, perts, sign):
        for p, pert in zip(self.params, perts):
            if pert[0] == 'vec':
                p.data.add_(sign * self.sigma * pert[1])
            else:
                _, a, b, shape, scale = pert
                p.data.add_(sign * self.sigma * (a @ b.T).reshape(shape) / scale)

    def step(self, loss_fn, lr=None):
        if lr is None:
            lr = self.lr
        losses_pos, losses_neg, all_perts = [], [], []

        with torch.no_grad():
            for _ in range(self.n_pairs):
                perts = self._sample_perturbation()
                all_perts.append(perts)

                self._apply(perts, +1.0)
                losses_pos.append(loss_fn())
                self._apply(perts, -2.0)
                losses_neg.append(loss_fn())
                self._apply(perts, +1.0)  # restore

            fitness = torch.tensor(
                [ln - lp for lp, ln in zip(losses_pos, losses_neg)],
                dtype=torch.float32,
            )
            if self.n_pairs > 1:
                fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

            eff_lr = lr / self.n_pairs

            for p_idx, param in enumerate(self.params):
                update = torch.zeros_like(param)
                for pair_idx, perts in enumerate(all_perts):
                    f = fitness[pair_idx].item()
                    pert = perts[p_idx]
                    if pert[0] == 'vec':
                        update.add_(f * pert[1])
                    else:
                        _, a, b, shape, scale = pert
                        update.add_(f * (a @ b.T).reshape(shape) / scale)
                param.data.add_(eff_lr * update)

        mean_abs_delta = sum(abs(lp - ln) for lp, ln in zip(losses_pos, losses_neg)) / self.n_pairs
        fitness_std_val = fitness.std().item()
        return (sum(losses_pos) + sum(losses_neg)) / (2 * self.n_pairs), fitness_std_val, mean_abs_delta


def cosine_decay(step, n_steps, lr_max, lr_min):
    progress = step / max(n_steps - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ── Training loop ──────────────────────────────────────────────────────────────

def run():
    sampler = make_sampler(DEVICE, random_seed=1)
    model = LUTTransformerLinearUnemb(cfg).to(DEVICE)
    model.train()

    n_adam_steps    = cfg['n_adam_steps']
    n_eggroll_steps = cfg['n_eggroll_steps']
    n_steps = n_adam_steps + n_eggroll_steps

    total_params, trainable_params = count_params(model)
    bw = compute_virtual_bandwidth(model)
    print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
    print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB dense  (ratio {bw["lut_ratio"]:.4f})')
    print(f'Phase 1: Adam {n_adam_steps} steps, lr={cfg["adam_lr"]}, warmup={cfg["adam_warmup_fraction"]}')
    print(f'Phase 2: EGGROLL {n_eggroll_steps} steps, lr {cfg["eggroll_lr_max"]}→{cfg["eggroll_lr_min"]} cosine, sigma={cfg["eggroll_sigma"]}, pop_size={cfg["eggroll_pop_size"]}')

    # ── Phase 1: Adam ──────────────────────────────────────────────────────────
    adam = torch.optim.Adam(model.parameters(), lr=cfg['adam_lr'])
    warmup_frac = cfg['adam_warmup_fraction']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        adam, lambda step: get_lr_scale(step, n_adam_steps, warmup_frac)
    )

    logger = MetricsLogger(EXP_DIR)
    train_losses, val_losses, steps_log = [], [], []
    best_val_loss = float('inf')
    best_step = 0
    ema = None
    alpha = 0.01
    start_time = time.time()

    print('\n=== Phase 1: Adam ===')
    for step in range(n_adam_steps):
        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), tgt.reshape(B * T), reduction='mean')

        adam.zero_grad()
        loss.backward()
        adam.step()
        scheduler.step()

        lv = loss.item()
        ema = lv if ema is None else (1 - alpha) * ema + alpha * lv

        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            warmup_steps = int(warmup_frac * n_adam_steps)
            phase = 'warmup' if step < warmup_steps else 'decay'
            print(f'step {step:6d} | train_loss={ema:.4f} | lr={lr:.2e} ({phase})')

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

    # ── Phase 2: EGGROLL ───────────────────────────────────────────────────────
    eggroll = EGGROLLOptimizer(
        model.parameters(),
        lr=cfg['eggroll_lr_max'],
        sigma=cfg['eggroll_sigma'],
        pop_size=cfg['eggroll_pop_size'],
        rank=cfg['eggroll_rank'],
    )

    lr_max = cfg['eggroll_lr_max']
    lr_min = cfg['eggroll_lr_min']

    print('\n=== Phase 2: EGGROLL ===')
    for local_step in range(n_eggroll_steps):
        step = n_adam_steps + local_step

        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        def loss_fn():
            logits = model(inp)
            B, T, V = logits.shape
            return F.cross_entropy(
                logits.reshape(B * T, V), tgt.reshape(B * T), reduction='mean'
            ).item()

        current_lr = cosine_decay(local_step, n_eggroll_steps, lr_max, lr_min)
        mean_loss, fitness_std, mean_abs_delta = eggroll.step(loss_fn, lr=current_lr)

        ema = mean_loss if ema is None else (1 - alpha) * ema + alpha * mean_loss

        if step % 100 == 0:
            print(f'step {step:6d} | train_loss={ema:.4f} | lr={current_lr:.2e} | fit_std={fitness_std:.4f} | delta={mean_abs_delta:.4f}')

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
