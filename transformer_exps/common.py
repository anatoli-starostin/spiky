"""
Shared dataset, constants, evaluation, logging, and training utilities
for transformer experiments.
"""
import os
import sys
import json
import csv
import math
import time
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/spiky'))

from spiky.util.text_snippet_sampler import TextSnippetSampler

# ── Constants ──────────────────────────────────────────────────────────────────
CONTEXT_SIZE = 32
RAW_VOCAB_SIZE = 256
BOS_ID = RAW_VOCAB_SIZE          # 256
VOCAB_SIZE = RAW_VOCAB_SIZE + 1  # 257
TESTING_LENGTH = 10_000
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'workbooks', 'fineweb_texts.txt')
)

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_sampler(device, random_seed=1):
    return TextSnippetSampler(
        DATA_PATH,
        CONTEXT_SIZE,
        TESTING_LENGTH,
        device,
        random_seed=random_seed,
    )

# ── Model utils ────────────────────────────────────────────────────────────────

def count_params(model):
    """Returns (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def collect_internal_loss(model) -> torch.Tensor:
    """
    Sum internal_loss() from all MultiHeadLut modules in the model.

    Returns a scalar tensor (0.0 if no LUT modules have a loss), or None if
    no internal losses were found. Use with a `delta_reg_weight` config key:

        loss = ce_loss + cfg.get('delta_reg_weight', 0.0) * collect_internal_loss(model)
    """
    try:
        from spiky.lutorch.multi_head_lut import MultiHeadLut
    except ImportError:
        return None

    losses = []
    for module in model.modules():
        if isinstance(module, MultiHeadLut):
            il = module.internal_loss()
            if il is not None:
                losses.append(il)
    if not losses:
        return None
    return torch.stack(losses).mean()


@contextmanager
def hard_mode(model):
    """
    Temporarily switch all smooth-mode modules to hard (non-blending) inference.

    Affects:
      - MultiHeadLut: smooth_mode=False → return_alternatives=False, no blending
      - MultiHeadLut.projection (LProjection): smooth_mode=False → no uncertainty blending
      - RankAttention: smooth_mode=False → STE/hard sign projection instead of soft
    """
    try:
        from spiky.lutorch.multi_head_lut import MultiHeadLut
        from spiky.lutorch.ranking_tools import RankAttention
    except ImportError:
        yield
        return

    saved = []
    for m in model.modules():
        if isinstance(m, MultiHeadLut):
            saved.append((m, 'smooth_mode', m.smooth_mode))
            saved.append((m.projection, 'smooth_mode', m.projection.smooth_mode))
            m.smooth_mode = False
            m.projection.smooth_mode = False
        elif isinstance(m, RankAttention):
            saved.append((m, 'smooth_mode', m.smooth_mode))
            m.smooth_mode = False
    try:
        yield
    finally:
        for m, attr, val in saved:
            setattr(m, attr, val)


def compute_virtual_bandwidth(model, seq_len=CONTEXT_SIZE, bytes_per_value=4):
    """
    Estimate virtual bandwidth (bytes read from memory) per forward pass for one sequence.

    Dense layers read ALL weights for every token.
    LUT layers read 1 entry per table at inference (smooth_mode=False), or 1+n_alternatives if smooth_mode=True.

    This is 'virtual' because GPU doesn't exploit LUT sparsity — but specialised hardware would.

    Returns:
        dict with:
            dense_bytes  — bytes read if everything were dense
            lut_bytes    — bytes actually read given LUT sparsity
            lut_ratio    — lut_bytes / dense_bytes
            breakdown    — per-module-type byte counts
    """
    # Import here to avoid circular deps at module level
    try:
        from spiky.lutorch.multi_head_lut import MultiHeadLut
    except ImportError:
        MultiHeadLut = None

    dense_bytes = 0
    lut_bytes = 0
    breakdown = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # reads all weights per token
            reads_per_token = module.in_features * module.out_features
            total = reads_per_token * seq_len * bytes_per_value
            dense_bytes += total
            lut_bytes += total
            breakdown[name] = {'type': 'Linear', 'bytes': total}

        elif isinstance(module, torch.nn.Embedding):
            # reads one row per token
            reads_per_token = module.embedding_dim
            total = reads_per_token * seq_len * bytes_per_value
            dense_bytes += module.num_embeddings * module.embedding_dim * bytes_per_value  # dense would read all
            lut_bytes += total
            breakdown[name] = {'type': 'Embedding', 'bytes': total}

        elif MultiHeadLut is not None and isinstance(module, MultiHeadLut):
            # smooth_mode=False: inference reads 1 entry per table (best match only)
            # smooth_mode=True: inference blends multiple entries (1 + n_alternatives)
            entries_read = (1 + module.n_alternatives) if module.smooth_mode else 1
            reads_per_token = module.n_heads * module.tables_per_head * entries_read * module.n_outputs
            # dense equivalent: reads the full table
            entries_per_table = 2 ** module.n_anchor_pairs
            dense_reads_per_token = module.n_heads * module.tables_per_head * entries_per_table * module.n_outputs
            lut_total = reads_per_token * seq_len * bytes_per_value
            dense_total = dense_reads_per_token * seq_len * bytes_per_value
            dense_bytes += dense_total
            lut_bytes += lut_total
            breakdown[name] = {
                'type': 'MultiHeadLut',
                'lut_bytes': lut_total,
                'dense_bytes': dense_total,
                'sparsity': f'1/{entries_per_table}',
            }

    lut_ratio = lut_bytes / dense_bytes if dense_bytes > 0 else 1.0
    return {
        'dense_bytes': dense_bytes,
        'lut_bytes': lut_bytes,
        'lut_ratio': round(lut_ratio, 6),
        'dense_MB': round(dense_bytes / 1e6, 2),
        'lut_MB': round(lut_bytes / 1e6, 2),
        'breakdown': breakdown,
    }

# ── Evaluation ─────────────────────────────────────────────────────────────────

def generate_text(model, prefix, length, device):
    """Multinomial generation for a byte-level causal model."""
    ctx = list(prefix.encode('utf-8'))
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
            x[0, 0] = BOS_ID
            trunc = ctx[-(CONTEXT_SIZE - 1):]
            if trunc:
                x[0, -len(trunc):] = torch.tensor(trunc, dtype=torch.long, device=device)
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :RAW_VOCAB_SIZE], dim=-1)[0]
            next_id = torch.multinomial(probs, 1).item()
            ctx.append(next_id)
    return bytes(c for c in ctx if 0 <= c < 256).decode('utf-8', errors='replace')


def evaluate_model(model, sampler, batch_size=256):
    """Mean cross-entropy loss over full test set. Prints a generation sample."""
    model.eval()
    losses = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            tgt = batch.long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V), tgt.reshape(B * T), reduction='mean'
            )
            losses.append(loss.item())

    gen = generate_text(model, 'Once upon a time ', length=80, device=device)
    print(f'\n[GEN]: {gen}\n')

    model.train()
    return sum(losses) / len(losses)

# ── Logging ────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """Incrementally writes step/train_loss/val_loss rows to metrics.csv."""

    def __init__(self, exp_dir):
        self.path = os.path.join(exp_dir, 'metrics.csv')
        self._f = open(self.path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow(['step', 'train_loss', 'val_loss', 'val_loss_hard'])

    def log(self, step, train_loss, val_loss, val_loss_hard=None):
        self._w.writerow([step, f'{train_loss:.6f}', f'{val_loss:.6f}',
                          f'{val_loss_hard:.6f}' if val_loss_hard is not None else ''])
        self._f.flush()

    def close(self):
        self._f.close()


def save_loss_plot(exp_dir, steps, train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_losses, label='train')
    plt.plot(steps, val_losses, label='val')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss.png'), dpi=100)
    plt.close()


def save_summary(exp_dir, data: dict):
    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(data, f, indent=2)

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr_scale(step, num_steps, warmup=0.1):
    """Linear warmup + cosine decay to 10% of peak."""
    w = int(warmup * num_steps)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(num_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def make_scheduler(optimizer, cfg):
    """Build scheduler from config. Supports 'warmup_cosine' and 'cosine'."""
    schedule = cfg.get('lr_schedule', 'warmup_cosine')
    n_steps = cfg['n_steps']
    if schedule == 'warmup_cosine':
        warmup = cfg.get('lr_warmup_fraction', 0.1)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: get_lr_scale(step, n_steps, warmup)
        )
    elif schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    else:
        return None


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Standard training harness for byte-level LM experiments.

    Expected cfg keys:
        n_steps, batch_size, test_batch_size, test_every,
        lr, lr_schedule ('cosine' | 'none'), random_seed, exp_name
    """

    def __init__(self, model, sampler, cfg, exp_dir, optimizer=None, scheduler=None):
        self.model = model
        self.sampler = sampler
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = next(model.parameters()).device

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            opt_name = cfg.get('optimizer', 'Adam')
            wd = cfg.get('weight_decay', 0.0)
            if opt_name == 'AdamW':
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=wd)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = make_scheduler(self.optimizer, cfg)

    def run(self):
        cfg = self.cfg
        model = self.model
        logger = MetricsLogger(self.exp_dir)

        total_params, trainable_params = count_params(model)
        bw = compute_virtual_bandwidth(model)
        print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')
        print(f'Virtual bandwidth: {bw["lut_MB"]:.1f} MB / {bw["dense_MB"]:.1f} MB dense  (ratio {bw["lut_ratio"]:.4f})')

        train_losses, val_losses, steps_log = [], [], []
        best_val_loss = float('inf')
        best_step = 0
        ema = None
        alpha = 0.01
        start_time = time.time()

        smooth_anneal_step = cfg.get('smooth_anneal_step', None)
        annealed = False

        model.train()
        for step in range(cfg['n_steps']):
            if smooth_anneal_step is not None and step == smooth_anneal_step and not annealed:
                try:
                    from spiky.lutorch.multi_head_lut import MultiHeadLut
                    from spiky.lutorch.ranking_tools import RankAttention
                except ImportError:
                    pass
                else:
                    for m in model.modules():
                        if isinstance(m, MultiHeadLut):
                            m.smooth_mode = False
                            m.projection.smooth_mode = False
                        elif isinstance(m, RankAttention):
                            m.smooth_mode = False
                annealed = True
                print(f'[ANNEAL] step {step}: switched to hard mode (smooth_mode=False)')
            x = self.sampler.sample_training_batch(cfg['batch_size']).long()
            inp = torch.empty_like(x)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = x[:, :-1]
            tgt = x

            logits = model(inp)
            B, T, V = logits.shape
            ce_loss = F.cross_entropy(
                logits.reshape(B * T, V), tgt.reshape(B * T), reduction='mean'
            )

            delta_reg_weight = cfg.get('delta_reg_weight', 0.0)
            il = None
            if delta_reg_weight > 0.0:
                il = collect_internal_loss(model)
            loss = ce_loss if (il is None) else ce_loss + delta_reg_weight * il

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            lv = ce_loss.item()
            ema = lv if ema is None else (1 - alpha) * ema + alpha * lv

            if step % 100 == 0:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else cfg['lr']
                warmup_steps = int(cfg.get('lr_warmup_fraction', 0.0) * cfg['n_steps'])
                phase = 'warmup' if step < warmup_steps else 'decay'
                il_str = f' | int_loss={il.item():.4f}' if il is not None else ''
                print(f'step {step:6d} | train_loss={ema:.4f}{il_str} | lr={lr:.2e} ({phase})')

            if step % cfg['test_every'] == 0:
                val_loss = evaluate_model(model, self.sampler, cfg['test_batch_size'])
                with hard_mode(model):
                    val_loss_hard = evaluate_model(model, self.sampler, cfg['test_batch_size'])
                print(f'[VAL] step {step}: val_loss={val_loss:.4f} | hard={val_loss_hard:.4f} | gap={val_loss_hard - val_loss:.4f}')
                train_losses.append(ema)
                val_losses.append(val_loss)
                steps_log.append(step)
                logger.log(step, ema, val_loss, val_loss_hard)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step

        logger.close()
        elapsed = time.time() - start_time

        save_loss_plot(self.exp_dir, steps_log, train_losses, val_losses)

        summary = {
            'exp_name': cfg['exp_name'],
            'best_val_loss': best_val_loss,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_step': best_step,
            'total_steps': cfg['n_steps'],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time_hours': round(elapsed / 3600, 3),
            'virtual_bandwidth_MB': bw['lut_MB'],
            'dense_bandwidth_MB': bw['dense_MB'],
            'bandwidth_ratio': bw['lut_ratio'],
        }
        save_summary(self.exp_dir, summary)
        print('\n=== DONE ===')
        print(json.dumps(summary, indent=2))
        return summary
