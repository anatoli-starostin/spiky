"""
fp8 dynamic rounding study.

Trains 3 identical models for 10k steps (exp135 arch):
  1. fp32 baseline
  2. fp8 rounding before each forward pass (straight-through gradients)
  3. bf16 rounding before each forward pass (straight-through gradients)

Rounding is applied in-place to all MultiHeadLut projection.weights before
each forward pass. Gradients flow through the rounding unchanged (STE).

Plots val_loss curves to check if reduced precision training converges.
"""
import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_exps.common import (
    make_sampler, evaluate_model, MetricsLogger,
    CONTEXT_SIZE, BOS_ID,
)
from spiky.lutorch.multi_head_lut import MultiHeadLut

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Reuse exp135 model ─────────────────────────────────────────────────────────
_exp135_dir = os.path.join(EXP_DIR, '..', 'exp135_full_coverage_tph264_480')
_src = open(os.path.join(_exp135_dir, 'train.py')).read()
_src = _src[:_src.rfind('sampler = make_sampler')]
# patch EXP_DIR so config.json loads from exp135
_src = _src.replace(
    "EXP_DIR = os.path.dirname(os.path.abspath(__file__))",
    f"EXP_DIR = r'{os.path.abspath(_exp135_dir)}'"
)
exec(_src, globals())

DEVICE = 'cuda:0'

# ── Rounding functions ─────────────────────────────────────────────────────────

def round_fp8(w):
    """Round to fp8 E4M3 precision (simulate via int16 mantissa truncation)."""
    w16 = w.half()
    bits = w16.view(torch.int16).to(torch.int32) & 0xFE00
    return bits.to(torch.int16).view(torch.float16).float()

def round_bf16(w):
    return w.bfloat16().float()

def apply_rounding(model, round_fn):
    """Round all LUT weights in-place (for forward pass)."""
    for m in model.modules():
        if isinstance(m, MultiHeadLut):
            m.projection.weights.data.copy_(round_fn(m.projection.weights.data))

# ── Train one model ────────────────────────────────────────────────────────────

def train(mode, sampler):
    """mode: 'fp32', 'fp8', 'bf16'"""
    torch.manual_seed(cfg['random_seed'])
    model = LUTTransformerLinearUnemb(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    round_fn = None
    if mode == 'fp8':
        round_fn = round_fp8
    elif mode == 'bf16':
        round_fn = round_bf16

    val_losses = []
    steps_log = []
    N_STEPS = 10001
    TEST_EVERY = 500

    for step in range(N_STEPS):
        if round_fn is not None:
            apply_rounding(model, round_fn)

        x = sampler.sample_training_batch(32).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        tgt = x

        logits = model(inp)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B*T, V), tgt.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f'  [{mode}] step {step:5d} | loss={loss.item():.4f}')

        if step % TEST_EVERY == 0:
            val_loss = evaluate_model(model, sampler, 256)
            val_losses.append(val_loss)
            steps_log.append(step)
            print(f'  [{mode}] [VAL] step {step}: val_loss={val_loss:.4f}')
            model.train()

    return steps_log, val_losses

# ── Run ────────────────────────────────────────────────────────────────────────

sampler = make_sampler(DEVICE, random_seed=1)

results = {}
for mode in ['fp32', 'bf16', 'fp8']:
    print(f'\n=== {mode} ===')
    steps, losses = train(mode, sampler)
    results[mode] = (steps, losses)

# ── Save data ──────────────────────────────────────────────────────────────────

torch.save(results, os.path.join(EXP_DIR, 'results.pt'))
print('\nResults saved.')

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
colors = {'fp32': 'steelblue', 'bf16': 'darkorange', 'fp8': 'green'}
for mode, (steps, losses) in results.items():
    ax.plot(steps, losses, label=mode, color=colors[mode], marker='o', markersize=3)

ax.set_xlabel('step')
ax.set_ylabel('val_loss')
ax.set_title('Val loss: fp32 vs bf16 vs fp8 dynamic rounding (10k steps, exp135 arch)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'fp8_study.png'), dpi=130)
print('Plot saved.')
