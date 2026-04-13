"""
Quantization analysis for exp146 checkpoint.

Checks:
  1. How close are weights to the D=256 grid (k/256)?
  2. Standard precision comparisons: fp32, bf16, fp16, fp8, int8, int4, int2
  3. D=256 snap (the trained grid) — should be near-free given the regularization.
"""
import sys, os, json
import torch
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(EXP_DIR, '..', '..'))

# Load model class from train.py
_src = open(os.path.join(EXP_DIR, 'train.py')).read()
_src = _src[:_src.rfind('sampler = make_sampler')]
exec(_src, globals())

from transformer_exps.common import make_sampler, evaluate_model
from spiky.lutorch.multi_head_lut import MultiHeadLut

DEVICE = 'cuda:0'

# ── Load checkpoint ────────────────────────────────────────────────────────────

print('Loading checkpoint...')
model = LUTTransformerLinearUnemb().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location=DEVICE, weights_only=True))
model.eval()

sampler = make_sampler(DEVICE, random_seed=1)

# ── Collect all LUT weights ────────────────────────────────────────────────────

all_weights = torch.cat([m.projection.weights.detach().flatten()
                         for m in model.modules() if isinstance(m, MultiHeadLut)])
print(f'\nTotal LUT weight elements: {all_weights.numel():,}')
print(f'Weight range: [{all_weights.min():.4f}, {all_weights.max():.4f}]')
print(f'Weight std:   {all_weights.std():.4f}')

# ── Grid alignment analysis ────────────────────────────────────────────────────

D = cfg['quant_reg_D']
residuals = (all_weights * D) - (all_weights * D).round()
print(f'\n--- Grid alignment (D={D}, spacing=1/{D}={1/D:.4f}) ---')
print(f'Mean |residual|: {residuals.abs().mean().item():.5f}')
print(f'Max  |residual|: {residuals.abs().max().item():.5f}')
print(f'Frac within 10% of grid: {(residuals.abs() < 0.05).float().mean().item():.3%}')
print(f'Frac within  1% of grid: {(residuals.abs() < 0.005).float().mean().item():.3%}')

# ── Quantization helpers ───────────────────────────────────────────────────────

def quantize_to_int(w, bits):
    w_max = w.abs().max()
    if w_max == 0:
        return w
    scale = w_max / (2 ** (bits - 1) - 1)
    return (w / scale).round().clamp(-(2**(bits-1)-1), 2**(bits-1)-1) * scale

def quantize_fp8_e4m3(w):
    w_fp16 = w.half()
    bits = w_fp16.view(torch.int16).to(torch.int32) & 0xFE00
    return bits.to(torch.int16).view(torch.float16).float()

def quantize_fp16(w): return w.half().float()
def quantize_bf16(w): return w.bfloat16().float()
def snap_to_grid(w, D): return (w * D).round() / D

def count_distinct(w, tol=1e-6):
    vals = np.sort(w.detach().cpu().float().flatten().numpy())
    return int((np.diff(vals) > tol).sum()) + 1

def save_weights(model):
    return {id(m): m.projection.weights.clone()
            for m in model.modules() if isinstance(m, MultiHeadLut)}

def restore_weights(model, saved):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiHeadLut) and id(m) in saved:
                m.projection.weights.copy_(saved[id(m)])

def apply_quant(model, fn):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiHeadLut):
                m.projection.weights.copy_(fn(m.projection.weights))

# ── Evaluate all precisions ────────────────────────────────────────────────────

print('\nEvaluating...\n')
original = save_weights(model)

precisions = [
    ('fp32 (baseline)',    None),
    (f'D={D} snap',        lambda w: snap_to_grid(w, D)),
    ('bf16',               quantize_bf16),
    ('fp16',               quantize_fp16),
    ('fp8 (sim E4M3)',     quantize_fp8_e4m3),
    ('int8',               lambda w: quantize_to_int(w, 8)),
    ('int4',               lambda w: quantize_to_int(w, 4)),
    ('int2',               lambda w: quantize_to_int(w, 2)),
]

print(f"{'Precision':<20}  {'val_loss':>9}  {'delta':>7}  {'distinct_vals':>13}  {'max_err':>8}")
print('-' * 72)

baseline_loss = None
for name, quant_fn in precisions:
    restore_weights(model, original)
    if quant_fn is not None:
        apply_quant(model, quant_fn)

    q_weights = torch.cat([m.projection.weights.detach().flatten()
                           for m in model.modules() if isinstance(m, MultiHeadLut)])
    n_distinct = count_distinct(q_weights)
    max_err = (q_weights - all_weights).abs().max().item() if quant_fn else 0.0

    val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])
    if baseline_loss is None:
        baseline_loss = val_loss
    delta = val_loss - baseline_loss

    print(f'{name:<20}  {val_loss:>9.4f}  {delta:>+7.4f}  {n_distinct:>13,}  {max_err:>8.5f}')

restore_weights(model, original)
print('\nDone.')
