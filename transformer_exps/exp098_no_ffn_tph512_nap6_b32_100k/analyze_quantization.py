"""
Quantization sensitivity analysis for exp098 checkpoint.

Simulates weight quantization at different precisions:
  fp32 (baseline), fp16, bf16, fp8 (simulated), int8, int4, int2

For each precision:
  1. Quantize all LUT projection.weights
  2. Run full test set evaluation
  3. Report val_loss and delta vs fp32 baseline

Also reports per-precision weight statistics (how many distinct values are used).
"""
import sys, os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(EXP_DIR, '..', '..'))
sys.path.insert(0, EXP_DIR)

# Import model classes directly from the experiment's train.py
import importlib.util
spec = importlib.util.spec_from_file_location('train098', os.path.join(EXP_DIR, 'train.py'))
train098 = importlib.util.load_from_spec = None  # avoid running run()

# Safer: exec just the class definitions
_src = open(os.path.join(EXP_DIR, 'train.py')).read()
# Stop before the run() call at the bottom
_src = _src[:_src.rfind('sampler = make_sampler')]
exec(_src, globals())

from transformer_exps.common import make_sampler, evaluate_model
from spiky.lutorch.multi_head_lut import MultiHeadLut

DEVICE = 'cuda:0'

# ── Load checkpoint ────────────────────────────────────────────────────────────

print('Loading checkpoint...')
ckpt = torch.load(os.path.join(EXP_DIR, 'checkpoint.pt'), map_location='cpu',
                  weights_only=False)

model = LUTTransformerV2(cfg).to(DEVICE)
model.load_state_dict(ckpt)
model.eval()

sampler = make_sampler(DEVICE, random_seed=1)

# ── Quantization functions ────────────────────────────────────────────────────

def quantize_to_int(w, bits):
    """Symmetric uniform int quantization."""
    w_max = w.abs().max()
    if w_max == 0:
        return w
    scale = w_max / (2 ** (bits - 1) - 1)
    return (w / scale).round().clamp(-(2**(bits-1)-1), 2**(bits-1)-1) * scale

def quantize_fp8_e4m3(w):
    """Simulate fp8 E4M3: keep top 3 mantissa bits of fp16 representation."""
    w_fp16 = w.half()
    # fp16: 1 sign + 5 exp + 10 mantissa bits
    # Keep top 3 mantissa bits → zero lower 7: mask = 1111 1110 0000 0000
    # As int32 to avoid overflow, then mask, then view as int16
    bits = w_fp16.view(torch.int16).to(torch.int32)
    bits = bits & 0xFE00
    return bits.to(torch.int16).view(torch.float16).float()

def quantize_fp16(w):
    return w.half().float()

def quantize_bf16(w):
    return w.bfloat16().float()

def count_distinct(w, tol=1e-6):
    """Approximate count of distinct values."""
    vals = w.detach().flatten().cpu().numpy()
    vals_sorted = np.sort(vals)
    diffs = np.diff(vals_sorted)
    return int((diffs > tol).sum()) + 1

# ── Apply quantization to all LUT weights ─────────────────────────────────────

def apply_quantization(model, quant_fn):
    """Apply quant_fn to all MultiHeadLut projection.weights in-place."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MultiHeadLut):
                w = module.projection.weights
                module.projection.weights.copy_(quant_fn(w))

def save_original_weights(model):
    return {id(m): m.projection.weights.clone()
            for m in model.modules() if isinstance(m, MultiHeadLut)}

def restore_weights(model, saved):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, MultiHeadLut) and id(m) in saved:
                m.projection.weights.copy_(saved[id(m)])

# ── Run evaluation ─────────────────────────────────────────────────────────────

print('Evaluating...\n')
original_weights = save_original_weights(model)

# Collect all LUT weights for stats
all_weights = torch.cat([m.projection.weights.flatten()
                         for m in model.modules() if isinstance(m, MultiHeadLut)])
print(f'Total LUT weight elements: {all_weights.numel():,}')
print(f'Weight range: [{all_weights.min():.4f}, {all_weights.max():.4f}]')
print(f'Weight std:   {all_weights.std():.4f}\n')

precisions = [
    ('fp32 (baseline)', None),
    ('bf16',            quantize_bf16),
    ('fp16',            quantize_fp16),
    ('fp8 (sim E4M3)',  quantize_fp8_e4m3),
    ('int8',            lambda w: quantize_to_int(w, 8)),
    ('int4',            lambda w: quantize_to_int(w, 4)),
    ('int2',            lambda w: quantize_to_int(w, 2)),
]

results = []
print(f"{'Precision':<20}  {'val_loss':>9}  {'delta':>7}  {'distinct_vals':>13}  {'max_err':>8}")
print('-' * 70)

baseline_loss = None
for name, quant_fn in precisions:
    restore_weights(model, original_weights)

    if quant_fn is not None:
        apply_quantization(model, quant_fn)

    # Count distinct values and max error after quantization
    q_weights = torch.cat([m.projection.weights.flatten()
                           for m in model.modules() if isinstance(m, MultiHeadLut)])
    n_distinct = count_distinct(q_weights)
    max_err = (q_weights - all_weights.to(q_weights.device)).abs().max().item() if quant_fn else 0.0

    val_loss = evaluate_model(model, sampler, cfg['test_batch_size'])

    if baseline_loss is None:
        baseline_loss = val_loss
    delta = val_loss - baseline_loss

    results.append((name, val_loss, delta, n_distinct, max_err))
    print(f'{name:<20}  {val_loss:>9.4f}  {delta:>+7.4f}  {n_distinct:>13,}  {max_err:>8.5f}')

restore_weights(model, original_weights)
print('\nDone.')
