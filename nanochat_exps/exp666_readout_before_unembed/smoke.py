#!/usr/bin/env python3
"""Smoke test: exp666 — build read_out LUT, verify shape & param count
(6.29M), residual flow, finite grads."""
import sys, json, os
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])

E    = cfg['embedding_dim']
nh   = int(cfg['readout_n_heads'])
nap  = int(cfg['readout_input_nap'])
tph  = int(cfg['readout_tph'])
n_per_head = E // nh

print(f'cfg: E={E} readout n_heads={nh} NAP={nap} K={2**nap} tph={tph} n_per_head={n_per_head}')

read_out = TinyMultiHeadLut(
    input_dim=E,
    n_heads=nh,
    n_outputs=n_per_head,
    n_anchor_pairs=nap,
    tables_per_head=tph,
    random_seed=cfg['random_seed'] + 800,
    device=DEV,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
print(f'read_out weights: {tuple(read_out.weights.shape)} numel={read_out.weights.numel():,}')

# Expected: nh * tph * 2^nap * n_per_head
expected = nh * tph * (2**nap) * n_per_head
print(f'expected = {expected:,} ({expected/1e6:.2f}M)')
assert read_out.weights.numel() == expected, f'mismatch: {read_out.weights.numel():,} vs {expected:,}'
assert expected == 6_291_456, f'expected 6.29M; got {expected:,}'

# Forward / backward (B,T) -> flatten -> LUT -> residual on E
B, T = 2, 8
x = torch.randn(B, T, E, device=DEV, requires_grad=True)
x_flat = x.reshape(B * T, E)
ro = read_out(x_flat).reshape(B * T, E)
assert ro.shape == (B * T, E), f'ro shape: {ro.shape}'
y = x + ro.reshape(B, T, E)                                       # residual
print(f'forward: x {x.shape}, ro {ro.shape}, y {y.shape}')

loss = y.sum()
loss.backward()
assert torch.isfinite(x.grad).all(),               'x grad non-finite'
assert torch.isfinite(read_out.weights.grad).all(),'read_out weights grad non-finite'
assert read_out.weights.grad.abs().sum().item() > 0, 'read_out grad is zero'
print('backward grads finite, read_out grad nonzero ✓')

print('\nSMOKE TEST PASSED')
