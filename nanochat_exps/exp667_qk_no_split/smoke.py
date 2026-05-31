#!/usr/bin/env python3
"""Smoke test: exp667 — qk_lut at S=1 (no split). Verify shape, param count
unchanged vs exp666, finite grads."""
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
H    = cfg['n_heads']
d_qk = cfg['d_qk']
S    = int(cfg['qk_sparsify_s'])
nap  = int(cfg['qk_input_nap'])
tph  = int(cfg['qk_tph'])

assert S == 1, f"expected qk_sparsify_s=1, got {S}"
qk_dim = 2 * d_qk

print(f'cfg: E={E} H={H} d_qk={d_qk} S={S} qk_dim={qk_dim}')
print(f'qk_lut: n_heads=H*S={H*S} tph={tph} NAP={nap} K={2**nap} n_out=qk_dim/S={qk_dim//S}')

qk_lut = TinyMultiHeadLut(
    input_dim=E,
    n_heads=H * S,
    n_outputs=qk_dim // S,
    n_anchor_pairs=nap,
    tables_per_head=tph,
    random_seed=cfg['random_seed'],
    device=DEV,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
print(f'qk_lut weights: {tuple(qk_lut.weights.shape)} numel={qk_lut.weights.numel():,}')

# Expected per-layer: H*S * tph * 2^NAP * (qk_dim/S) — invariant to S
expected = H * S * tph * (2**nap) * (qk_dim // S)
expected_param_match = H * tph * (2**nap) * qk_dim
assert expected == expected_param_match, 'param-match check failed in math'
print(f'expected = {expected:,} ({expected/1e6:.2f}M per layer)')
assert qk_lut.weights.numel() == expected, f'mismatch: {qk_lut.weights.numel():,} vs {expected:,}'
# exp666 reference: 786,432
assert expected == 786_432, f'expected 786,432; got {expected:,}'

# Forward / backward
B, T = 2, 8
x = torch.randn(B * T, E, device=DEV, requires_grad=True)
qk_out = qk_lut(x)
print(f'qk_out shape: {qk_out.shape}  expected: [{B*T}, {H*S}, {qk_dim//S}]')
assert qk_out.shape == (B * T, H * S, qk_dim // S)

# Reshape to per-real-head — with S=1 it's a no-op
qk_per_head = qk_out.reshape(B * T, H, qk_dim)
print(f'qk_per_head shape: {qk_per_head.shape}  expected: [{B*T}, {H}, {qk_dim}]')
assert qk_per_head.shape == (B * T, H, qk_dim)

# With S=1, qk_out and qk_per_head are bytewise identical
for bt in [0, 7]:
    for h in [0, 5]:
        for n in [0, 64, 127]:
            a = qk_per_head[bt, h, n].item()
            b = qk_out[bt, h, n].item()  # S=1 → identical addressing
            assert abs(a - b) < 1e-6, f'reshape unexpected mismatch bt={bt} h={h} n={n}'
print('S=1 reshape identity verified ✓')

loss = qk_out.sum()
loss.backward()
assert torch.isfinite(x.grad).all(), 'x grad non-finite'
assert torch.isfinite(qk_lut.weights.grad).all(), 'qk_lut weights grad non-finite'
assert qk_lut.weights.grad.abs().sum().item() > 0, 'qk_lut grad zero'
print('backward grads finite, qk_lut grad nonzero ✓')

print('\nSMOKE TEST PASSED')
