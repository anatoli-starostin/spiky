#!/usr/bin/env python3
"""Smoke test: exp668 — qk_lut at S=1 with tph=128 (2x exp667). Verify shape,
param count = 1.57M/layer = 9.43M total qk; finite grads."""
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
N_L  = cfg['num_layers']

assert S == 1, f"expected qk_sparsify_s=1, got {S}"
assert tph == 128, f"expected qk_tph=128, got {tph}"
qk_dim = 2 * d_qk

print(f'cfg: E={E} H={H} d_qk={d_qk} S={S} qk_dim={qk_dim} tph={tph}')
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

# Expected per-layer = 6 * 128 * 16 * 128 = 1,572,864
expected_layer = H * S * tph * (2**nap) * (qk_dim // S)
print(f'expected per-layer = {expected_layer:,} ({expected_layer/1e6:.2f}M)')
assert qk_lut.weights.numel() == expected_layer, f'mismatch: {qk_lut.weights.numel():,} vs {expected_layer:,}'
assert expected_layer == 1_572_864, f'expected 1,572,864; got {expected_layer:,}'

# Total qk over 6 layers: +4.72M vs exp667
expected_total = expected_layer * N_L
delta_vs_exp667 = expected_total - (H * S * 64 * (2**nap) * (qk_dim // S)) * N_L
print(f'total qk over {N_L} layers = {expected_total:,} ({expected_total/1e6:.2f}M)')
print(f'delta vs exp667 (tph=64) = +{delta_vs_exp667:,} (+{delta_vs_exp667/1e6:.2f}M)')

# Forward / backward
B, T = 2, 8
x = torch.randn(B * T, E, device=DEV, requires_grad=True)
qk_out = qk_lut(x)
print(f'qk_out shape: {qk_out.shape}  expected: [{B*T}, {H}, {qk_dim}]')
assert qk_out.shape == (B * T, H * S, qk_dim // S)

loss = qk_out.sum()
loss.backward()
assert torch.isfinite(x.grad).all(), 'x grad non-finite'
assert torch.isfinite(qk_lut.weights.grad).all(), 'qk_lut weights grad non-finite'
assert qk_lut.weights.grad.abs().sum().item() > 0, 'qk_lut grad zero'
print('backward grads finite, qk_lut grad nonzero ✓')

print('\nSMOKE TEST PASSED')
