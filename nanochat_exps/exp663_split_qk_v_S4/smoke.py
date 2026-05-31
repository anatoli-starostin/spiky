#!/usr/bin/env python3
"""Smoke test: exp663 split-S qk_lut + v_lut shapes + reshape layout + finite grads."""
import sys, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']; H = cfg['n_heads']; d_qk = cfg['d_qk']; d_v = cfg['d_v']
Sqk = int(cfg['qk_sparsify_s'])
Sv  = int(cfg['v_sparsify_s'])

print(f'cfg: E={E} H={H} d_qk={d_qk} d_v={d_v} Sqk={Sqk} Sv={Sv}')
print(f'qk: n_heads={H*Sqk} n_out={2*d_qk//Sqk} NAP={cfg["qk_input_nap"]} tph={cfg["qk_tph"]} K={2**cfg["qk_input_nap"]}')
print(f'v:  n_heads={H*Sv}  n_out={d_v//Sv}    NAP={cfg["v_input_nap"]}  tph={cfg["v_tph"]}  K={2**cfg["v_input_nap"]}')

qk_lut = TinyMultiHeadLut(
    input_dim=E, n_heads=H*Sqk, n_outputs=2*d_qk//Sqk,
    n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
    random_seed=cfg['random_seed'], device=DEV,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
v_lut = TinyMultiHeadLut(
    input_dim=E, n_heads=H*Sv, n_outputs=d_v//Sv,
    n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
    random_seed=cfg['random_seed']+200, device=DEV,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
print(f'qk_lut weights shape: {qk_lut.weights.shape} n_params: {qk_lut.weights.numel():,}')
print(f'v_lut  weights shape: {v_lut.weights.shape} n_params: {v_lut.weights.numel():,}')

B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)

qk_out = qk_lut(x)
print(f'qk_out shape: {qk_out.shape}  expected: [{B*T}, {H*Sqk}, {2*d_qk//Sqk}]')
assert qk_out.shape == (B*T, H*Sqk, 2*d_qk//Sqk)
qk_per_head = qk_out.reshape(B*T, H, 2*d_qk)
print(f'qk_per_head shape: {qk_per_head.shape}  expected: [{B*T}, {H}, {2*d_qk}]')

# Verify reshape layout: contiguous sub-heads of real head h are LUT heads h*Sqk..h*Sqk+Sqk-1
for bt in [0, 1]:
    for h in [0, 1, 2]:
        for s in range(Sqk):
            for n in range(2*d_qk//Sqk):
                a = qk_per_head[bt, h, s*(2*d_qk//Sqk) + n].item()
                b = qk_out[bt, h*Sqk + s, n].item()
                assert abs(a - b) < 1e-6, f'qk reshape mismatch bt={bt} h={h} s={s} n={n}'
print('qk reshape layout verified ✓')

v_out = v_lut(x)
print(f'v_out shape: {v_out.shape}  expected: [{B*T}, {H*Sv}, {d_v//Sv}]')
assert v_out.shape == (B*T, H*Sv, d_v//Sv)
v_per_head = v_out.reshape(B*T, H, d_v)
print(f'v_per_head shape: {v_per_head.shape}  expected: [{B*T}, {H}, {d_v}]')
for bt in [0]:
    for h in [0, 1]:
        for s in range(Sv):
            for n in range(d_v//Sv):
                a = v_per_head[bt, h, s*(d_v//Sv) + n].item()
                b = v_out[bt, h*Sv + s, n].item()
                assert abs(a - b) < 1e-6, f'v reshape mismatch bt={bt} h={h} s={s} n={n}'
print('v  reshape layout verified ✓')

# Backward
loss = (qk_per_head.sum() + v_per_head.sum())
loss.backward()
assert torch.isfinite(x.grad).all()
assert torch.isfinite(qk_lut.weights.grad).all()
assert torch.isfinite(v_lut.weights.grad).all()
print('backward grads finite ✓')

print('\nSMOKE TEST PASSED')
