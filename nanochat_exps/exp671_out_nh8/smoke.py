#!/usr/bin/env python3
"""Smoke test: exp671 — out_proj n_heads=8 (n_outputs/head=48), param-matched."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']
nap=int(cfg['out_input_nap']); tph=int(cfg['out_tph']); nh=int(cfg['out_n_heads'])
assert nh == 8, f"expected out_n_heads=8, got {nh}"
assert E % nh == 0, f"E={E} not divisible by nh={nh}"
n_per_head = E // nh
print(f'out_proj: nh={nh} NAP={nap} K={2**nap} tph={tph} n_per_head={n_per_head}')

op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=nh, n_outputs=n_per_head,
                     n_anchor_pairs=nap, tables_per_head=tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth',
                     soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
expected = nh*tph*(2**nap)*n_per_head
print(f'out_proj weights: {tuple(op.weights.shape)} numel={op.weights.numel():,} expected={expected:,}')
assert op.weights.numel() == expected
assert expected == 6_291_456, f'expected 6.29M (= exp664); got {expected:,}'

B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
out = op(x)
assert out.shape == (B*T, nh, n_per_head)
print(f'out shape {out.shape}; reshape -> {out.reshape(B*T, E).shape}')
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(op.weights.grad).all()
assert op.weights.grad.abs().sum() > 0
print('SMOKE TEST PASSED')
