#!/usr/bin/env python3
"""Smoke test: exp676 — exp675 with out_proj reduced via nh=4, tph=256 (same total tables)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']
nh=int(cfg['out_n_heads']); tph=int(cfg['out_tph']); nap=int(cfg['out_input_nap'])
n_per = E // nh
print(f'out_proj: nh={nh} tph={tph} NAP={nap} n_per_head={n_per} (total tables = nh*tph = {nh*tph})')

op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=nh, n_outputs=n_per,
                     n_anchor_pairs=nap, tables_per_head=tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
expected = nh*tph*(2**nap)*n_per
print(f'out_proj weights: {tuple(op.weights.shape)} numel={op.weights.numel():,} expected={expected:,}')
assert op.weights.numel() == expected
assert expected == 6_291_456, f'expected 6.29M (1/4 of exp675 out_proj 25.17M); got {expected:,}'

# Same total tables check
exp675_total_tables = 1 * 1024
assert nh * tph == exp675_total_tables, f'total tables {nh*tph} != exp675 {exp675_total_tables}'

B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
out = op(x); assert out.shape == (B*T, nh, n_per)
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(op.weights.grad).all()
print(f'SMOKE TEST PASSED  (total tables preserved at {nh*tph})')
