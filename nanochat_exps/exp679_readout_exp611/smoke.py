#!/usr/bin/env python3
"""Smoke test: exp679 — exp670 + read_out at exp611 settings (nh=1, NAP=5, tph=1536)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']

# out_proj (unchanged from exp670)
op_nh=int(cfg['out_n_heads']); op_tph=int(cfg['out_tph']); op_nap=int(cfg['out_input_nap'])
assert op_nh == 4 and op_tph == 256 and op_nap == 6
op_per = E // op_nh
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=op_nh, n_outputs=op_per,
                     n_anchor_pairs=op_nap, tables_per_head=op_tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
print(f'out_proj: nh={op_nh} tph={op_tph} NAP={op_nap} K={2**op_nap} n_per_head={op_per} params={op.weights.numel():,}')
assert op.weights.numel() == 6_291_456

# read_out at exp611 settings
ro_nh=int(cfg['readout_n_heads']); ro_tph=int(cfg['readout_tph']); ro_nap=int(cfg['readout_input_nap'])
assert ro_nh == 1 and ro_tph == 1536 and ro_nap == 5, f"expected nh=1 tph=1536 NAP=5, got nh={ro_nh} tph={ro_tph} NAP={ro_nap}"
ro_per = E // ro_nh
ro = TinyMultiHeadLut(input_dim=E, n_heads=ro_nh, n_outputs=ro_per,
                     n_anchor_pairs=ro_nap, tables_per_head=ro_tph,
                     random_seed=cfg['random_seed']+800, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
print(f'read_out: nh={ro_nh} tph={ro_tph} NAP={ro_nap} K={2**ro_nap} n_per_head={ro_per} params={ro.weights.numel():,}')
assert ro.weights.numel() == 18_874_368, f'expected 18.87M (= exp611 read_out); got {ro.weights.numel():,}'

print(f'\nTotals: out_proj 6.29M × 6 layers = 37.75M; read_out one-time = 18.87M')
print(f'expected total model: 86.5M + 18.87M - 6.29M (no exp666-style read_out) = wait, base is exp670 with NO read_out = 86.5M; +18.87M = 105.37M')

# Sanity forward / backward
B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
o = op(x); assert o.shape == (B*T, op_nh, op_per)
o.sum().backward()
assert torch.isfinite(op.weights.grad).all()

xr = torch.randn(B*T, E, device=DEV, requires_grad=True)
r = ro(xr); assert r.shape == (B*T, ro_nh, ro_per)
r.sum().backward()
assert torch.isfinite(ro.weights.grad).all()

print('SMOKE TEST PASSED')
