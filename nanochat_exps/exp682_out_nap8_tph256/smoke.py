#!/usr/bin/env python3
"""Smoke test: exp682 — exp681 with out_proj NAP=8 tph=256 (param-matched)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']
nap=int(cfg['out_input_nap']); tph=int(cfg['out_tph'])
assert nap == 8 and tph == 256, f"expected NAP=8 tph=256, got NAP={nap} tph={tph}"

op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                     n_anchor_pairs=nap, tables_per_head=tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
print(f'out_proj: nh=1 NAP={nap} K={2**nap} tph={tph} n_out=E={E} params={op.weights.numel():,}')
assert op.weights.numel() == 16_777_216, f'expected 16.78M (= exp681 same params); got {op.weights.numel():,}'

# Bandwidth check
bw_per_layer = 2 * 1 * tph * E * 2  # bytes
print(f'bandwidth: 2 * 1 * {tph} * {E} * 2 bytes = {bw_per_layer:,} B/layer ({bw_per_layer/1024:.0f} KB)')
print(f'  vs exp681 (NAP=6 tph=1024): 2 * 1 * 1024 * 256 * 2 = 1,048,576 B/layer (1024 KB)')
print(f'  -> 4x bandwidth reduction')

B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
out = op(x); assert out.shape == (B*T, 1, E)
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(op.weights.grad).all()
print('SMOKE TEST PASSED')
