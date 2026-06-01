#!/usr/bin/env python3
"""Smoke test: exp674 — out_proj NAP=8 tph=256 (4× capacity vs exp670)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']
nap=int(cfg['out_input_nap']); tph=int(cfg['out_tph']); nh=int(cfg['out_n_heads'])
assert nap == 8 and tph == 256, f"expected NAP=8 tph=256, got NAP={nap} tph={tph}"
assert nh == 4, f"expected nh=4, got {nh}"
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
assert expected == 25_165_824, f'expected 25.17M (4x exp670 6.29M); got {expected:,}'

bw_per_layer = 2 * nh * tph * n_per_head * 2  # bytes (2-row hybrid_smooth blend, bf16)
print(f'bandwidth: 2 * {nh} * {tph} * {n_per_head} * 2 bytes = {bw_per_layer:,} B/layer ({bw_per_layer/1024:.0f} KB)')
print(f'  exp670 same formula -> 384 KB/layer -> same bandwidth (just more rows per table, only 2 read)')

B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
out = op(x)
assert out.shape == (B*T, nh, n_per_head)
out.sum().backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(op.weights.grad).all()
print('SMOKE TEST PASSED')
