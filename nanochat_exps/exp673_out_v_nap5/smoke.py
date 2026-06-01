#!/usr/bin/env python3
"""Smoke test: exp673 — out_proj NAP=5 + v_lut NAP=5 (K=32 in both). NOT param-matched."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']; d_qk=cfg['d_qk']
out_nap=int(cfg['out_input_nap']); out_tph=int(cfg['out_tph']); out_nh=int(cfg['out_n_heads'])
v_nap=int(cfg['v_input_nap']); v_tph=int(cfg['v_tph']); v_S=int(cfg['v_sparsify_s'])

assert out_nap == 5, f"expected out_input_nap=5, got {out_nap}"
assert v_nap == 5, f"expected v_input_nap=5, got {v_nap}"

n_per_head_out = E // out_nh
print(f'out_proj: nh={out_nh} NAP={out_nap} K={2**out_nap} tph={out_tph} n_per_head={n_per_head_out}')
print(f'v_lut:    n_heads={H*v_S} NAP={v_nap} K={2**v_nap} tph={v_tph} n_out={d_v//v_S}')

# Build out_proj at NAP=5
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=out_nh, n_outputs=n_per_head_out,
                     n_anchor_pairs=out_nap, tables_per_head=out_tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth',
                     soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
op_expected = out_nh * out_tph * (2**out_nap) * n_per_head_out
print(f'out_proj weights: {tuple(op.weights.shape)} numel={op.weights.numel():,} expected={op_expected:,}')
assert op.weights.numel() == op_expected
assert op_expected == 3_145_728, f'expected 3.15M (half of exp670 6.29M); got {op_expected:,}'

# Build v_lut at NAP=5
vl = TinyMultiHeadLut(input_dim=E, n_heads=H*v_S, n_outputs=d_v//v_S,
                     n_anchor_pairs=v_nap, tables_per_head=v_tph,
                     random_seed=cfg['random_seed']+200, device=DEV,
                     backward_mode='hybrid_smooth',
                     soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
v_expected = H * v_S * v_tph * (2**v_nap) * (d_v//v_S)
print(f'v_lut weights:   {tuple(vl.weights.shape)} numel={vl.weights.numel():,} expected={v_expected:,}')
assert vl.weights.numel() == v_expected
assert v_expected == 1_572_864, f'expected 1.57M (half of exp670 3.15M); got {v_expected:,}'

# Total LUT delta over 6 layers
N_L = cfg['num_layers']
op_delta = (6_291_456 - op_expected) * N_L
v_delta  = (3_145_728 - v_expected)  * N_L
total_delta = op_delta + v_delta
print(f'param savings over {N_L} layers: out_proj=-{op_delta:,}  v_lut=-{v_delta:,}  total=-{total_delta:,} ({-total_delta/1e6:.2f}M)')

# Forward / backward
B,T=2,8
x = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
o = op(x)
assert o.shape == (B*T, out_nh, n_per_head_out)
o.sum().backward()
assert torch.isfinite(x.grad).all() and torch.isfinite(op.weights.grad).all()

xv = torch.randn(B*T, E, device=DEV, requires_grad=True)
ov = vl(xv)
assert ov.shape == (B*T, H*v_S, d_v//v_S)
ov.sum().backward()
assert torch.isfinite(xv.grad).all() and torch.isfinite(vl.weights.grad).all()

print('SMOKE TEST PASSED')
