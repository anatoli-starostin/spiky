#!/usr/bin/env python3
"""Smoke test: exp675 — exp611 architecture (qk/v/out_proj) minus read_out."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']

# qk
qk_S = int(cfg['qk_sparsify_s']); qk_tph = int(cfg['qk_tph']); qk_nap = int(cfg['qk_input_nap'])
assert qk_S == 1, f"expected qk_sparsify_s=1, got {qk_S}"
qk_lut = TinyMultiHeadLut(input_dim=E, n_heads=H*qk_S, n_outputs=(2*d_qk)//qk_S,
                          n_anchor_pairs=qk_nap, tables_per_head=qk_tph,
                          random_seed=cfg['random_seed'], device=DEV,
                          backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                          learnable_temps=True, use_bf16=True)
qk_exp = H*qk_S * qk_tph * (2**qk_nap) * ((2*d_qk)//qk_S)
print(f'qk_lut: n_heads={H*qk_S} tph={qk_tph} NAP={qk_nap} K={2**qk_nap} n_out={(2*d_qk)//qk_S}')
print(f'        weights={tuple(qk_lut.weights.shape)} numel={qk_lut.weights.numel():,} expected={qk_exp:,}')
assert qk_lut.weights.numel() == qk_exp
assert qk_exp == 3_145_728, f'expected 3.15M; got {qk_exp:,}'

# v
v_S = int(cfg['v_sparsify_s']); v_tph = int(cfg['v_tph']); v_nap = int(cfg['v_input_nap'])
assert v_S == 1, f"expected v_sparsify_s=1, got {v_S}"
v_lut = TinyMultiHeadLut(input_dim=E, n_heads=H*v_S, n_outputs=d_v//v_S,
                         n_anchor_pairs=v_nap, tables_per_head=v_tph,
                         random_seed=cfg['random_seed']+200, device=DEV,
                         backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                         learnable_temps=True, use_bf16=True)
v_exp = H*v_S * v_tph * (2**v_nap) * (d_v//v_S)
print(f'v_lut:  n_heads={H*v_S} tph={v_tph} NAP={v_nap} K={2**v_nap} n_out={d_v//v_S}')
print(f'        weights={tuple(v_lut.weights.shape)} numel={v_lut.weights.numel():,} expected={v_exp:,}')
assert v_lut.weights.numel() == v_exp
assert v_exp == 12_582_912, f'expected 12.58M; got {v_exp:,}'

# out_proj
nh = int(cfg['out_n_heads']); o_tph = int(cfg['out_tph']); o_nap = int(cfg['out_input_nap'])
assert nh == 1, f"expected out_n_heads=1, got {nh}"
n_per = E // nh
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=nh, n_outputs=n_per,
                     n_anchor_pairs=o_nap, tables_per_head=o_tph,
                     random_seed=cfg['random_seed']+400, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
op_exp = nh*o_tph*(2**o_nap)*n_per
print(f'out_proj: nh={nh} tph={o_tph} NAP={o_nap} K={2**o_nap} n_out={n_per}')
print(f'          weights={tuple(op.weights.shape)} numel={op.weights.numel():,} expected={op_exp:,}')
assert op.weights.numel() == op_exp
assert op_exp == 25_165_824, f'expected 25.17M; got {op_exp:,}'

lut_total = (qk_exp + v_exp + op_exp) * N_L
print(f'\nLUT total over {N_L} layers: {lut_total:,} ({lut_total/1e6:.1f}M)')
print(f'  qk: {qk_exp*N_L/1e6:.1f}M  v: {v_exp*N_L/1e6:.1f}M  out_proj: {op_exp*N_L/1e6:.1f}M')

# Sanity forward / backward
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
qk_out = qk_lut(x); assert qk_out.shape == (B*T, H*qk_S, (2*d_qk)//qk_S)
v_out = v_lut(x);   assert v_out.shape == (B*T, H*v_S, d_v//v_S)
xv = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
o = op(xv); assert o.shape == (B*T, nh, n_per)
(qk_out.sum() + v_out.sum() + o.sum()).backward()
assert torch.isfinite(qk_lut.weights.grad).all() and torch.isfinite(v_lut.weights.grad).all() and torch.isfinite(op.weights.grad).all()
print('SMOKE TEST PASSED')
