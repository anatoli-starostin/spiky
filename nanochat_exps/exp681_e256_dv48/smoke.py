#!/usr/bin/env python3
"""Smoke test: exp681 — exp611 architecture with E=256, d_v=48, D=384."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 256 and D == 384 and d_v == 48, f"expected E=256 D=384 d_v=48, got E={E} D={D} d_v={d_v}"

kwargs = dict(backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)

# qkv_lut (= qk, n_outputs = 2*d_qk)
qkv = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2*d_qk,
                       n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                       random_seed=cfg['random_seed'], device=DEV, **kwargs)
print(f'qkv_lut: input={E} n_heads={H} n_out={2*d_qk} NAP={cfg["qkv_input_nap"]} K={2**cfg["qkv_input_nap"]} tph={cfg["qkv_tph"]} params={qkv.weights.numel():,}')
assert qkv.weights.numel() == 3_145_728, f'expected 3.15M; got {qkv.weights.numel():,}'

# v_lut
vl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                      random_seed=cfg['random_seed']+200, device=DEV, **kwargs)
print(f'v_lut:   input={E} n_heads={H} n_out={d_v} NAP={cfg["v_input_nap"]} K={2**cfg["v_input_nap"]} tph={cfg["v_tph"]} params={vl.weights.numel():,}')
assert vl.weights.numel() == 9_437_184, f'expected 9.44M; got {vl.weights.numel():,}'

# out_proj
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                      n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                      random_seed=cfg['random_seed']+400, device=DEV, **kwargs)
print(f'out_proj: input={H*d_v} n_heads=1 n_out={E} NAP={cfg["out_input_nap"]} K={2**cfg["out_input_nap"]} tph={cfg["out_tph"]} params={op.weights.numel():,}')
assert op.weights.numel() == 16_777_216, f'expected 16.78M; got {op.weights.numel():,}'

# read_out
ro = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['readout_input_nap'], tables_per_head=cfg['readout_tph'],
                      random_seed=cfg['random_seed']+800, device=DEV, **kwargs)
print(f'read_out: input={E} n_heads=1 n_out={D} NAP={cfg["readout_input_nap"]} K={2**cfg["readout_input_nap"]} tph={cfg["readout_tph"]} params={ro.weights.numel():,}')
assert ro.weights.numel() == 18_874_368, f'expected 18.87M; got {ro.weights.numel():,}'

# Totals
lut_total = (qkv.weights.numel() + vl.weights.numel() + op.weights.numel()) * N_L + ro.weights.numel()
print(f'\nLUT total over {N_L} layers + 1 read_out: {lut_total:,} ({lut_total/1e6:.2f}M)')
print(f'  qkv:    {qkv.weights.numel()*N_L/1e6:.2f}M')
print(f'  v:      {vl.weights.numel()*N_L/1e6:.2f}M')
print(f'  op:     {op.weights.numel()*N_L/1e6:.2f}M')
print(f'  ro:     {ro.weights.numel()/1e6:.2f}M (once)')

# Quick fwd/bwd
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
q = qkv(x); assert q.shape == (B*T, H, 2*d_qk)
v = vl(x);  assert v.shape == (B*T, H, d_v)
xa = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
o = op(xa); assert o.shape == (B*T, 1, E)
xr = torch.randn(B*T, E, device=DEV, requires_grad=True)
r = ro(xr); assert r.shape == (B*T, 1, D)
(q.sum() + v.sum() + o.sum() + r.sum()).backward()
for mod in [qkv, vl, op, ro]:
    assert torch.isfinite(mod.weights.grad).all()
print('SMOKE TEST PASSED')
