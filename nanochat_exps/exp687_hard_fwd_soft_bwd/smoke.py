#!/usr/bin/env python3
"""Smoke test: exp687 — exp684 forked to backward_mode='soft' (hard fwd + soft bwd)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 144 and D == 384 and d_v == 24

kwargs = dict(backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)
assert cfg['backward_mode'] == 'soft', f'config backward_mode must be soft, got {cfg["backward_mode"]!r}'

rl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                      random_seed=cfg['random_seed']+600, device=DEV, **kwargs)
print(f'residual_lut: input={E} n_heads=1 n_out={D} NAP={cfg["residual_input_nap"]} K={2**cfg["residual_input_nap"]} tph={cfg["residual_tph"]} params={rl.weights.numel():,}')
assert rl.weights.numel() == 3_145_728, f'expected 3.15M per layer; got {rl.weights.numel():,}'

total_residual = rl.weights.numel() * N_L
print(f'\nresidual_lut total over {N_L} layers: {total_residual:,} ({total_residual/1e6:.2f}M)')
assert total_residual == 18_874_368, f'expected 18.87M total (=exp683 single read_out); got {total_residual:,}'

bw_per_layer = 2 * 1 * cfg['residual_tph'] * D * 2
bw_total = bw_per_layer * N_L
print(f'bandwidth: per layer {bw_per_layer:,} B ({bw_per_layer/1024:.0f} KB); × {N_L} layers = {bw_total:,} B ({bw_total/1024/1024:.2f} MB)')
print(f'  vs exp683 single read_out (NAP=5 tph=1536): 2*1*1536*384*2 = 2,359,296 B = 2.25 MB')
print(f'  same total bandwidth, just distributed across layers')

B, T = 2, 8
xr = torch.randn(B*T, E, device=DEV, requires_grad=True)
r = rl(xr); assert r.shape == (B*T, 1, D)
r.sum().backward()
assert torch.isfinite(rl.weights.grad).all()
print('SMOKE TEST PASSED')
