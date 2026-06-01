#!/usr/bin/env python3
"""Smoke test: exp680 — exp679 with v_lut at exp611 settings (nh=6 S=1, NAP=6, tph=512)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; H=cfg['n_heads']; d_v=cfg['d_v']

# v_lut — exp611 settings (no S, n_heads=H=6, tph=512, NAP=6, n_outputs=64)
v_S=int(cfg['v_sparsify_s']); v_tph=int(cfg['v_tph']); v_nap=int(cfg['v_input_nap'])
assert v_S == 1 and v_tph == 512 and v_nap == 6, f"expected v S=1 tph=512 NAP=6, got S={v_S} tph={v_tph} NAP={v_nap}"
v_lut = TinyMultiHeadLut(input_dim=E, n_heads=H*v_S, n_outputs=d_v//v_S,
                         n_anchor_pairs=v_nap, tables_per_head=v_tph,
                         random_seed=cfg['random_seed']+200, device=DEV,
                         backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                         learnable_temps=True, use_bf16=True)
print(f'v_lut: n_heads={H*v_S} tph={v_tph} NAP={v_nap} K={2**v_nap} n_out={d_v//v_S} params={v_lut.weights.numel():,}')
assert v_lut.weights.numel() == 12_582_912, f'expected 12.58M; got {v_lut.weights.numel():,}'

# read_out (exp611 settings, unchanged from exp679)
ro_nh=int(cfg['readout_n_heads']); ro_tph=int(cfg['readout_tph']); ro_nap=int(cfg['readout_input_nap'])
assert ro_nh == 1 and ro_tph == 1536 and ro_nap == 5
ro = TinyMultiHeadLut(input_dim=E, n_heads=ro_nh, n_outputs=E//ro_nh,
                     n_anchor_pairs=ro_nap, tables_per_head=ro_tph,
                     random_seed=cfg['random_seed']+800, device=DEV,
                     backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
                     learnable_temps=True, use_bf16=True)
print(f'read_out: nh={ro_nh} tph={ro_tph} NAP={ro_nap} K={2**ro_nap} n_per_head={E//ro_nh} params={ro.weights.numel():,}')

# Sanity
B,T=2,8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
v = v_lut(x); assert v.shape == (B*T, H*v_S, d_v//v_S)
v.sum().backward()
assert torch.isfinite(v_lut.weights.grad).all()

print(f'\nv_lut over 6 layers: {12_582_912*6/1e6:.1f}M')
print('SMOKE TEST PASSED')
