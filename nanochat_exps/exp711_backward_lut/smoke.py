#!/usr/bin/env python3
"""Smoke test: exp711 — exp706 + per-block backward_lut with detached MSE aux."""
import sys, json, os, torch
import torch.nn.functional as F
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 96 and D == 384 and d_v == 16 and H * d_v == E

kwargs = dict(backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)

bwd = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=E,
                       n_anchor_pairs=cfg['backward_input_nap'], tables_per_head=cfg['backward_tph'],
                       random_seed=cfg['random_seed']+900, device=DEV, **kwargs)
assert bwd.weights.numel() == 786_432, bwd.weights.shape   # 1 * 256 * 32 * 96
print(f'backward_lut: input={E} n_heads=1 n_out={E} NAP={cfg["backward_input_nap"]} tph={cfg["backward_tph"]} params={bwd.weights.numel():,}')

# Sanity fwd/bwd including detached MSE
B, T = 2, 8
x_in  = torch.randn(B*T, E, device=DEV, requires_grad=True)
x_out = x_in + 0.1 * torch.randn(B*T, E, device=DEV)
# pseudo block-output, treat as actual output
bwd_pred = bwd(x_out).squeeze(1)
mse = F.mse_loss(bwd_pred, x_in.detach())
mse.backward()
assert torch.isfinite(bwd.weights.grad).all()
print(f'sanity MSE={mse.item():.4f}, grad_norm={bwd.weights.grad.norm().item():.4e}')
print(f'block param delta vs exp706: +{6 * bwd.weights.numel() / 1e6:.2f}M (backward_lut x 6 layers)')
print(f'expected total params ~89.66M (exp706 84.94M + 4.72M backward_lut)')
print('SMOKE TEST PASSED')
