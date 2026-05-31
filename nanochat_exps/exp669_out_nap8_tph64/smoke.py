#!/usr/bin/env python3
"""Smoke test: exp669 — out_proj at NAP=8, tph=64 (param-matched to exp664's
NAP=6, tph=256). Verify shape, param count = 6.29M/layer = 37.75M total."""
import sys, json, os
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])

E    = cfg['embedding_dim']
H    = cfg['n_heads']
d_v  = cfg['d_v']
nap  = int(cfg['out_input_nap'])
tph  = int(cfg['out_tph'])
nh   = int(cfg['out_n_heads'])
N_L  = cfg['num_layers']

assert nap == 8 and tph == 64, f"expected NAP=8 tph=64, got NAP={nap} tph={tph}"

n_per_head = E // nh

print(f'cfg: E={E} H={H} d_v={d_v} nh={nh} NAP={nap} K={2**nap} tph={tph} n_per_head={n_per_head}')

out_proj = TinyMultiHeadLut(
    input_dim=H * d_v,
    n_heads=nh,
    n_outputs=n_per_head,
    n_anchor_pairs=nap,
    tables_per_head=tph,
    random_seed=cfg['random_seed'] + 400,
    device=DEV,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
print(f'out_proj weights: {tuple(out_proj.weights.shape)} numel={out_proj.weights.numel():,}')

# Expected per-layer: nh * tph * 2^NAP * n_per_head
expected = nh * tph * (2**nap) * n_per_head
print(f'expected per-layer = {expected:,} ({expected/1e6:.2f}M)')
assert out_proj.weights.numel() == expected
assert expected == 6_291_456, f'expected 6.29M (=exp664); got {expected:,}'

total = expected * N_L
print(f'total out_proj over {N_L} layers = {total:,} ({total/1e6:.2f}M)  [== exp664]')

# Bandwidth per token (2-row hybrid_smooth blend, bf16):
bytes_per_token_layer = 2 * nh * tph * n_per_head * 2
print(f'bandwidth: 2 * {nh} * {tph} * {n_per_head} * 2 bytes = {bytes_per_token_layer:,} B/layer ({bytes_per_token_layer/1024:.0f} KB)')
print(f'  vs exp664 (NAP=6 tph=256): 2 * 6 * 256 * 64 * 2 = 393,216 B/layer (384 KB)')
print(f'  → 4x bandwidth reduction')

# Forward / backward
B, T = 2, 8
x = torch.randn(B * T, H * d_v, device=DEV, requires_grad=True)
out = out_proj(x)
print(f'out shape: {out.shape}  expected: [{B*T}, {nh}, {n_per_head}]')
assert out.shape == (B * T, nh, n_per_head)

loss = out.sum()
loss.backward()
assert torch.isfinite(x.grad).all(), 'x grad non-finite'
assert torch.isfinite(out_proj.weights.grad).all(), 'out_proj weights grad non-finite'
assert out_proj.weights.grad.abs().sum().item() > 0, 'out_proj grad zero'
print('backward grads finite, out_proj grad nonzero ✓')

print('\nSMOKE TEST PASSED')
