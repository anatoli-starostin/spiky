#!/usr/bin/env python3
"""Smoke test: exp665 — build out_proj + ffn pair, run fwd+bwd, verify shapes,
finite grads, param count (out_proj + ffn = ~37.75M), and inner-residual flow."""
import sys, json, os
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])

E       = cfg['embedding_dim']
H       = cfg['n_heads']
d_v     = cfg['d_v']
N_LAY   = cfg['num_layers']

# out_proj
out_nh   = int(cfg['out_n_heads'])
out_tph  = int(cfg['out_tph'])
out_nap  = int(cfg['out_input_nap'])
out_n_per_head = E // out_nh

# ffn
ffn_nh   = int(cfg['ffn_n_heads'])
ffn_tph  = int(cfg['ffn_tph'])
ffn_nap  = int(cfg['ffn_input_nap'])
ffn_n_per_head = E // ffn_nh

print(f'cfg: E={E} H={H} d_v={d_v} num_layers={N_LAY}')
print(f'out_proj: n_heads={out_nh} tph={out_tph} NAP={out_nap} K={2**out_nap} n_per_head={out_n_per_head} input_dim=H*d_v={H*d_v}')
print(f'ffn:      n_heads={ffn_nh} tph={ffn_tph} NAP={ffn_nap} K={2**ffn_nap} n_per_head={ffn_n_per_head} input_dim=E={E}')

_kwargs = dict(
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)

out_proj = TinyMultiHeadLut(
    input_dim=H * d_v,
    n_heads=out_nh,
    n_outputs=out_n_per_head,
    n_anchor_pairs=out_nap,
    tables_per_head=out_tph,
    random_seed=cfg['random_seed'] + 400,
    device=DEV,
    **_kwargs,
)
ffn = TinyMultiHeadLut(
    input_dim=E,
    n_heads=ffn_nh,
    n_outputs=ffn_n_per_head,
    n_anchor_pairs=ffn_nap,
    tables_per_head=ffn_tph,
    random_seed=cfg['random_seed'] + 600,
    device=DEV,
    **_kwargs,
)

print(f'out_proj weights: {tuple(out_proj.weights.shape)} numel={out_proj.weights.numel():,}')
print(f'ffn      weights: {tuple(ffn.weights.shape)} numel={ffn.weights.numel():,}')

# Per-layer expected: out_n_heads * tph * 2^NAP * n_per_head
out_per_layer = out_nh * out_tph * (2**out_nap) * out_n_per_head
ffn_per_layer = ffn_nh * ffn_tph * (2**ffn_nap) * ffn_n_per_head
print(f'expected out_proj per-layer = {out_per_layer:,}; got {out_proj.weights.numel():,}')
print(f'expected ffn      per-layer = {ffn_per_layer:,}; got {ffn.weights.numel():,}')
assert out_proj.weights.numel() == out_per_layer, 'out_proj per-layer mismatch'
assert ffn.weights.numel() == ffn_per_layer, 'ffn per-layer mismatch'

combined = (out_per_layer + ffn_per_layer) * N_LAY
print(f'combined out_proj + ffn over {N_LAY} layers = {combined:,} ({combined / 1e6:.2f}M)')
expected_combined = 37_748_736
assert combined == expected_combined, f'combined != 37.75M: got {combined:,} expected {expected_combined:,}'
print(f'param-budget check ✓ combined = 37.75M (matches exp664 out_proj-only at tph=256)')

# Forward / backward
B, T = 2, 8
x_flat   = torch.randn(B * T, E, device=DEV, requires_grad=True)
out_in   = torch.randn(B * T, H * d_v, device=DEV, requires_grad=True)
out_e    = out_proj(out_in).reshape(B * T, E)
assert out_e.shape == (B * T, E), f'out_e shape: {out_e.shape}'
ffn_in   = out_e + x_flat
ffn_out  = ffn(ffn_in).reshape(B * T, E)
assert ffn_out.shape == (B * T, E), f'ffn_out shape: {ffn_out.shape}'
block_out = x_flat + ffn_out
print(f'forward: out_e {out_e.shape}, ffn_in {ffn_in.shape}, ffn_out {ffn_out.shape}, block_out {block_out.shape}')

loss = block_out.sum()
loss.backward()
assert torch.isfinite(x_flat.grad).all(),  'x_flat grad has non-finite values'
assert torch.isfinite(out_in.grad).all(),  'out_in grad has non-finite values'
assert torch.isfinite(out_proj.weights.grad).all(), 'out_proj weights grad non-finite'
assert torch.isfinite(ffn.weights.grad).all(),      'ffn weights grad non-finite'
print('backward grads finite for x_flat, out_in, out_proj.weights, ffn.weights ✓')

# Both LUT weights received nontrivial gradient mass
assert ffn.weights.grad.abs().sum().item() > 0, 'ffn grad is zero'
assert out_proj.weights.grad.abs().sum().item() > 0, 'out_proj grad is zero'
print('out_proj and ffn weight grads both nonzero ✓')

print('\nSMOKE TEST PASSED')
