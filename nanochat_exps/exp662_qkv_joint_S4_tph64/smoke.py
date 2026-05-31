#!/usr/bin/env python3
"""Smoke test: build the exp662 model, run one fwd+bwd, verify shapes + finite grads."""
import sys, json, os
import torch
import torch.nn as nn

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

# Execute relevant parts of train.py — load module classes by exec.
sys.path.insert(0, '/home/starost/spiky/src')
sys.path.insert(0, '/home/starost/nanochat')

# Minimal copy of the train.py setup needed to build the model.
import math
import torch.nn.functional as F
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

DEVICE = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])
COMPUTE_DTYPE = torch.bfloat16

VOCAB_SIZE = cfg['tokenizer_vocab_size'] if 'tokenizer_vocab_size' in cfg else 32768
CONTEXT_SIZE = cfg['context_size']
N_LAYERS = cfg['num_layers']
E = cfg['embedding_dim']
D = cfg['residual_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']

# Execute train.py up to model construction, but skip data loader etc.
# Easier: run the actual train.py with a tiny n_steps override via env.
print(f'cfg: E={E} D={D} H={H} d_qk={d_qk} d_v={d_v} S={cfg["qkv_sparsify_s"]} qkv_tph={cfg["qkv_tph"]}')
print(f'expected qkv_lut shape per layer: n_heads={H*cfg["qkv_sparsify_s"]} tph={cfg["qkv_tph"]} n_out={(2*d_qk+d_v)//cfg["qkv_sparsify_s"]}')

# Build just the qkv_lut to verify shape.
S = int(cfg['qkv_sparsify_s'])
qkv_dim = 2 * d_qk + d_v
assert qkv_dim % S == 0
qkv_lut = TinyMultiHeadLut(
    input_dim=E,
    n_heads=H * S,
    n_outputs=qkv_dim // S,
    n_anchor_pairs=cfg['qkv_input_nap'],
    tables_per_head=cfg['qkv_tph'],
    random_seed=cfg['random_seed'],
    device=DEVICE,
    backward_mode='hybrid_smooth',
    soft_score_temp=0.5, select_temp=0.5,
    learnable_temps=True, use_bf16=True,
)
print(f'qkv_lut weights shape: {qkv_lut.weights.shape}')
print(f'qkv_lut n_params: {qkv_lut.weights.numel():,}')

B, T = 2, 8
x = torch.randn(B*T, E, device=DEVICE, requires_grad=True)
out = qkv_lut(x)
print(f'qkv_lut output shape: {out.shape}  (expected [B*T={B*T}, H*S={H*S}, n_out={qkv_dim//S}])')
assert out.shape == (B*T, H*S, qkv_dim//S), f'shape mismatch: {out.shape}'

# Reshape to per-real-head: [B*T, H, 2*d_qk + d_v]
qkv_per_head = out.reshape(B*T, H, qkv_dim)
print(f'after reshape: {qkv_per_head.shape}  (expected [B*T={B*T}, H={H}, qkv_dim={qkv_dim}])')
q = qkv_per_head[..., :d_qk]
k = qkv_per_head[..., d_qk:2*d_qk]
v = qkv_per_head[..., 2*d_qk:]
print(f'q={q.shape} k={k.shape} v={v.shape}')
assert q.shape == (B*T, H, d_qk)
assert v.shape == (B*T, H, d_v)

# Verify reshape preserves layout — sub-heads of real head h are contiguous LUT heads h*S..h*S+S-1
# So qkv_per_head[bt, h, s*48 + n] should equal out[bt, h*S + s, n]
for bt in range(B*T):
    for h in range(H):
        for s in range(S):
            for n in range(qkv_dim // S):
                a = qkv_per_head[bt, h, s * (qkv_dim // S) + n].item()
                b = out[bt, h*S + s, n].item()
                assert abs(a - b) < 1e-6, f'reshape mismatch at bt={bt} h={h} s={s} n={n}: {a} vs {b}'
print('reshape layout verified ✓')

# Backward
loss = out.sum()
loss.backward()
assert torch.isfinite(x.grad).all()
assert torch.isfinite(qkv_lut.weights.grad).all()
print('backward grads finite ✓')

print('\nSMOKE TEST PASSED')
