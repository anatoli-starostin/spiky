#!/usr/bin/env python3
"""Smoke test: exp693 — exp692 with v_tph 512 -> 256."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 144 and D == 384 and d_v == 24
assert cfg['v_tph'] == 256, f'exp693 requires v_tph=256, got {cfg["v_tph"]}'
assert cfg['out_tph'] == 512, f'exp693 inherits out_tph=512 from exp692, got {cfg["out_tph"]}'

kwargs = dict(backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)

qkv = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                       n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                       random_seed=cfg['random_seed'], device=DEV, **kwargs)
assert qkv.weights.numel() == 3_145_728, qkv.weights.shape    # 6 * 256 * 16 * 128

vl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                      random_seed=cfg['random_seed']+200, device=DEV, **kwargs)
assert vl.weights.numel() == 2_359_296, vl.weights.shape      # 6 * 256 * 64 * 24 (HALF of exp692's 4.72M)

op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                     n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                     random_seed=cfg['random_seed']+400, device=DEV, **kwargs)
assert op.weights.numel() == 4_718_592, op.weights.shape      # 1 * 512 * 64 * 144

rl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                      random_seed=cfg['random_seed']+600, device=DEV, **kwargs)
assert rl.weights.numel() == 3_145_728, rl.weights.shape

lut_total = (qkv.weights.numel() + vl.weights.numel() + op.weights.numel() + rl.weights.numel()) * N_L
print(f'LUT total over {N_L} layers: {lut_total:,} ({lut_total/1e6:.2f}M)')
print(f'  qkv={qkv.weights.numel()*N_L/1e6:.2f}M  v={vl.weights.numel()*N_L/1e6:.2f}M  '
      f'op={op.weights.numel()*N_L/1e6:.2f}M  residual={rl.weights.numel()*N_L/1e6:.2f}M')
print(f'  delta vs exp692 (v was 4.72M): -{(4_718_592 - vl.weights.numel())*N_L/1e6:.2f}M')

bw_per_layer = (
    2 * H * cfg['qkv_tph'] * 2*d_qk * 2 +
    2 * H * cfg['v_tph']   * d_v    * 2 +
    2 * 1 * cfg['out_tph'] * E      * 2 +
    2 * 1 * cfg['residual_tph'] * D * 2
)
print(f'bandwidth per layer: {bw_per_layer/1024:.0f} KB; × {N_L} layers = {bw_per_layer*N_L/1024/1024:.2f} MB/token')

# Sanity fwd/bwd
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
q = qkv(x); assert q.shape == (B*T, H, 2*d_qk)
v = vl(x);  assert v.shape == (B*T, H, d_v)
xa = torch.randn(B*T, H*d_v, device=DEV, requires_grad=True)
o = op(xa); assert o.shape == (B*T, 1, E)
xr = torch.randn(B*T, E, device=DEV, requires_grad=True)
r = rl(xr); assert r.shape == (B*T, 1, D)
(q.sum() + v.sum() + o.sum() + r.sum()).backward()
for mod in [qkv, vl, op, rl]:
    assert torch.isfinite(mod.weights.grad).all()
print('SMOKE TEST PASSED')
