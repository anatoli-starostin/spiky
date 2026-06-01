#!/usr/bin/env python3
"""Smoke test: exp695 — exp693 with a pre_lut after tok_emb_E."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 144 and D == 384 and d_v == 24
assert H * d_v == E, 'pre_lut requires H*d_v = E (reshape to E-stream)'
assert cfg['pre_input_nap'] == 6, 'pre_lut NAP should match v_lut'
assert cfg['pre_tph'] == 256, 'pre_lut tph should match v_lut'

kwargs = dict(backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)

pl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['pre_input_nap'], tables_per_head=cfg['pre_tph'],
                      random_seed=cfg['random_seed']+800, device=DEV, **kwargs)
assert pl.weights.numel() == 2_359_296, pl.weights.shape   # 6 * 256 * 64 * 24
print(f'pre_lut: input={E} n_heads={H} n_out={d_v} NAP={cfg["pre_input_nap"]} tph={cfg["pre_tph"]} params={pl.weights.numel():,}')

qkv = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                       n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                       random_seed=cfg['random_seed'], device=DEV, **kwargs)
assert qkv.weights.numel() == 3_145_728
vl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                      random_seed=cfg['random_seed']+200, device=DEV, **kwargs)
assert vl.weights.numel() == 2_359_296
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                     n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                     random_seed=cfg['random_seed']+400, device=DEV, **kwargs)
assert op.weights.numel() == 4_718_592
rl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                      random_seed=cfg['random_seed']+600, device=DEV, **kwargs)
assert rl.weights.numel() == 3_145_728

lut_block_total = (qkv.weights.numel() + vl.weights.numel() + op.weights.numel() + rl.weights.numel()) * N_L
lut_pre = pl.weights.numel()
lut_total = lut_block_total + lut_pre
print(f'LUT total: {lut_total/1e6:.2f}M  (block={lut_block_total/1e6:.2f}M × {N_L} layers + pre={lut_pre/1e6:.2f}M)')
print(f'  delta vs exp693: +{lut_pre/1e6:.2f}M (pre_lut)')

bw_per_layer = (
    2 * H * cfg['qkv_tph'] * 2*d_qk * 2 +
    2 * H * cfg['v_tph']   * d_v    * 2 +
    2 * 1 * cfg['out_tph'] * E      * 2 +
    2 * 1 * cfg['residual_tph'] * D * 2
)
bw_pre = 2 * H * cfg['pre_tph'] * d_v * 2
print(f'bandwidth per layer: {bw_per_layer/1024:.0f} KB; × {N_L} layers + pre {bw_pre/1024:.0f} KB = {(bw_per_layer*N_L+bw_pre)/1024/1024:.2f} MB/token')

# Sanity fwd/bwd
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
p = pl(x); assert p.shape == (B*T, H, d_v), p.shape
p_reshape = p.reshape(B, T, E)
q = qkv(x); v = vl(x)
xa = torch.randn(B*T, H*d_v, device=DEV)
o = op(xa); r = rl(x)
(p.sum() + q.sum() + v.sum() + o.sum() + r.sum()).backward()
for mod in [pl, qkv, vl, op, rl]:
    assert torch.isfinite(mod.weights.grad).all()
print('SMOKE TEST PASSED')
