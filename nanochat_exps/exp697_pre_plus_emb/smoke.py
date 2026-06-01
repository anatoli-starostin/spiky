#!/usr/bin/env python3
"""Smoke test: exp697 — exp696 + pre_lut (compose emb_resid_lut and pre_lut)."""
import sys, json, os, torch
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f: cfg = json.load(f)
sys.path.insert(0, '/home/starost/spiky/src')
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut

DEV = torch.device('cuda:0'); torch.manual_seed(cfg['random_seed'])
E=cfg['embedding_dim']; D=cfg['residual_dim']; H=cfg['n_heads']; d_qk=cfg['d_qk']; d_v=cfg['d_v']
N_L = cfg['num_layers']
assert E == 144 and D == 384 and d_v == 24
assert H * d_v == E

kwargs = dict(backward_mode='hybrid_smooth', soft_score_temp=0.5, select_temp=0.5,
              learnable_temps=True, use_bf16=True)

erl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
                      random_seed=cfg['random_seed']+800, device=DEV, **kwargs)
assert erl.weights.numel() == 3_145_728   # emb_resid_lut

pl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['pre_input_nap'], tables_per_head=cfg['pre_tph'],
                      random_seed=cfg['random_seed']+900, device=DEV, **kwargs)
assert pl.weights.numel() == 2_359_296    # pre_lut

qkv = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                       n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                       random_seed=cfg['random_seed'], device=DEV, **kwargs)
vl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                      random_seed=cfg['random_seed']+200, device=DEV, **kwargs)
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                     n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                     random_seed=cfg['random_seed']+400, device=DEV, **kwargs)
rl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                      random_seed=cfg['random_seed']+600, device=DEV, **kwargs)

lut_block_total = (qkv.weights.numel() + vl.weights.numel() + op.weights.numel() + rl.weights.numel()) * N_L
lut_total = lut_block_total + erl.weights.numel() + pl.weights.numel()
print(f'LUT total: {lut_total/1e6:.2f}M')
print(f'  blocks  ({N_L}×): {lut_block_total/1e6:.2f}M')
print(f'  emb_resid_lut:    {erl.weights.numel()/1e6:.2f}M')
print(f'  pre_lut:          {pl.weights.numel()/1e6:.2f}M')
print(f'  delta vs exp696: +{pl.weights.numel()/1e6:.2f}M (pre_lut added)')

# Sanity fwd/bwd
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
e = erl(x); assert e.shape == (B*T, 1, D)
p = pl(x); assert p.shape == (B*T, H, d_v)
q = qkv(x); v = vl(x)
xa = torch.randn(B*T, H*d_v, device=DEV); o = op(xa); r = rl(x)
(e.sum() + p.sum() + q.sum() + v.sum() + o.sum() + r.sum()).backward()
for mod in [erl, pl, qkv, vl, op, rl]:
    assert torch.isfinite(mod.weights.grad).all()
print('SMOKE TEST PASSED')
