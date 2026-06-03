#!/usr/bin/env python3
"""Smoke test: exp706 — exp696 with E=96, d_v=16 (H*d_v=96=E preserved)."""
import sys, json, os, torch
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

erl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['emb_resid_input_nap'], tables_per_head=cfg['emb_resid_tph'],
                      random_seed=cfg['random_seed']+800, device=DEV, **kwargs)
assert erl.weights.numel() == 3_145_728, erl.weights.shape
print(f'emb_resid_lut: input={E} n_heads=1 n_out={D} NAP={cfg["emb_resid_input_nap"]} tph={cfg["emb_resid_tph"]} params={erl.weights.numel():,}')

qkv = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk,
                       n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
                       random_seed=cfg['random_seed'], device=DEV, **kwargs)
assert qkv.weights.numel() == 3_145_728   # 6 * 256 * 16 * 128 (unchanged)
vl = TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
                      n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
                      random_seed=cfg['random_seed']+200, device=DEV, **kwargs)
assert vl.weights.numel() == 1_572_864   # 6 * 256 * 64 * 16 (d_v 24->16)
op = TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
                     n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
                     random_seed=cfg['random_seed']+400, device=DEV, **kwargs)
assert op.weights.numel() == 3_145_728   # 1 * 512 * 64 * 96 (E 144->96)
rl = TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=D,
                      n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
                      random_seed=cfg['random_seed']+600, device=DEV, **kwargs)
assert rl.weights.numel() == 3_145_728

lut_block_total = (qkv.weights.numel() + vl.weights.numel() + op.weights.numel() + rl.weights.numel()) * N_L
lut_total = lut_block_total + erl.weights.numel()
print(f'LUT total: {lut_total/1e6:.2f}M  (block={lut_block_total/1e6:.2f}M × {N_L} layers + emb_resid={erl.weights.numel()/1e6:.2f}M)')
print(f'  delta vs exp696 LUT: -14.15M (v_lut -0.79M/layer, out_proj -1.57M/layer)')

# Sanity fwd/bwd
B, T = 2, 8
x = torch.randn(B*T, E, device=DEV, requires_grad=True)
e = erl(x); assert e.shape == (B*T, 1, D), e.shape
e_reshape = e.reshape(B, T, D)
q = qkv(x); v = vl(x)
xa = torch.randn(B*T, H*d_v, device=DEV)
o = op(xa); r = rl(x)
(e.sum() + q.sum() + v.sum() + o.sum() + r.sum()).backward()
for mod in [erl, qkv, vl, op, rl]:
    assert torch.isfinite(mod.weights.grad).all()
print('SMOKE TEST PASSED')
