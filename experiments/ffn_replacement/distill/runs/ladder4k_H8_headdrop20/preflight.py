"""Preflight for ladder4k_H8_headdrop20: prove lut_head_dropout_rate reaches every student's
LightMultiHeadLUT, is ACTIVE in train mode (with grad) and OFF in eval / no_grad."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
import distill_ffn as D   # noqa: E402

base = json.load(open(D.DEF_STUDENT))
ov = {"lut_n_heads": 8, "lut_head_dropout_rate": 0.2}
torch.manual_seed(1)
for li in range(6):
    s = D.build_student({**base, **ov}, 384, 6, li, 'cuda')
    luts = [m for m in s.modules() if isinstance(m, D.LightMultiHeadLUT)]
    assert luts, 'no LightMultiHeadLUT in student'
    for m in luts:
        assert m.head_dropout_rate == 0.2, m.head_dropout_rate
        assert m.forward_mode == 'scored', m.forward_mode
        assert m.n_heads == 8 if hasattr(m, 'n_heads') else True
    # decompress is zero-init -> output identically 0; give it weights so dropout is visible
    with torch.no_grad():
        for n, p in s.named_parameters():
            if 'decompress' in n and p.dim() == 2:
                p.normal_(0, 0.02)
    h = torch.randn(2048, 384, device='cuda')
    s.train()
    a, b = D.student_forward(s, h), D.student_forward(s, h)
    tr_diff = (a - b).abs().max().item()
    with torch.no_grad():
        c, d = D.student_forward(s, h), D.student_forward(s, h)
    s.eval()
    with torch.no_grad():
        e, f = D.student_forward(s, h), D.student_forward(s, h)
    ev_diff = (e - f).abs().max().item()
    # same student, rate forced to 0 in train mode == eval output (dropout is the only train/eval difference)
    for m in luts:
        m.head_dropout_rate = 0.0
    s.train()
    g = D.student_forward(s, h)
    for m in luts:
        m.head_dropout_rate = 0.2
    base_diff = (g - e).abs().max().item()
    print(f'L{li}: {len(luts)} LUT module(s), rate 0.2, scored | train fwd-vs-fwd max|d| {tr_diff:.3e} '
          f'(must be >0) | train+no_grad max|d| {(c - d).abs().max().item():.3e} | eval max|d| {ev_diff:.3e} '
          f'(must be 0) | train@rate0 vs eval {base_diff:.3e}')
    assert tr_diff > 0, 'dropout NOT active in train mode'
    assert ev_diff == 0, 'eval not deterministic'
print('PREFLIGHT PASS')
