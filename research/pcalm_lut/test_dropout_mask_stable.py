"""The dropout masks must not move inside one weight update.

The PC arms run T inner forward passes over the SAME batch and take a vjp through them. If the mask were
resampled per forward call (the library default), the energy would be non-stationary -- the relaxation
would chase a moving target -- and the vjp would differentiate a different sub-network than the one that
made the prediction. These tests assert the pinned mask holds across calls, that it is genuinely random
between updates, that the unpinned default really does move (so the tests are not vacuous), and that no
no_grad / eval path sees dropout at all.

Run directly (NOT under pytest -- this directory has no fixtures):  python3 test_dropout_mask_stable.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import PairedLUTStack  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
B, W, L, NT = 16, 32, 5, 8


def build(table_p=0.0, residual_p=0.0, seed=0):
    m = PairedLUTStack(64, 10, width=W, depth=L, n_tables=NT, device=DEV, seed=seed,
                       table_dropout=table_p, residual_dropout=residual_p)
    m.train()
    return m, torch.randn(B, 64, device=DEV)


def check(name, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    assert cond, name


def test_pinned_mask_is_stable(table_p, residual_p, tag):
    """T forward passes inside one update must be bit-identical, and the vjp must match the prediction."""
    m, x = build(table_p, residual_p)
    m.resample_dropout(B)
    outs = [m(x) for _ in range(8)]                       # stands in for the T inner steps
    check(f'[{tag}] 8 forwards in one update are bit-identical',
          all(torch.equal(outs[0], o) for o in outs[1:]))

    # the mask the vjp differentiates through is the same OBJECT the prediction used
    masks = [l._head_drop_mask for l in m.luts()]
    m(x)
    check(f'[{tag}] the pinned mask tensors survive a forward unchanged',
          all(a is b for a, b in zip(masks, [l._head_drop_mask for l in m.luts()])))

    # the vjp taken through a SECOND forward equals the one through the first. NOT bit-equality: the
    # embedding_bag backward reduces with atomics and is non-deterministic at ~1e-13 absolute, with
    # dropout off exactly as much as on (measured), so the floor below is that, not the mask moving.
    p = m.f_lut[0].tables
    g1 = torch.autograd.grad(m(x).square().sum(), p)[0]
    g2 = torch.autograd.grad(m(x).square().sum(), p)[0]
    d = float((g1 - g2).abs().max())
    check(f'[{tag}] the vjp is the same through two separate forwards (max diff {d:.2e} < 1e-11)',
          d < 1e-11)

    # ... and a resample must actually change something, or dropout is not happening at all
    before = m(x)
    m.resample_dropout(B)
    after = m(x)
    check(f'[{tag}] a new update draws a different mask', not torch.equal(before, after))


def test_unpinned_default_moves():
    """The library default (no pin) resamples per call -- exactly the behaviour we must not use."""
    m, x = build(table_p=0.25)
    m.clear_dropout()
    check('[default] unpinned table dropout resamples per call (so pinning is load-bearing)',
          not torch.equal(m(x), m(x)))


def test_drop_rate_matches_p():
    m, _ = build(table_p=0.25)
    m.resample_dropout(4096)
    mask = m.f_lut[0]._head_drop_mask
    dropped = float((mask == 0).float().mean())
    scale = float(mask[mask > 0].mean())
    check(f'table-dropout rate {dropped:.4f} ~ 0.25', abs(dropped - 0.25) < 0.02)
    check(f'survivors scaled by {scale:.4f} ~ 1/(1-p) = 1.3333', abs(scale - 1 / 0.75) < 1e-5)


def test_no_grad_and_eval_see_the_full_network():
    m, x = build(table_p=0.25, residual_p=0.25)
    m.resample_dropout(B)
    with torch.no_grad():
        a = m(x)
        m.resample_dropout(B)
        b = m(x)
    check('no_grad forward is unaffected by the mask (dropout is train+grad only)', torch.equal(a, b))
    m.eval()
    c = m(x)
    m.resample_dropout(B)
    d = m(x)
    check('eval() forward is unaffected by the mask', torch.equal(c, d))


def test_off_by_default():
    m, x = build()
    m.resample_dropout(B)
    check('p=0 is an exact no-op (masks stay None)', m.f_lut[0]._head_drop_mask is None
          and torch.equal(m(x), m(x)))


if __name__ == '__main__':
    print(f'device {DEV}')
    test_off_by_default()
    test_pinned_mask_is_stable(0.25, 0.0, 'table')
    test_pinned_mask_is_stable(0.0, 0.25, 'residual')
    test_pinned_mask_is_stable(0.25, 0.25, 'both')
    test_unpinned_default_moves()
    test_drop_rate_matches_p()
    test_no_grad_and_eval_see_the_full_network()
    print('all dropout mask-stability tests passed')
