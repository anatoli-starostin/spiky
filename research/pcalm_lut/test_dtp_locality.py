"""The claims that make this DTP and not something else, asserted rather than asserted-in-prose.

  1. No target carries autograd history -- the top-down signal never passes through a layer Jacobian.
  2. A layer's local loss reaches ONLY that layer's own parameters; every other layer gets exactly zero.
  3. The difference form is what it says: t_{i-1} - h_{i-1} == g(t_i) - g(h_i), exactly.
  4. Its fixed point is right: if t_i == h_i then t_{i-1} == h_{i-1}, exactly (no drift).
  5. The plain-target ablation really does differ from the difference form.
  6. The g objective does not touch f, and the f objective does not touch g.

Run directly (NOT under pytest):  python3 test_dtp_locality.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtp import (PairedMLP, all_f_params, all_g_params, f_losses, f_params, forward_states,  # noqa: E402
                 g_losses, g_params, make_targets)
from paired import PairedLUTStack  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
B, W, L, NT = 32, 64, 4, 32


def build(kind, seed=0):
    if kind == 'lut':
        m = PairedLUTStack(64, 10, width=W, depth=L, n_tables=NT, device=DEV, seed=seed,
                           table_dropout=0.0, clamp_mode='pinned')
    else:
        m = PairedMLP(64, 10, width=W, depth=L, device=DEV, seed=seed)
    m.train()
    x = torch.randn(B, 64, device=DEV)
    y = torch.zeros(B, 10, device=DEV)
    y[torch.arange(B), torch.randint(0, 10, (B,), device=DEV)] = 1.0
    return m, x, y


def check(name, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    assert cond, name


def test_targets_carry_no_history(kind):
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    t = make_targets(m, hs, yhat, y)
    check(f'[{kind}] no target requires grad or has a grad_fn',
          all((not q.requires_grad) and q.grad_fn is None for q in t))


def test_difference_form(kind):
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    t = make_targets(m, hs, yhat, y)
    with torch.no_grad():
        lhs = t[-1] - hs[-1]
        rhs = m.back_readout(y) - m.back_readout(yhat)
        d = float((lhs - rhs).abs().max())
    check(f'[{kind}] readout level: t - h equals g(t_top) - g(yhat) (max diff {d:.2e})', d < 1e-5)
    for i in range(m.n_hidden - 2, -1, -1):
        with torch.no_grad():
            lhs = t[i] - hs[i]
            rhs = m.back_layer(i, t[i + 1]) - m.back_layer(i, hs[i + 1])
            d = float((lhs - rhs).abs().max())
        check(f'[{kind}] level {i}: t - h equals g(t) - g(h) (max diff {d:.2e})', d < 1e-5)


def test_fixed_point(kind):
    """Feed the network's own output back as the top target: every target must collapse onto h."""
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    t = make_targets(m, hs, yhat, yhat)          # t_top = yhat, i.e. "nothing to correct"
    worst = max(float((q - h).abs().max()) for q, h in zip(t, hs))
    check(f'[{kind}] t_top = yhat leaves every target exactly at h (max drift {worst:.2e})', worst < 1e-6)


def test_plain_target_differs(kind):
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    td = make_targets(m, hs, yhat, y, difference=True)
    tp = make_targets(m, hs, yhat, y, difference=False)
    gap = max(float((a - b).abs().max()) for a, b in zip(td, tp))
    check(f'[{kind}] the plain-target ablation really is a different target (max gap {gap:.3e})',
          gap > 1e-6)


def test_locality(kind):
    """Each layer's local loss must move only its own parameters."""
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    t = make_targets(m, hs, yhat, y)
    keys = ['in'] + list(range(m.n_hidden - 1)) + ['out']
    for k in keys:
        m.zero_grad(set_to_none=True)
        f_losses(m, x, hs, t, y)[k].backward()
        own = {id(p) for p in f_params(m, k)}
        leaked = [n for n, p in m.named_parameters()
                  if id(p) not in own and p.grad is not None and float(p.grad.abs().sum()) > 0]
        check(f'[{kind}] f loss for layer {k!r} touches only its own parameters'
              + (f' (leaked into {leaked[:3]})' if leaked else ''), not leaked)
    m.zero_grad(set_to_none=True)


def test_g_and_f_do_not_cross(kind):
    m, x, y = build(kind)
    hs, yhat = forward_states(m, x)
    t = make_targets(m, hs, yhat, y)
    fp, gp = {id(p) for p in all_f_params(m)}, {id(p) for p in all_g_params(m)}

    m.zero_grad(set_to_none=True)
    sum(g_losses(m, hs, yhat, 0.1).values()).backward()
    hit_f = [n for n, p in m.named_parameters()
             if id(p) in fp and p.grad is not None and float(p.grad.abs().sum()) > 0]
    check(f'[{kind}] the g objective puts no gradient on f' + (f' (hit {hit_f[:3]})' if hit_f else ''),
          not hit_f)

    m.zero_grad(set_to_none=True)
    sum(f_losses(m, x, hs, t, y).values()).backward()
    hit_g = [n for n, p in m.named_parameters()
             if id(p) in gp and p.grad is not None and float(p.grad.abs().sum()) > 0]
    check(f'[{kind}] the f objective puts no gradient on g' + (f' (hit {hit_g[:3]})' if hit_g else ''),
          not hit_g)
    m.zero_grad(set_to_none=True)


if __name__ == '__main__':
    print(f'device {DEV}')
    for kind in ('lut', 'mlp'):
        print(f'\n-- {kind}')
        test_targets_carry_no_history(kind)
        test_difference_form(kind)
        test_fixed_point(kind)
        test_plain_target_differs(kind)
        test_locality(kind)
        test_g_and_f_do_not_cross(kind)
    print('\nall DTP locality checks passed')
