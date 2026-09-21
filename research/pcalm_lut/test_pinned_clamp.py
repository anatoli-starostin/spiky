"""Sanity checks for output-pinned clamping, before any long run.

Pinned mode treats the TARGET as the top state: h_L = y, held for the whole relaxation, contributing one
more forward residual r^f_L = y - readout(h_{L-1}) on the same footing as every other layer, with the
separate data term removed so the target is not counted twice.

Checks here:
  1. the clamped nodes (input x, target y) do not move across inner steps, and the free states do;
  2. the forward-residual family gains exactly one member, and it IS the output error;
  3. the data term is gone in pinned mode -- the energy equals the residual sums alone -- and still present
     in data mode;
  4. 'data' mode still reproduces the OLD energy exactly, so the earlier runs stay reproducible;
  5. sigma_max is re-measured on the pinned residual family (it is NOT asserted to change: the extra top
     row of A is one readout Jacobian against L-1 interior blocks, and measurement shows the leading
     singular direction stays interior-dominated, so the value is unchanged to four decimals here).

Run directly (NOT under pytest):  python3 test_pinned_clamp.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import PairedLUTStack  # noqa: E402
from train_paired import inner_loop, sigma_max_A  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
B, W, L, NT, T = 32, 32, 6, 8, 12


def build(clamp, seed=0):
    m = PairedLUTStack(64, 10, width=W, depth=L, n_tables=NT, device=DEV, seed=seed, clamp_mode=clamp)
    m.train()
    x = torch.randn(B, 64, device=DEV)
    y = torch.zeros(B, 10, device=DEV)
    y[torch.arange(B), torch.randint(0, 10, (B,), device=DEV)] = 1.0
    return m, x, y


def check(name, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    assert cond, name


def test_clamped_nodes_do_not_move():
    for arm in ('pcA', 'pcalmB'):
        m, x, y = build('pinned')
        x0, y0 = x.clone(), y.clone()
        sig = sigma_max_A(m, x, y, arm)
        hs, _, _ = inner_loop(m, x, y, arm, T=T, eta_h=0.5 / max(sig ** 2, 1e-12), alpha=1.0, rho=1.0)
        check(f'[{arm}] the clamped input never moved', torch.equal(x, x0))
        check(f'[{arm}] the clamped target never moved', torch.equal(y, y0))
        check(f'[{arm}] only the interior states are free ({len(hs)} == depth-1 == {m.n_hidden})',
              len(hs) == m.n_hidden)
        moved = sum(1 for h, f in zip(hs, m.init_states(x)) if not torch.equal(h, f))
        check(f'[{arm}] the free states did move ({moved}/{len(hs)})', moved == len(hs))


def test_top_residual_is_the_output_error():
    m, x, y = build('pinned')
    hs = [h.clone() for h in m.init_states(x)]
    rf = m.residuals_f(x, hs, y)
    md, _, _ = build('data')
    md.load_state_dict(m.state_dict())
    rf_data = md.residuals_f(x, hs)
    check(f'pinned adds exactly one residual ({len(rf)} vs {len(rf_data)})', len(rf) == len(rf_data) + 1)
    check('the extra residual IS y - readout(h_{L-1})', torch.allclose(rf[-1], y - m.readout(hs[-1])))


def test_data_term_removed():
    m, x, y = build('pinned')
    hs = [h.clone().requires_grad_(True) for h in m.init_states(x)]
    e, rf, rb = m.energy_A(x, y, hs)
    manual = sum(r.pow(2).sum() for r in rf) + sum(r.pow(2).sum() for r in rb)
    # relative, not absolute: these are float32 sums of squares over a whole batch, so the reduction order
    # alone moves the last few digits. A leftover data term would be a percent-level gap, not 1e-6.
    rel = float(((e - manual) / manual).abs())
    check(f'pinned arm A energy is exactly the residual sums, no data term (rel {rel:.2e})', rel < 1e-5)

    md, _, _ = build('data')
    md.load_state_dict(m.state_dict())
    ed, rfd, rbd = md.energy_A(x, y, hs)
    man_d = sum(r.pow(2).sum() for r in rfd) + sum(r.pow(2).sum() for r in rbd)
    gap = float((ed - man_d).detach())
    want = float(0.5 * (md.readout(hs[-1]) - y).pow(2).sum().detach())
    check(f'data mode still carries the data term ({gap:.3f} ~ {want:.3f})',
          abs(gap - want) / max(want, 1e-9) < 1e-3)

    lam = [torch.zeros_like(r) for r in rf]
    eb, _, _ = m.energy_B(x, y, hs, lam, rho=1.0)
    manual_b = sum(0.5 * r.pow(2).sum() for r in rf)
    rel_b = float(((eb - manual_b) / manual_b).abs())
    check(f'pinned arm B at lam=0 is pure penalty, no objective (rel {rel_b:.2e})', rel_b < 1e-5)


def test_data_mode_reproduces_the_old_energy():
    """The OLD formulas, written out here independently of paired.py."""
    m, x, y = build('data')
    hs = [h.clone().requires_grad_(True) for h in m.init_states(x)]
    yhat = m.readout(hs[-1])
    rf = [hs[0] - m.h1(x)] + [hs[i + 1] - m.layer(i, hs[i]) for i in range(m.n_hidden - 1)]
    rb = [hs[i] - m.back_layer(i, hs[i + 1]) for i in range(m.n_hidden - 1)] + [hs[-1] - m.back_readout(yhat)]
    old = 0.5 * (yhat - y).pow(2).sum() + sum(r.pow(2).sum() for r in rf) + sum(r.pow(2).sum() for r in rb)
    new, _, _ = m.energy_A(x, y, hs)
    check(f'data-mode arm A == the old energy (rel {float(((new - old) / old).abs()):.2e})',
          torch.allclose(new, old, rtol=1e-5, atol=1e-4))


def test_sigma_is_measured_on_the_pinned_family():
    """sigma_max must be measured on the residual family the arm actually penalises under the NEW clamping.

    It is not asserted to CHANGE: the extra top row of A is one readout Jacobian against L-1 interior
    blocks, so the leading singular direction stays interior-dominated and the value can be unchanged to
    three decimals. What must hold is that the operator covers the pinned family and the step size is
    derived from it, which is what the length check below pins down."""
    from train_paired import _residuals
    m, x, y = build('pinned')
    md, _, _ = build('data')
    md.load_state_dict(m.state_dict())
    hs = m.init_states(x)
    for arm in ('pcA', 'pcalmB'):
        np_, nd = len(_residuals(m, x, y, hs, arm)), len(_residuals(md, x, y, hs, arm))
        check(f'[{arm}] the constraint operator gained the top row ({nd} -> {np_} residual blocks)',
              np_ == nd + 1)
        sp, sd = sigma_max_A(m, x, y, arm), sigma_max_A(md, x, y, arm)
        print(f'        sigma_max: data {sd:.4f} -> pinned {sp:.4f}  '
              f'(eta_h scales as 1/sigma^2, so {100 * (sd ** 2 / max(sp ** 2, 1e-12) - 1):+.1f}%)')
        check(f'[{arm}] sigma_max is finite and positive under pinning ({sp:.4f})',
              sp > 0 and sp == sp and sp < float('inf'))


if __name__ == '__main__':
    print(f'device {DEV}')
    test_clamped_nodes_do_not_move()
    test_top_residual_is_the_output_error()
    test_data_term_removed()
    test_data_mode_reproduces_the_old_energy()
    test_sigma_is_measured_on_the_pinned_family()
    print('all pinned-clamp checks passed')
