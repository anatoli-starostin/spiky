"""Correctness checks for the PC / PC-ALM implementation (no data needed).

The decisive one is Proposition 1/3 of the paper: in a LINEAR network at fixed weights, PC-ALM's inner loop
converges to the KKT point, where the multipliers are the BP adjoints and the weight gradient of L_rho IS the
BP gradient. So with enough inner steps, PC-ALM's parameter gradient must match autograd's to high precision,
while PC's (lambda pinned to 0) must NOT.

    python test_pcalm.py
"""
import torch

from pcalm import ResidualMLP, squared_error, constraint_sigma_max, stability_ok

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def grads(model, x, y, mode, T, eta_h, alpha=1.0, rho=1.0):
    """Parameter gradient produced by one PC / PC-ALM inner solve (no optimiser step)."""
    alpha = 0.0 if mode == 'pc' else alpha
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    lam = [torch.zeros_like(h) for h in hs]
    for _ in range(T):
        e, r = model.energy(x, y, hs, lam, rho, squared_error)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
            if alpha:
                for li, ri in zip(lam, r):
                    li += alpha * ri.detach()
    e, _ = model.energy(x, y, hs, lam, rho, squared_error)
    model.zero_grad(set_to_none=True)
    e.backward()
    return [p.grad.detach().clone() for p in model.parameters()], hs, lam


def bp_grads(model, x, y):
    model.zero_grad(set_to_none=True)
    squared_error(model(x), y).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def cos(a, b):
    fa = torch.cat([t.flatten() for t in a]).double()
    fb = torch.cat([t.flatten() for t in b]).double()
    return float(torch.dot(fa, fb) / (fa.norm() * fb.norm()))


def main():
    torch.manual_seed(0)
    for depth in (4, 8, 16):
        m = ResidualMLP(20, 5, width=16, depth=depth, act='id', device=DEV, seed=1).double()
        x = torch.randn(32, 20, device=DEV, dtype=torch.float64)
        y = torch.randn(32, 5, device=DEV, dtype=torch.float64)
        sig = constraint_sigma_max(m, x)
        eta = 0.5 / max(sig ** 2, 1e-12)
        assert stability_ok(eta, sig, rho=1.0, alpha=1.0), (eta, sig)
        g_bp = bp_grads(m, x, y)
        g_alm, hs, lam = grads(m, x, y, 'pcalm', T=4000, eta_h=eta)
        g_pc, _, _ = grads(m, x, y, 'pc', T=4000, eta_h=eta)
        # at the KKT point the residuals vanish and lambda is the BP adjoint
        r = m.residuals(x, hs)
        resid = max(float(ri.abs().max()) for ri in r)
        print(f'L={depth:3d} sigma_max {sig:8.4f} eta_h {eta:9.3e} | cos(PC-ALM, BP) {cos(g_alm, g_bp):.8f} | '
              f'cos(PC, BP) {cos(g_pc, g_bp):.6f} | max|r| at convergence {resid:.2e}')
        assert cos(g_alm, g_bp) > 0.9999, 'PC-ALM must recover the BP gradient in a linear net'
        assert resid < 1e-6, 'activations must return to the forward pass at convergence'
    print('OK: linear-net KKT identity holds (PC-ALM == BP), PC differs as expected')


if __name__ == '__main__':
    main()
