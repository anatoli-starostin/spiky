"""How many inner steps does the PC-ALM inner solve need to reach the KKT point (linear net, fixed weights)?"""
import sys

import torch

from pcalm import ResidualMLP, squared_error, constraint_sigma_max
from test_pcalm import bp_grads, cos

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
eta_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

torch.manual_seed(0)
m = ResidualMLP(20, 5, width=16, depth=depth, act='id', device=DEV, seed=1).double()
x = torch.randn(32, 20, device=DEV, dtype=torch.float64)
y = torch.randn(32, 5, device=DEV, dtype=torch.float64)
sig = constraint_sigma_max(m, x)
eta = eta_frac / max(sig ** 2, 1e-12)
g_bp = bp_grads(m, x, y)
hs = [h.detach().clone().requires_grad_(True) for h in m.init_states(x)]
lam = [torch.zeros_like(h) for h in hs]
print(f'L={depth} sigma_max={sig:.4f} eta_h={eta:.3e} alpha={alpha} rho=1')
for t in range(1, 200001):
    e, r = m.energy(x, y, hs, lam, 1.0, squared_error)
    g = torch.autograd.grad(e, hs)
    with torch.no_grad():
        for h, gi in zip(hs, g):
            h -= eta * gi
        for li, ri in zip(lam, r):
            li += alpha * ri.detach()
    if t in (10, 100, 1000, 4000, 16000, 64000, 200000):
        e2, r2 = m.energy(x, y, hs, lam, 1.0, squared_error)
        m.zero_grad(set_to_none=True)
        e2.backward()
        g_alm = [p.grad.detach().clone() for p in m.parameters()]
        with torch.no_grad():
            resid = max(float(ri.abs().max()) for ri in r2)
            dual = max(float(li.abs().max()) for li in lam)
        print(f'  t={t:7d} max|r| {resid:.3e} max|lam| {dual:.3e} cos(PC-ALM,BP) {cos(g_alm, g_bp):.8f}')
        hs = [h.detach().clone().requires_grad_(True) for h in hs]
