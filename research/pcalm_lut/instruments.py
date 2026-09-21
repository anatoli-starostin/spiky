"""The two instruments the study cares about more than loss curves.

1. `layer_alignment` -- cosine between the weight gradient an arm produces and the BP gradient, PER LAYER.
   For the LUT stack "BP" is itself a surrogate (the score-gate path), so this measures alignment to the
   composed-surrogate gradient; for MLPs it is the true gradient. Depth dependence is the point: BP composes L
   surrogates, PC/PC-ALM apply one per layer per inner step.
2. `inference_flips` -- address flips during one inference phase (gate-0 instrument), returned per layer.
"""
import torch

from pcalm import squared_error


def arm_param_grads(model, x, y, mode, *, T, eta_h, alpha=1.0, rho=1.0, loss_fn=squared_error):
    """Parameter gradients from one arm's inner solve, without stepping the optimiser."""
    model.zero_grad(set_to_none=True)
    if mode == 'bp':
        loss_fn(model(x), y).backward()
        return [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()], None, None
    alpha = 0.0 if mode == 'pc' else alpha
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    lam = [torch.zeros_like(h) for h in hs]
    for _ in range(T):
        e, r = model.energy(x, y, hs, lam, rho, loss_fn)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
            if alpha:
                for li, ri in zip(lam, r):
                    li += alpha * ri.detach()
    e, _ = model.energy(x, y, hs, lam, rho, loss_fn)
    model.zero_grad(set_to_none=True)
    (e / x.shape[0]).backward()
    return [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()], hs, lam


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    n = a.norm() * b.norm()
    return float(torch.dot(a, b) / n) if n > 0 else float('nan')


def layer_alignment(model, x, y, modes, *, T, eta_h, alpha=1.0, rho=1.0):
    """{mode: [cos per parameter tensor]} against BP's gradient on the same batch and weights."""
    names = [n for n, _ in model.named_parameters()]
    ref, _, _ = arm_param_grads(model, x, y, 'bp', T=T, eta_h=eta_h)
    out = {}
    for mode in modes:
        g, _, _ = arm_param_grads(model, x, y, mode, T=T, eta_h=eta_h, alpha=alpha, rho=rho)
        out[mode] = [(n, _cos(gi, ri)) for n, gi, ri in zip(names, g, ref) if gi is not None and ri is not None]
    return out


@torch.no_grad()
def _addr(model, hs):
    return model.addresses(hs) if hasattr(model, 'addresses') else None


def inference_flips(model, x, y, mode, *, T, eta_h, alpha=1.0, rho=1.0, loss_fn=squared_error):
    """Fraction of (sample, table) address slots that differ from the forward pass after the inference phase."""
    alpha = 0.0 if mode == 'pc' else alpha
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    lam = [torch.zeros_like(h) for h in hs]
    a0 = _addr(model, hs)
    if a0 is None:
        return None
    for _ in range(T):
        e, r = model.energy(x, y, hs, lam, rho, loss_fn)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
            if alpha:
                for li, ri in zip(lam, r):
                    li += alpha * ri.detach()
    a1 = _addr(model, hs)
    return [float((p != q).float().mean()) for p, q in zip(a0, a1)]
