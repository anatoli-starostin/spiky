"""WHY is the PC-A weight update anti-aligned with backprop in almost every layer?

At L=16 the per-layer cosine of arm A's update to the BP gradient is negative in 11-14 of 16 forward
layers, in all three seeds, from the very first step. Since it is already at full strength at step 1, the
cause can be studied at INITIALISATION -- no training run needed.

Arm A's energy has three parts (DERIVATION_v2 section 2):

    E = 1/2||yhat - y||^2  +  sum_i ||r^f_i||^2  +  sum_i ||r^b_i||^2
        \_____ data _____/    \___ forward ___/     \__ backward __/

BP's gradient is the gradient of the data term alone, taken at the FEEDFORWARD states. This script takes
the relaxed states h* that arm A actually uses, then differentiates each of the three parts separately
with respect to the weights, and reports each part's per-layer cosine to the BP gradient. Whichever part
carries the negative cosine is the mechanism.

It also reports the same decomposition at the FEEDFORWARD states (h = forward pass, where r^f = 0 by
construction), which separates "the relaxation moved the states somewhere unhelpful" from "the extra
energy terms pull the weights the wrong way wherever you evaluate them".

Usage: python3 probe_cosine_decomposition.py [--depth 16] [--seed 0] [--T 32]
"""
import argparse
import json
import os
import sys

import torch
import torch._functorch.config as _ft_config

# this probe takes several gradients through ONE graph (one per energy part), which the compiled
# backward refuses while it may donate its buffers. Diagnostic only; the trainer never sets this.
_ft_config.donated_buffer = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from train_paired import cos, inner_loop, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def grads_of(model, scalar):
    """d scalar / d (forward LUT tables), as a dict layer -> tensor. Never touches .grad."""
    names, params = [], []
    for i, l in enumerate(model.f_lut):
        names.append(f'L{i}')
        params.append(l.tables)
    names.append(f'L{len(model.f_lut)}')            # the readout is the last forward layer
    params.append(model.f_out.tables)
    g = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return {n: (gi if gi is not None else torch.zeros_like(p)) for n, p, gi in zip(names, params, g)}


def parts_at(model, x, y, hs):
    """The three energy parts of arm A evaluated at the given states, each as a scalar with graph."""
    yhat = model.readout(hs[-1])
    rf = model.residuals_f(x, hs)
    rb = model.residuals_b(hs, yhat)
    data = 0.5 * (yhat - y).pow(2).sum()
    fwd = sum(ri.pow(2).sum() for ri in rf)
    bwd = sum(ri.pow(2).sum() for ri in rb)
    return {'data': data, 'forward': fwd, 'backward': bwd, 'total': data + fwd + bwd}


def report(tag, ref, parts_g, names):
    """A part that does not reach a layer at all has gradient EXACTLY zero there (the data term reaches
    only the readout once the states are free variables); that prints as '.' rather than a nan cosine."""
    print(f'\n  {tag}')
    print(f'   {"part":10s} ' + ' '.join(f'{n:>6s}' for n in names) + '   mean   #neg')
    out = {}
    for part, gd in parts_g.items():
        c = [cos(gd[n], ref[n]) for n in names]
        live = [v for v in c if v == v]
        print(f'   {part:10s} ' + ' '.join((f'{v:>+6.2f}' if v == v else f'{".":>6s}') for v in c)
              + (f'  {sum(live) / len(live):>+5.2f}' if live else f'  {"-":>5s}')
              + f' {sum(1 for v in live if v < 0):>5d}')
        out[part] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=16)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=32)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eta-frac', type=float, default=0.5)
    ap.add_argument('--train-steps', type=int, default=50,
                    help='train arm A this many steps FIRST. At initialisation the decomposition is '
                         'degenerate (r^f = 0 at the feedforward point, so the interior gradient is '
                         'exactly zero) and one Adam step already moves every table entry by ~lr, which '
                         'is larger than the entries themselves -- the effect lives after a few steps.')
    ap.add_argument('--out', default='runs/cosine_decomposition.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    xtr, ytr = load('fashion', train=True, device=dev)
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    x, y = next(iter(loader))

    if a.train_steps:
        from train_paired import arm_grads
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sg = sigma_max_A(model, x, 'pcA')
        it = iter(loader)
        for s in range(a.train_steps):
            try:
                bx, by = next(it)
            except StopIteration:
                it = iter(loader)
                bx, by = next(it)
            arm_grads(model, bx, by, 'pcA', T=a.T, eta_h=a.eta_frac / max(sg ** 2, 1e-12), alpha=1.0, rho=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.zero_grad(set_to_none=True)
        print(f'warmed up {a.train_steps} arm-A steps before decomposing')

    # --- the reference: BP's gradient at the feedforward states
    ref = grads_of(model, 0.5 * (model(x) - y).pow(2).sum())
    names = list(ref)
    print(f'L={a.depth} N={a.width} tables={a.tables} T={a.T} seed={a.seed} | reference = BP gradient')

    out = {'cfg': vars(a)}
    # --- at the FEEDFORWARD states (r^f = 0 there by construction)
    hs0 = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    out['feedforward'] = report('evaluated at the FEEDFORWARD states', ref,
                                {k: grads_of(model, v) for k, v in parts_at(model, x, y, hs0).items()}, names)

    # --- at the RELAXED states arm A actually uses
    sigma = sigma_max_A(model, x, 'pcA')
    eta_h = a.eta_frac / max(sigma ** 2, 1e-12)
    hs, _, _ = inner_loop(model, x, y, 'pcA', T=a.T, eta_h=eta_h, alpha=1.0, rho=1.0)
    hs = [h.detach().clone().requires_grad_(True) for h in hs]
    print(f'\n  sigma_max {sigma:.3f}  eta_h {eta_h:.4f}')
    out['relaxed'] = report('evaluated at the RELAXED states h* (what arm A differentiates)', ref,
                            {k: grads_of(model, v) for k, v in parts_at(model, x, y, hs).items()}, names)

    # --- how much of the total gradient NORM each part carries, per layer (a part can be anti-aligned
    #     and still irrelevant if it is tiny)
    gp = {k: grads_of(model, v) for k, v in parts_at(model, x, y, hs).items()}
    print('\n  share of the total gradient norm, per layer (relaxed states)')
    print(f'   {"part":10s} ' + ' '.join(f'{n:>6s}' for n in names))
    for part in ('data', 'forward', 'backward'):
        sh = [float(gp[part][n].norm() / max(float(gp['total'][n].norm()), 1e-30)) for n in names]
        print(f'   {part:10s} ' + ' '.join(f'{v:>6.2f}' for v in sh))
        out.setdefault('norm_share', {})[part] = sh

    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
