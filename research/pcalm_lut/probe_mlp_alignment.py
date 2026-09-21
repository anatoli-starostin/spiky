"""Is the interior anti-alignment PC-generic, or something the LUT architecture causes?

The LUT stack shows interior cosine-to-BP of -0.5 to -0.7 with the readout at +0.95. This runs the SAME
measurement on the plain residual MLP of the gate-1 reproduction, matched to the LUT debug config as far
as an MLP allows: depth 4, width 64, Fashion-MNIST, seed 0, 500 steps, T=8, output pinned, no dropout,
the same eta_h derivation from sigma_max(A).

If the MLP interior is also strongly negative, the anti-alignment is a property of predictive coding's
target assignment and has nothing to do with LUTs. If the MLP interior is positive, the LUT architecture
is making PC's per-layer targets pathological.

The second question this answers: does the MLP's data loss actually MOVE while the cosine is negative?
The LUT arms are stuck at ~0.489 against a trivial 0.5. PC learning fine while disagreeing with BP would
mean cosine-to-BP is the wrong yardstick.

Usage: python3 probe_mlp_alignment.py [--depth 4] [--steps 500]
"""
import argparse
import json
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from instruments import layer_alignment  # noqa: E402
from pcalm import ResidualMLP, constraint_sigma_max, squared_error, train_step  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def split_cos(pairs, n_hidden):
    """Separate the per-parameter cosines into readout vs interior. W1 is the input map, WL the readout."""
    readout = [c for n, c in pairs if n.startswith('WL')]
    interior = [c for n, c in pairs if n.startswith('Wi')]
    inp = [c for n, c in pairs if n.startswith('W1')]
    return interior, readout, inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--clamp', default='pinned', choices=['pinned', 'data'])
    ap.add_argument('--at', default='1,100,250,500')
    ap.add_argument('--out', default='runs_debug/mlp_alignment.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr, ytr = load('fashion', train=True, device=dev)
    xte, yte = load('fashion', train=False, device=dev)
    want = sorted(int(v) for v in a.at.split(','))
    out = {'cfg': vars(a), 'arms': {}}

    for mode in ('pc', 'pcalm'):
        torch.manual_seed(a.seed)
        model = ResidualMLP(xtr.shape[1], 10, a.width, a.depth, device=dev, seed=a.seed)
        model.clamp_mode = a.clamp
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
        px, py = xtr[:a.batch], ytr[:a.batch]
        sigma = constraint_sigma_max(model, px)
        eta_h = 1.0 / max(sigma ** 2, 1e-12)
        rows, it, step = [], iter(loader), 0
        print(f'\n=== {mode}  depth {a.depth} width {a.width} T {a.T} clamp {a.clamp} '
              f'sigma_max {sigma:.4f} eta_h {eta_h:.4e}')
        while step <= max(want):
            if step in want:
                al = layer_alignment(model, px, py, [mode], T=a.T, eta_h=eta_h)[mode]
                inter, read, inp = split_cos(al, model.n_hidden)
                with torch.no_grad():
                    pred = model(xte[:2000])
                    acc = float((pred.argmax(-1) == yte[:2000].argmax(-1)).float().mean())
                    dl = float(squared_error(model(px), py))
                rows.append({'step': step, 'interior': inter, 'readout': read, 'input': inp,
                             'interior_mean': st.mean(inter) if inter else float('nan'),
                             'readout_mean': st.mean(read) if read else float('nan'),
                             'input_mean': st.mean(inp) if inp else float('nan'),
                             'data_loss': dl, 'test_acc': acc})
                r = rows[-1]
                print(f'  step {step:>4d}  interior {r["interior_mean"]:+.3f}  readout {r["readout_mean"]:+.3f}'
                      f'  input {r["input_mean"]:+.3f}  data loss {dl:.4f}  test acc {acc:.4f}'
                      f'   per-layer interior ' + ' '.join(f'{c:+.3f}' for c in inter))
            try:
                bx, by = next(it)
            except StopIteration:
                it = iter(loader)
                bx, by = next(it)
            train_step(model, bx, by, opt, mode, T=a.T, eta_h=eta_h, grad_clip=1.0)
            step += 1
        out['arms'][mode] = rows

    print('\n== ANSWER')
    for mode, rows in out['arms'].items():
        f, l = rows[0], rows[-1]
        print(f'   {mode:6s} interior cosine {f["interior_mean"]:+.3f} -> {l["interior_mean"]:+.3f}, '
              f'readout {f["readout_mean"]:+.3f} -> {l["readout_mean"]:+.3f}, '
              f'data loss {f["data_loss"]:.4f} -> {l["data_loss"]:.4f}, '
              f'test acc {f["test_acc"]:.4f} -> {l["test_acc"]:.4f}')

    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
