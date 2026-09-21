"""GATE 1: reproduce the paper's result on plain residual MLPs -- PC-ALM matches BP, PC lags in deep narrow nets.

Paper setup (App. B, F): residual MLP at the mean-field point, weights ~ N(0,1) with per-layer pre-multipliers,
Adam with eta = eta0 * gamma0^2 * sqrt(N/L), one epoch, batch 64, squared error on one-hot targets,
eta_h = 1/sigma_max(A)^2 estimated per cell, alpha = 1, rho = 1, T = 2L inner steps for PC and PC-ALM.

    python run_mlp_gate1.py --cells 32x8,32x64,8x64 --acts tanh --datasets fashion --seeds 1
Writes runs/gate1_<dataset>.json and prints a table.
"""
import argparse
import json
import os
import time

import torch

from data import TensorLoader, load
from pcalm import ResidualMLP, constraint_sigma_max, run_epochs, squared_error, stability_ok

ap = argparse.ArgumentParser()
ap.add_argument('--cells', default='32x8,32x64,8x64,128x8')
ap.add_argument('--acts', default='tanh')
ap.add_argument('--datasets', default='fashion')
ap.add_argument('--arms', default='bp,pc,pcalm')
ap.add_argument('--seeds', type=int, default=1)
ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--eta0', type=float, default=1e-3)
ap.add_argument('--gamma0', type=float, default=1.0)
ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--rho', type=float, default=1.0)
ap.add_argument('--out-dir', default='runs')
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def accuracy(model, x, y, bs=2000):
    hit = 0
    for i in range(0, x.shape[0], bs):
        hit += int((model(x[i:i + bs]).argmax(-1) == y[i:i + bs].argmax(-1)).sum())
    return hit / x.shape[0]


for ds in a.datasets.split(','):
    xtr, ytr = load(ds, train=True, device=dev)
    xte, yte = load(ds, train=False, device=dev)
    results = []
    for cell in a.cells.split(','):
        N, L = (int(v) for v in cell.split('x'))
        for act in a.acts.split(','):
            for seed in range(a.seeds):
                lr = a.eta0 * a.gamma0 ** 2 * (N / L) ** 0.5
                sig = None
                for arm in a.arms.split(','):
                    torch.manual_seed(seed)
                    model = ResidualMLP(xtr.shape[1], 10, N, L, act=act, gamma0=a.gamma0, device=dev, seed=seed)
                    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=seed)
                    kw = {}
                    if arm != 'bp':
                        if sig is None:                       # same estimate for both inner-loop arms
                            x0, _ = next(iter(loader))
                            sig = constraint_sigma_max(model, x0)
                        eta_h = 1.0 / max(sig ** 2, 1e-12)
                        assert stability_ok(eta_h, sig, a.rho, a.alpha if arm == 'pcalm' else 0.0)
                        kw = dict(T=2 * L, eta_h=eta_h, alpha=a.alpha, rho=a.rho)
                    t0 = time.time()
                    hist, wall = run_epochs(model, loader, arm, epochs=a.epochs, lr=lr, device=dev,
                                            log_every=100, **kw)
                    acc = accuracy(model, xte, yte)
                    row = {'dataset': ds, 'N': N, 'L': L, 'act': act, 'seed': seed, 'arm': arm, 'lr': lr,
                           'sigma_max': sig, 'test_acc': acc, 'final_loss': hist[-1]['loss'],
                           'wall_s': wall, 'hist': hist}
                    results.append(row)
                    print(f'{ds} N={N:3d} L={L:3d} {act:4s} seed{seed} {arm:5s} | test acc {acc:.4f} | '
                          f'final train loss {hist[-1]["loss"]:.4f} | {wall:6.1f}s', flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, f'gate1_{ds}.json')
    old = json.load(open(p)) if os.path.exists(p) else []
    json.dump(old + results, open(p, 'w'), indent=1)
    print('wrote', p)
