"""GATE 1 (truncated): the deep-narrow Fashion-MNIST cell (N=32, L=64) at a fraction of an epoch.

Strict prefix of the full sweep: same model/init/seed/batch order/hyper-parameters, only fewer steps. Primary
instrument is per-layer cosine alignment of each arm's weight gradient to BP's, measured on a FIXED probe batch
at several checkpoints; loss/accuracy are secondary.

    python run_gate1_fast.py [--steps 250] [--width 32 --depth 64]
"""
import argparse
import json
import os
import time

import torch

from data import TensorLoader, load
from instruments import layer_alignment
from pcalm import ResidualMLP, constraint_sigma_max, run_epochs, squared_error, stability_ok

ap = argparse.ArgumentParser()
ap.add_argument('--width', type=int, default=32)
ap.add_argument('--depth', type=int, default=64)
ap.add_argument('--steps', type=int, default=250)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--dataset', default='fashion')
ap.add_argument('--act', default='tanh')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--eta0', type=float, default=1e-3)
ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--rho', type=float, default=1.0)
ap.add_argument('--checkpoints', default='0,50,150,250')
ap.add_argument('--out', default='runs/gate1_fast.json')
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
N, L = a.width, a.depth
lr = a.eta0 * (N / L) ** 0.5
xtr, ytr = load(a.dataset, train=True, device=dev)
xte, yte = load(a.dataset, train=False, device=dev)
probe_x, probe_y = xtr[:64], ytr[:64]                      # fixed batch for the alignment instrument
ck = [int(v) for v in a.checkpoints.split(',')]


@torch.no_grad()
def accuracy(model, x, y, bs=2000):
    return sum(int((model(x[i:i + bs]).argmax(-1) == y[i:i + bs].argmax(-1)).sum())
               for i in range(0, x.shape[0], bs)) / x.shape[0]


def fresh():
    torch.manual_seed(a.seed)
    return ResidualMLP(xtr.shape[1], 10, N, L, act=a.act, device=dev, seed=a.seed)


m0 = fresh()
xb, _ = next(iter(TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)))
sigma = constraint_sigma_max(m0, xb)
eta_h = 1.0 / max(sigma ** 2, 1e-12)
assert stability_ok(eta_h, sigma, a.rho, a.alpha)
T = 2 * L
epoch_steps = xtr.shape[0] // a.batch
print(f'{a.dataset} N={N} L={L} {a.act} seed{a.seed} | lr {lr:.2e} | sigma_max {sigma:.4f} eta_h {eta_h:.3e} '
      f'| T={T} alpha={a.alpha} rho={a.rho} | budget {a.steps}/{epoch_steps} steps = '
      f'{a.steps / epoch_steps:.3f} epoch ({a.steps * a.batch:,} samples)', flush=True)

report = {'args': vars(a), 'sigma_max': sigma, 'eta_h': eta_h, 'T': T, 'lr': lr,
          'epoch_steps': epoch_steps, 'arms': {}, 'alignment': {}}

# --- alignment at each checkpoint: train a BP model to that step, then compare arms' gradients there --------
for c in ck:
    m = fresh()
    if c:
        run_epochs(m, TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed), 'bp', lr=lr, device=dev,
                   max_steps=c, log_every=10 ** 9)
    al = layer_alignment(m, probe_x, probe_y, ('pc', 'pcalm'), T=T, eta_h=eta_h, alpha=a.alpha, rho=a.rho)
    report['alignment'][c] = al
    for mode, rows in al.items():
        inter = [v for n, v in rows if n.startswith('Wi.')]
        print(f'  alignment @step {c:4d} {mode:5s}: readout {dict(rows).get("WL", float("nan")):+.4f} | '
              f'interior mean {sum(inter) / len(inter):+.4f} | first {inter[0]:+.4f} last {inter[-1]:+.4f} | '
              f'W1 {dict(rows).get("W1", float("nan")):+.4f}', flush=True)

# --- the three arms at the truncated budget -----------------------------------------------------------------
for arm in ('bp', 'pc', 'pcalm'):
    m = fresh()
    kw = {} if arm == 'bp' else dict(T=T, eta_h=eta_h, alpha=(0.0 if arm == 'pc' else a.alpha), rho=a.rho)
    t0 = time.time()
    hist, wall = run_epochs(m, TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed), arm, lr=lr, device=dev,
                            log_every=10, max_steps=a.steps, **kw)
    acc = accuracy(m, xte, yte)
    report['arms'][arm] = {'hist': hist, 'wall_s': wall, 'test_acc': acc, 'final_loss': hist[-1]['loss'],
                           'ms_per_step': wall / a.steps * 1e3}
    print(f'{arm:5s} | final train loss {hist[-1]["loss"]:.4f} | test acc {acc:.4f} | {wall:6.1f}s '
          f'({wall / a.steps * 1e3:.0f} ms/step)', flush=True)

# --- steps-to-target and time-to-target ---------------------------------------------------------------------
targets = [0.30, 0.25, 0.22, 0.20]
report['targets'] = {}
for tgt in targets:
    row = {}
    for arm, d in report['arms'].items():
        hit = next((h for h in d['hist'] if h['loss'] <= tgt), None)
        row[arm] = None if hit is None else {'step': hit['step'], 'time_s': hit['time']}
    report['targets'][tgt] = row
    print(f'  target train loss {tgt:.2f}: ' + ' | '.join(
        f'{arm} ' + ('never' if v is None else f'step {v["step"]:4d} / {v["time_s"]:6.1f}s') for arm, v in row.items()),
        flush=True)

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump(report, open(a.out, 'w'), indent=1)
print('wrote', a.out)
