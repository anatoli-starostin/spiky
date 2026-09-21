"""GATE 0 (cheap, ~minutes): does the PC / PC-ALM inference phase flip LUT addresses at all?

The address-search hypothesis needs the inference phase to move h across sign thresholds so a different cell is
read. The energy is piecewise constant in the address, so the relaxation could instead (a) never flip -- the
gradient lives only in the continuous score/readout terms and dies on the plateau -- or (b) flip chaotically,
oscillating between cells without settling. Both would kill the hypothesis; this probe distinguishes them
BEFORE any sweep.

Reports, per inner iteration t = 1..T, on a fixed batch and an untrained (and optionally briefly trained) stack:
  * flip rate: fraction of (sample, table) address slots differing from the FORWARD-PASS address
  * new flips at t and flip-backs (a slot returning to a cell it already held) -> oscillation vs settling
  * how many slots ever flipped, and how many distinct cells each slot visited
  * margin statistics: |d| of the smallest-margin bit, which sets how far h must move to flip

    python probe_address_flips.py [--depth 4] [--width 32] [--tables 16] [--nap 8] [--mode pcalm|pc]
"""
import argparse
import json
import os

import torch

from lut_stack import LUTStack
from pcalm import squared_error, constraint_sigma_max

ap = argparse.ArgumentParser()
ap.add_argument('--depth', type=int, default=4)
ap.add_argument('--width', type=int, default=32)
ap.add_argument('--tables', type=int, default=16)
ap.add_argument('--nap', type=int, default=8)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--in-dim', type=int, default=64)
ap.add_argument('--out-dim', type=int, default=10)
ap.add_argument('--T', type=int, default=None, help='inner steps (default 2L)')
ap.add_argument('--eta-frac', type=float, default=1.0, help='eta_h = eta_frac / sigma_max^2')
ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--rho', type=float, default=1.0)
ap.add_argument('--modes', default='pc,pcalm')
ap.add_argument('--pretrain-steps', type=int, default=0, help='BP steps before probing (0 = at init)')
ap.add_argument('--out', default=None)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

model = LUTStack(a.in_dim, a.out_dim, a.width, a.depth, tables_per_layer=a.tables, nap=a.nap, device=dev, seed=1)
x = torch.randn(a.batch, a.in_dim, device=dev)
y = torch.nn.functional.one_hot(torch.randint(0, a.out_dim, (a.batch,), device=dev), a.out_dim).float()
if a.pretrain_steps:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(a.pretrain_steps):
        opt.zero_grad(set_to_none=True)
        squared_error(model(x), y).backward()
        opt.step()

n_cells = 1 << a.nap
sigma = constraint_sigma_max(model, x)
eta = a.eta_frac / max(sigma ** 2, 1e-12)
T = a.T or 2 * a.depth
print(f'LUTStack depth {a.depth} (L-2 = {a.depth - 2} LUT layers), width {a.width}, {a.tables} tables/layer, '
      f'nap {a.nap} -> {n_cells} cells, read_top_n 2')
print(f'sigma_max(A) {sigma:.4f} -> eta_h {eta:.3e}; T {T}; alpha {a.alpha}; rho {a.rho}; '
      f'pretrain {a.pretrain_steps} BP steps')
report = {'args': vars(a), 'sigma_max': sigma, 'eta_h': eta, 'modes': {}}

for mode in a.modes.split(','):
    alpha = 0.0 if mode == 'pc' else a.alpha
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    lam = [torch.zeros_like(h) for h in hs]
    a0 = model.addresses(hs)                                  # forward-pass address
    seen = [{} for _ in a0]                                   # per layer: slot -> set of visited cells
    for li, aa in enumerate(a0):
        seen[li] = [set() for _ in range(aa.numel())]
        for k, v in enumerate(aa.reshape(-1).tolist()):
            seen[li][k].add(v)
    prev = [aa.clone() for aa in a0]
    m0 = model.margins(hs)
    print(f'--- {mode}: min|d| per layer (median over slots): ' +
          ' '.join(f'{float(mm.min(-1).values.median()):.4f}' for mm in m0))
    rows = []
    for t in range(1, T + 1):
        e, r = model.energy(x, y, hs, lam, a.rho, squared_error)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta * gi
            if alpha:
                for li_, ri in zip(lam, r):
                    li_ += alpha * ri.detach()
            cur = model.addresses(hs)
            vs_fwd = [float((c != f).float().mean()) for c, f in zip(cur, a0)]
            vs_prev = [float((c != p).float().mean()) for c, p in zip(cur, prev)]
            backs = 0
            for li_, cc in enumerate(cur):
                for k, v in enumerate(cc.reshape(-1).tolist()):
                    if v in seen[li_][k] and v != prev[li_].reshape(-1)[k].item():
                        backs += 1
                    seen[li_][k].add(v)
            prev = [cc.clone() for cc in cur]
            dh = float(torch.stack([gi.pow(2).mean() for gi in g]).mean().sqrt()) * eta
        rows.append({'t': t, 'flip_vs_forward': vs_fwd, 'flip_vs_prev': vs_prev, 'flip_backs': backs,
                     'step_rms': dh})
        if t <= 4 or t == T or t % max(1, T // 6) == 0:
            print(f'  t={t:3d} flips vs forward {["%.4f" % v for v in vs_fwd]} | new-vs-prev '
                  f'{["%.4f" % v for v in vs_prev]} | flip-backs {backs} | |dh| rms {dh:.2e}')
    with torch.no_grad():
        ever = [float((c != f).float().mean()) for c, f in zip(prev, a0)]
        visited = [sum(len(s) for s in layer) / len(layer) for layer in seen]
    print(f'  END: fraction of slots whose cell != forward-pass cell: {["%.4f" % v for v in ever]}')
    print(f'       mean distinct cells visited per slot (1 = never moved): {["%.3f" % v for v in visited]}')
    report['modes'][mode] = {'rows': rows, 'ever_changed': ever, 'cells_visited_mean': visited,
                             'min_margin_median': [float(mm.min(-1).values.median()) for mm in m0]}

if a.out:
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(report, open(a.out, 'w'), indent=1)
    print('wrote', a.out)
