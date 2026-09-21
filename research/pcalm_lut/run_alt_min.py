"""Experiment 2: replace PC's gradient weight step with the EXACT least-squares solution.

Each outer step: run the relaxation with the output node clamped to the label (pinned), take the
resulting per-layer targets, and solve each layer's parameters exactly against those targets instead of
taking a gradient step. That is alternating minimisation: the relaxation proposes, least squares
disposes. If it trains, PC's targets are fine and the UPDATE RULE is the defect; if it does not, the
targets are reachable but wrong and the RELAXATION is the defect.

REPORTING. With the output pinned and an exact LS step, the readout can be solved almost trivially and a
single aggregate loss would say nothing about the interior. So the readout and the interior are reported
apart:

  readout data loss   1/2||y - readout(h_fwd)||^2 per sample, the model's actual loss.
  INTERIOR data loss  the same but with the readout re-solved by least squares on the CURRENT interior
                      features -- the loss the interior could support if its readout were perfect. This
                      is the number that says whether the interior is learning anything.
  probe accuracy      argmax accuracy of that optimally-read-out interior, train and test.
  r_cur/r_zero        interior constraint violation, same definition as the reachability probe.

The fit batch is 16,384 so the LUT least-squares stays overdetermined 2:1 (8192 free entries per output
dimension); at the training batch of 128 the fit would be trivially exact and would mean nothing.

Usage: python3 run_alt_min.py --outer 40
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from pcalm import ResidualMLP, constraint_sigma_max, squared_error  # noqa: E402
from probe_ls_direction import layer_targets  # noqa: E402
from probe_reachability import lut_lstsq, read_structure  # noqa: E402
from train_paired import inner_loop, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------ LUT -----------
def lut_probe(model, x, y, xte, yte, hs, tg, ridge=1e-3):
    """Interior quality (readout re-solved optimally) separated from the readout's own loss."""
    out = {}
    with torch.no_grad():
        h = model.h1(x)
        for i in range(model.n_hidden - 1):
            h = model.layer(i, h)
        yhat = model.readout(h)
        out['readout_data_loss'] = float(0.5 * (yhat - y).pow(2).sum() / x.shape[0])
        out['train_acc'] = float((yhat.argmax(-1) == y.argmax(-1)).float().mean())
        hte = model.h1(xte)
        for i in range(model.n_hidden - 1):
            hte = model.layer(i, hte)
        out['test_acc'] = float((model.readout(hte).argmax(-1) == yte.argmax(-1)).float().mean())
    # the interior's own quality: solve the readout exactly on these features
    n_cells = model.f_out.n_tables * model.f_out.table_size
    fi, co = read_structure(model.f_out, h, model.ai)
    V, fit, _ = lut_lstsq(fi, co, y, n_cells, iters=300, ridge=ridge)   # ONE solve: fit + test probe
    out['interior_data_loss'] = float(0.5 * (fit - y).pow(2).sum() / x.shape[0])
    out['probe_train_acc'] = float((fit.argmax(-1) == y.argmax(-1)).float().mean())
    with torch.no_grad():
        old = model.f_out.tables.detach().clone()
        model.f_out.tables.copy_(V.view_as(model.f_out.tables))
        hte = model.h1(xte)
        for i in range(model.n_hidden - 1):
            hte = model.layer(i, hte)
        out['probe_test_acc'] = float((model.readout(hte).argmax(-1) == yte.argmax(-1)).float().mean())
        model.f_out.tables.copy_(old)
    # interior constraint violation against the relaxation's own targets
    viol = []
    for i, (zin, target, is_out) in enumerate(tg[:-1]):
        with torch.no_grad():
            cur = model.ai * model.f_lut[i](zin)
            viol.append(float((target - cur).pow(2).sum()) / max(float(target.pow(2).sum()), 1e-30))
    out['interior_viol'] = sum(viol) / len(viol)
    return out


def run_lut(a, xtr, ytr, xte, yte, dev, arm):
    torch.manual_seed(a.seed)
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev,
                           seed=a.seed, table_dropout=0.0, clamp_mode='pinned')
    fx, fy = xtr[:a.batch], ytr[:a.batch]
    sig = sigma_max_A(model, fx[:128], fy[:128], arm)
    eta = (0.5 / max(sig ** 2, 1e-12) if arm == 'pcA' else 2.0 / max(sig ** 2 * 3.0, 1e-12))
    n_cells = model.f_out.n_tables * model.f_out.table_size
    luts = list(model.f_lut) + [model.f_out]
    rows = []
    for step in range(a.outer + 1):
        hs, _, _ = inner_loop(model, fx, fy, arm, T=a.T, eta_h=eta, alpha=1.0, rho=1.0)
        hs = [h.detach() for h in hs]
        tg = layer_targets(model, hs, fx, fy)
        if step % a.probe_every == 0:
            r = lut_probe(model, fx, fy, xte[:2000], yte[:2000], hs, tg, ridge=a.ridge)
            r['step'] = step
            rows.append(r)
            print(f'  [{arm}] outer {step:>3d}  readout loss {r["readout_data_loss"]:.4f}  '
                  f'interior loss {r["interior_data_loss"]:.4f}  acc {r["train_acc"]:.4f}/'
                  f'{r["test_acc"]:.4f}  probe acc {r["probe_train_acc"]:.4f}/{r["probe_test_acc"]:.4f}  '
                  f'viol {r["interior_viol"]:.4f}', flush=True)
        if step == a.outer:
            break
        sols = []
        for i, (lut, (zin, target, is_out)) in enumerate(zip(luts, tg)):
            fi, co = read_structure(lut, zin, model.ai)
            V, _, _ = lut_lstsq(fi, co, target, n_cells, iters=a.cg_iters, ridge=a.ridge)
            sols.append(V)
        with torch.no_grad():
            for lut, V in zip(luts, sols):
                lut.tables.mul_(1 - a.damping).add_(a.damping * V.view_as(lut.tables))
    return rows


# ------------------------------------------------------------------------------------ MLP -----------
def run_mlp(a, xtr, ytr, xte, yte, dev, mode):
    torch.manual_seed(a.seed)
    model = ResidualMLP(xtr.shape[1], 10, a.width, a.depth, device=dev, seed=a.seed)
    model.clamp_mode = 'pinned'
    fx, fy = xtr[:a.batch], ytr[:a.batch]
    sig = constraint_sigma_max(model, fx[:128], fy[:128])
    eta = 1.0 / max(sig ** 2, 1e-12)
    alpha = 0.0 if mode == 'pc' else 1.0
    rows = []
    for step in range(a.outer + 1):
        hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(fx)]
        with torch.no_grad():
            lam = [torch.zeros_like(r) for r in model.residuals(fx, hs, fy)]
        for _ in range(a.T):
            e, r = model.energy(fx, fy, hs, lam, 1.0, squared_error)
            g = torch.autograd.grad(e, hs)
            with torch.no_grad():
                for h, gi in zip(hs, g):
                    h -= eta * gi
                if alpha:
                    for li, ri in zip(lam, r):
                        li += alpha * ri.detach()
        hs = [h.detach() for h in hs]
        if step % a.probe_every == 0:
            with torch.no_grad():
                h = model.h1(fx)
                for i in range(model.n_hidden - 1):
                    h = model.layer(i, h)
                yhat = model.readout(h)
                hte = model.h1(xte)
                for i in range(model.n_hidden - 1):
                    hte = model.layer(i, hte)
                A = model.aL * model.act(h)
                sol = torch.linalg.lstsq(A, fy).solution
                fit = A @ sol
                Ate = model.aL * model.act(hte)
                rows.append({'step': step,
                             'readout_data_loss': float(0.5 * (yhat - fy).pow(2).sum() / fx.shape[0]),
                             'interior_data_loss': float(0.5 * (fit - fy).pow(2).sum() / fx.shape[0]),
                             'train_acc': float((yhat.argmax(-1) == fy.argmax(-1)).float().mean()),
                             'test_acc': float((model.readout(hte).argmax(-1) == yte.argmax(-1)).float().mean()),
                             'probe_train_acc': float((fit.argmax(-1) == fy.argmax(-1)).float().mean()),
                             'probe_test_acc': float(((Ate @ sol).argmax(-1) == yte.argmax(-1)).float().mean()),
                             'interior_viol': float(sum(
                                 float((hs[i + 1] - model.layer(i, hs[i])).pow(2).sum())
                                 / max(float((hs[i + 1] - hs[i]).pow(2).sum()), 1e-30)
                                 for i in range(model.n_hidden - 1)) / max(model.n_hidden - 1, 1))})
            r = rows[-1]
            print(f'  [mlp {mode}] outer {step:>3d}  readout loss {r["readout_data_loss"]:.4f}  '
                  f'interior loss {r["interior_data_loss"]:.4f}  acc {r["train_acc"]:.4f}/'
                  f'{r["test_acc"]:.4f}  probe acc {r["probe_train_acc"]:.4f}/{r["probe_test_acc"]:.4f}  '
                  f'viol {r["interior_viol"]:.4f}', flush=True)
        if step == a.outer:
            break
        with torch.no_grad():
            for i in range(model.n_hidden - 1):
                A = model.ai * model.act(hs[i])
                sol = torch.linalg.lstsq(A, hs[i + 1] - hs[i]).solution
                model.Wi[i].mul_(1 - a.damping).add_(a.damping * sol.t())
            A = model.aL * model.act(hs[-1])
            sol = torch.linalg.lstsq(A, fy).solution
            model.WL.mul_(1 - a.damping).add_(a.damping * sol.t())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--outer', type=int, default=40)
    ap.add_argument('--probe-every', type=int, default=2)
    ap.add_argument('--damping', type=float, default=1.0, help='1.0 = pure alternating minimisation')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cg-iters', type=int, default=400)
    ap.add_argument('--ridge', type=float, default=1e-3,
                    help='Tikhonov ridge as a fraction of the mean diagonal of A^T A. Without '
                         'it the table least-squares blows up on rarely addressed cells.')
    ap.add_argument('--out', default='runs_debug/alt_min.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    xtr, ytr = load('fashion', train=True, device=dev)
    xte, yte = load('fashion', train=False, device=dev)
    out = {'cfg': vars(a), 'arms': {}}
    for arm in ('pcA', 'pcalmB'):
        print(f'\n=== LUT {arm}, exact LS weight step, output pinned, damping {a.damping}')
        out['arms'][f'lut-{arm}'] = run_lut(a, xtr, ytr, xte, yte, dev, arm)
    for mode in ('pc',):
        print(f'\n=== MLP {mode}, exact LS weight step, output pinned, damping {a.damping}')
        out['arms'][f'mlp-{mode}'] = run_mlp(a, xtr, ytr, xte, yte, dev, mode)
    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
