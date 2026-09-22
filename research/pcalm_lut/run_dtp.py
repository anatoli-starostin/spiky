"""Train the paired stacks with pure difference target propagation, and with BP as the matched control.

Standing debug rules: L=4, 500 steps, seed 0, no dropout, local logging only, nothing at L=16.

The result lives in the instrumentation rather than in accuracy: inverse quality ||g(f(h)) - h||/||h||
per layer is the load-bearing quantity, because DTP cannot route credit at all through an inverse that
does not invert. within_cell_fraction reports the predicted floor for the LUT.

Usage: python3 run_dtp.py --model lut --rule dtp --optimizer adam --lr 1e-3
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from dtp import (PairedMLP, all_f_params, all_g_params, dtp_step, forward_states,  # noqa: E402
                 inverse_quality, within_cell_fraction)
from paired import PairedLUTStack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build(a, in_dim, dev):
    if a.model == 'lut':
        return PairedLUTStack(in_dim, 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed,
                              table_dropout=0.0, clamp_mode='pinned')
    return PairedMLP(in_dim, 10, width=a.width, depth=a.depth, device=dev, seed=a.seed)


def make_opt(kind, params, lr):
    return (torch.optim.Adam(params, lr=lr) if kind == 'adam'
            else torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0))


def margins(model, hs):
    if isinstance(model, PairedMLP):
        return float('nan')
    luts = list(model.f_lut) + [model.f_out]
    zs = [hs[i] for i in range(model.n_hidden - 1)] + [hs[-1]]
    q = []
    with torch.no_grad():
        for lut, z in zip(luts, zs):
            d = (z[:, lut.anchor_a] - z[:, lut.anchor_b]).abs()
            q.append(float(d.min(-1).values.median()))
    return sum(q) / len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='lut', choices=['lut', 'mlp'])
    ap.add_argument('--rule', default='dtp', choices=['dtp', 'dtp-plain', 'bp'])
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--g-lr', type=float, default=None, help='defaults to --lr')
    ap.add_argument('--g-steps', type=int, default=1)
    ap.add_argument('--sigma', type=float, default=0.1,
                    help='noise for the g objective, as a fraction of each layer per-sample RMS')
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--probe-every', type=int, default=25)
    ap.add_argument('--out-dir', default='runs_dtp')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    xtr, ytr = load('fashion', train=True, device=dev)
    xte, yte = load('fashion', train=False, device=dev)
    model = build(a, xtr.shape[1], dev)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    px, py = xtr[:a.batch], ytr[:a.batch]
    name = a.name or f'{a.model}-{a.rule}-{a.optimizer}{a.lr}'
    fp, gp = all_f_params(model), all_g_params(model)
    opt_f = make_opt(a.optimizer, fp, a.lr)
    opt_g = make_opt(a.optimizer, gp, a.g_lr if a.g_lr is not None else a.lr)
    print(f'{name} | {a.model} L={a.depth} N={a.width} tables={a.tables} | rule {a.rule} | '
          f'{a.optimizer} lr {a.lr} g_lr {a.g_lr if a.g_lr is not None else a.lr} | '
          f'sigma {a.sigma} g_steps {a.g_steps} | {len(fp)} f tensors, {len(gp)} g tensors', flush=True)

    hist, it, t0 = [], iter(loader), time.time()
    for step in range(1, a.steps + 1):
        try:
            bx, by = next(it)
        except StopIteration:
            it = iter(loader)
            bx, by = next(it)
        ts = time.time()
        if a.rule == 'bp':
            opt_f.zero_grad(set_to_none=True)
            loss = 0.5 * (model(bx) - by).pow(2).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fp, 1.0)
            opt_f.step()
            opt_f.zero_grad(set_to_none=True)
            dl, extra = float(loss.detach()), {}
        else:
            dl, extra = dtp_step(model, bx, by, opt_f, opt_g, sigma=a.sigma, g_steps=a.g_steps,
                                 difference=(a.rule == 'dtp'), fp=fp, gp=gp)
        dt = time.time() - ts
        if step % a.probe_every == 0 or step == 1:
            hs, yhat = forward_states(model, px)
            inv = inverse_quality(model, hs)
            wcf = within_cell_fraction(model, hs)
            with torch.no_grad():
                acc = float((model(xte[:2000]).argmax(-1) == yte[:2000].argmax(-1)).float().mean())
                tra = float((model(px).argmax(-1) == py.argmax(-1)).float().mean())
                pdl = float(0.5 * (model(px) - py).pow(2).sum(-1).mean())
            row = {'step': step, 'train/data_loss': dl, 'train/s_per_step': dt,
                   'eval/test_acc': acc, 'eval/train_acc': tra, 'eval/probe_data_loss': pdl,
                   'inv/mean': sum(inv) / len(inv), 'margin/m_min_p50_mean': margins(model, hs)}
            for i, v in enumerate(inv):
                row[f'inv/L{i}'] = v
            if wcf is not None:
                row['cell/within_frac_mean'] = sum(wcf) / len(wcf)
                for i, v in enumerate(wcf):
                    row[f'cell/within_frac_L{i}'] = v
            for k, v in extra.get('f_loss', {}).items():
                row[f'floss/{k}'] = v
            for k, v in extra.get('g_loss', {}).items():
                row[f'gloss/{k}'] = v
            hist.append(row)
            print(f'  step {step:>4d}  data {pdl:.4f}  acc {tra:.4f}/{acc:.4f}  '
                  f'inv {row["inv/mean"]:.4f}  m_min {row["margin/m_min_p50_mean"]:.4f}  '
                  f'{dt * 1e3:.0f} ms/step', flush=True)
            if not math.isfinite(pdl):
                print('  diverged (non-finite loss) -- stopping', flush=True)
                break

    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(vars(a), exp_name=name,
               _note='Pure difference target propagation: targets are two g reads and a subtraction, '
                     'computed under no_grad; autograd appears only in each layer local loss w.r.t. its '
                     'own parameters. test_dtp_locality.py asserts both.')
    summary = {'wall_s': time.time() - t0, 'final_test_acc': hist[-1]['eval/test_acc'],
               'final_data_loss': hist[-1]['eval/probe_data_loss'], 'final_inv': hist[-1]['inv/mean'],
               'steps_done': hist[-1]['step'],
               's_per_step': st.median([r['train/s_per_step'] for r in hist])}
    json.dump({'cfg': cfg, 'hist': hist, 'summary': summary},
              open(os.path.join(out_dir, 'run.json'), 'w'), indent=1)
    print(f'{name} done: {summary["wall_s"]:.1f}s, acc {summary["final_test_acc"]:.4f}, '
          f'data loss {summary["final_data_loss"]:.4f}, inverse {summary["final_inv"]:.4f}')


if __name__ == '__main__':
    main()
