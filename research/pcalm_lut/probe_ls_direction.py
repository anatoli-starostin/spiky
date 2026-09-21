"""Experiment 3: does "reachable" survive unfreezing the reads, and where does PC's update point?

A structural fact decides half of this before any measurement. In LightMHL the read structure -- the
packed address, the confidence score, the blend weights -- is computed from the layer's INPUT and its
anchors alone:

    d = z[anchor_a] - z[anchor_b];  index = pack(sign(d));  score = s(d);  w = softmax(-2 m_min / tau)

None of it touches the table entries. So for a layer in isolation, "freeze the reads, fit the tables"
is not an approximation at all: recomputing the reads after installing the fitted tables returns exactly
the same reads. That is asserted numerically here (step 2 of the report) rather than argued.

The question with teeth is COMPOSITIONAL. Installing the least-squares solution at layer i changes what
layer i OUTPUTS, which changes the input of layer i+1, which does change that layer's reads. So this
probe installs the LS solution at every forward layer at once, recomputes the whole forward pass, and
re-measures every layer's residual against the SAME targets. If the residual snaps back, the fits are
individually reachable but jointly incompatible.

It also reports the cosine between PC's ACTUAL weight update and the LS direction (V* - V_current), per
layer and per checkpoint: whether PC is moving toward the exactly-solvable answer, away from it, or
sideways.

Usage: python3 probe_ls_direction.py --batch 16384 --steps 500
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from probe_reachability import lut_lstsq, read_structure  # noqa: E402
from train_paired import arm_grads, cos, inner_loop, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def layer_targets(model, hs, x, y):
    """(input, target, is_readout) per forward layer, from the relaxed states."""
    out = []
    for i in range(model.n_hidden - 1):
        out.append((hs[i], hs[i + 1] - hs[i], False))        # residual block: fit a_i LUT(h) to h' - h
    top = y if model.top_pinned() else model.readout(hs[-1])
    out.append((hs[-1], top, True))                          # readout is bare: fit a_i LUT(h) to the top
    return out


def checkpoint(model, x, y, *, T, eta_h, arm, tag, cg_iters):
    luts = list(model.f_lut) + [model.f_out]
    n_cells = model.f_out.n_tables * model.f_out.table_size
    hs, _, _ = inner_loop(model, x, y, arm, T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    hs = [h.detach() for h in hs]
    tg = layer_targets(model, hs, x, y)

    # PC's actual update direction on this batch, for the cosine
    arm_grads(model, x, y, arm, T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    pc_grad = [l.tables.grad.detach().clone() if l.tables.grad is not None else None for l in luts]
    model.zero_grad(set_to_none=True)

    rows, sols = [], []
    for i, (lut, (zin, target, is_out)) in enumerate(zip(luts, tg)):
        with torch.no_grad():
            cur = (model.readout(zin) if is_out else model.ai * lut(zin))
            r_cur = float((target - cur).pow(2).sum())
            base = float(target.pow(2).sum())
        flat_idx, coef = read_structure(lut, zin, model.ai)
        V, fit, cg_rel = lut_lstsq(flat_idx, coef, target, n_cells, iters=cg_iters)
        r_opt = float((target - fit).pow(2).sum())
        sols.append(V)
        # is PC's update pointing at the LS answer? direction = V* - V_current; PC steps along -grad
        with torch.no_grad():
            ls_dir = V.view_as(lut.tables) - lut.tables
        c = cos(-pc_grad[i], ls_dir) if pc_grad[i] is not None else float('nan')
        rows.append({'layer': 'readout' if is_out else f'L{i}', 'r_cur': r_cur, 'r_opt_frozen': r_opt,
                     'r_zero': base, 'cg_rel_resid': cg_rel, 'cos_pc_update_vs_ls': c})

    # --- (ii)-(iv): install the LS solutions and re-measure, first per layer, then all at once
    old = [l.tables.detach().clone() for l in luts]
    for i, (lut, (zin, target, is_out)) in enumerate(zip(luts, tg)):
        with torch.no_grad():
            lut.tables.copy_(sols[i].view_as(lut.tables))
            cur = (model.readout(zin) if is_out else model.ai * lut(zin))
            rows[i]['r_recomputed_same_input'] = float((target - cur).pow(2).sum())
            lut.tables.copy_(old[i])
    # all layers installed together, whole forward pass recomputed from the same inputs h_{i-1}
    with torch.no_grad():
        for lut, V in zip(luts, sols):
            lut.tables.copy_(V.view_as(lut.tables))
        h = model.h1(x)
        for i in range(model.n_hidden - 1):
            h = model.layer(i, h)
            rows[i]['r_composed'] = float((hs[i + 1] - h).pow(2).sum())
        rows[-1]['r_composed'] = float((tg[-1][1] - model.readout(h)).pow(2).sum())
        for lut, V in zip(luts, old):
            lut.tables.copy_(V)
    return {'tag': tag, 'layers': rows}


def show(name, cps):
    print(f'\n== {name}')
    print(f'   {"step":>5s} {"layer":>8s} {"r_cur":>11s} {"r_opt frozen":>13s} {"r_recomputed":>13s} '
          f'{"r_composed":>11s} {"opt/cur":>8s} {"comp/cur":>9s} {"cos(PC, LS)":>12s} {"CG":>9s}')
    for cp in cps:
        for r in cp['layers']:
            print(f'   {cp["tag"]:>5s} {r["layer"]:>8s} {r["r_cur"]:>11.4e} {r["r_opt_frozen"]:>13.4e} '
                  f'{r["r_recomputed_same_input"]:>13.4e} {r["r_composed"]:>11.4e} '
                  f'{r["r_opt_frozen"] / max(r["r_cur"], 1e-30):>8.4f} '
                  f'{r["r_composed"] / max(r["r_cur"], 1e-30):>9.4f} '
                  f'{r["cos_pc_update_vs_ls"]:>+12.4f} {r["cg_rel_resid"]:>9.1e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--train-batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cg-iters', type=int, default=800)
    ap.add_argument('--at', default='1,100,250,500')
    ap.add_argument('--out', default='runs_debug/ls_direction.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    want = sorted(int(v) for v in a.at.split(','))
    xtr, ytr = load('fashion', train=True, device=dev)
    fx, fy = xtr[:a.batch], ytr[:a.batch]
    out = {'cfg': vars(a), 'arms': {}}

    for arm in ('pcA', 'pcalmB'):
        torch.manual_seed(a.seed)
        model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev,
                               seed=a.seed, table_dropout=0.0, clamp_mode='pinned')
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = TensorLoader(xtr, ytr, batch_size=a.train_batch, seed=a.seed)
        px, py = xtr[:a.train_batch], ytr[:a.train_batch]
        sig = sigma_max_A(model, px, py, arm)
        eta = (0.5 / max(sig ** 2, 1e-12) if arm == 'pcA' else 2.0 / max(sig ** 2 * 3.0, 1e-12))
        cps, it, step = [], iter(loader), 0
        while step <= max(want):
            if step in want:
                cps.append(checkpoint(model, fx, fy, T=a.T, eta_h=eta, arm=arm, tag=str(step),
                                      cg_iters=a.cg_iters))
                print(f'  [{arm}] step {step} done', flush=True)
            try:
                bx, by = next(it)
            except StopIteration:
                it = iter(loader)
                bx, by = next(it)
            arm_grads(model, bx, by, arm, T=a.T, eta_h=eta, alpha=1.0, rho=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
        out['arms'][arm] = cps
        show(f'LUT {arm}', cps)

    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
