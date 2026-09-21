"""Why is the INTERIOR update anti-aligned with backprop?

Adam normalises per-parameter, so the 979x magnitude excess should largely divide out and direction is
the suspect. Three measurements, all on the same trained model at matched checkpoints:

 (a) PATH ATTRIBUTION. The read is  score_t * sum_i w_i row_i , and the two factors compose
     multiplicatively, so detaching one leaves the other's gradient exact. The relaxed states h* are
     computed ONCE with both paths live; the weight gradient is then recomputed with the blend weights
     detached ('score only') and with the score detached ('routing only'), and each is compared with
     backprop. This attributes the UPDATE, holding the relaxation fixed.

 (b) ADDRESS SEARCH, DIRECTLY. At every inner step, the state update u = -eta*grad(E) is compared with
     the direction that DECREASES the smallest margins, -grad_h(sum of min margins). A positive cosine
     means the relaxation is actively walking states toward cell boundaries.

 (c) IS ADDRESS SEARCH THE CAUSE? A variant where the cells read are frozen at their feedforward values
     for the whole relaxation -- no address search at all, continuous gates still live. If the cosine to
     BP goes positive, address search is what buys the anti-alignment; if it stays negative, the
     anti-alignment lives in the continuous path and address search is innocent.

Usage: python3 probe_interior_paths.py --table-dropout 0.25 --steps 500
"""
import argparse
import json
import math
import os
import sys

import torch
import torch._functorch.config as _ft_config

_ft_config.donated_buffer = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from train_paired import arm_grads, cos, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def f_luts(model):
    return list(model.f_lut) + [model.f_out]


def set_ablation(model, mode):
    for l in f_luts(model):
        l.set_path_ablation(mode)


def freeze_cells(model, capture=False, on=True):
    for l in f_luts(model):
        l.freeze_blend_cells(on=on, capture=capture)


def margin_objective(model, hs):
    """M(h) = sum over forward LUTs and (sample, table) of the SMALLEST anchor margin at that layer."""
    zs = [hs[i] for i in range(model.n_hidden - 1)] + [hs[-1]]
    tot = 0.0
    for lut, z in zip(f_luts(model), zs):
        d = (z[:, lut.anchor_a] - z[:, lut.anchor_b]).abs()
        tot = tot + d.min(-1).values.sum()
    return tot


def relax_with_diagnostics(model, x, y, *, T, eta_h):
    """Arm A's relaxation, instrumented: how the update direction relates to the margin direction."""
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    with torch.no_grad():
        m_start = float(margin_objective(model, hs))
    coss = []
    for _ in range(T):
        e, rf, rb = model.energy_A(x, y, hs)
        g = torch.autograd.grad(e, hs)
        hm = [h.detach().clone().requires_grad_(True) for h in hs]
        gm = torch.autograd.grad(margin_objective(model, hm), hm)
        u = torch.cat([(-gi).flatten() for gi in g])            # the direction h actually moves
        dm = torch.cat([(-gj).flatten() for gj in gm])          # the direction that DECREASES margins
        coss.append(cos(u, dm))
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
    with torch.no_grad():
        m_end = float(margin_objective(model, hs))
    return hs, {'cos_update_vs_margin_decrease': sum(coss) / len(coss),
                'cos_first': coss[0], 'cos_last': coss[-1],
                'margin_sum_start': m_start, 'margin_sum_end': m_end,
                'margin_change': m_end / max(m_start, 1e-12)}


def interior_cos(model, x, y, hs, ref):
    """Per-layer cosine to BP of the weight gradient taken at the GIVEN states."""
    model.zero_grad(set_to_none=True)
    e, _, _ = model.energy_A(x, y, hs)
    (e / x.shape[0]).backward()
    out = [cos(l.tables.grad, ref[i]) if l.tables.grad is not None else float('nan')
           for i, l in enumerate(f_luts(model))]
    model.zero_grad(set_to_none=True)
    return out


def checkpoint(model, x, y, *, T, eta_h, tag):
    row = {'tag': tag}
    # backprop reference on these weights
    arm_grads(model, x, y, 'bp', T=T, eta_h=0.0, alpha=1.0, rho=1.0)
    ref = [l.tables.grad.detach().clone() for l in f_luts(model)]
    model.zero_grad(set_to_none=True)

    # (b) the relaxation, instrumented; its h* is reused for (a)
    hs, diag = relax_with_diagnostics(model, x, y, T=T, eta_h=eta_h)
    row.update(diag)
    hs = [h.detach().clone().requires_grad_(True) for h in hs]
    with torch.no_grad():
        ff = model.init_states(x)
        row['margin_ff'] = float(margin_objective(model, ff)) / x.shape[0]
        row['margin_relaxed'] = float(margin_objective(model, hs)) / x.shape[0]

    # (a) path attribution, same states, one path detached at a time
    for mode, name in ((None, 'full'), ('score_only', 'score_only'), ('routing_only', 'routing_only')):
        set_ablation(model, mode)
        row[f'cos_{name}'] = interior_cos(model, x, y, hs, ref)
    set_ablation(model, None)

    # (c) frozen cells: the relaxation itself runs with no address search
    freeze_cells(model, capture=True)
    with torch.no_grad():
        model(x)                                    # one forward to capture the feedforward cells
    hs_f = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    for _ in range(T):
        e, _, _ = model.energy_A(x, y, hs_f)
        g = torch.autograd.grad(e, hs_f)
        with torch.no_grad():
            for h, gi in zip(hs_f, g):
                h -= eta_h * gi
    hs_f = [h.detach().clone().requires_grad_(True) for h in hs_f]
    row['cos_frozen_cells'] = interior_cos(model, x, y, hs_f, ref)
    with torch.no_grad():
        row['margin_relaxed_frozen'] = float(margin_objective(model, hs_f)) / x.shape[0]
    freeze_cells(model, on=False)
    return row


def show(rows, key, title):
    print(f'\n   {title}')
    n = len(rows[0][key])
    print(f'   {"step":>6s} ' + ' '.join(f'{("L%d" % i) if i < n - 1 else "out":>7s}' for i in range(n))
          + f' {"mean":>7s} {"interior":>9s}')
    for r in rows:
        v = r[key]
        inter = [q for q in v[:-1]]
        print(f'   {r["tag"]:>6s} ' + ' '.join(f'{q:>+7.3f}' for q in v)
              + f' {sum(v) / len(v):>+7.3f} {sum(inter) / len(inter):>+9.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--table-dropout', type=float, default=0.25)
    ap.add_argument('--clamp', default='pinned')
    ap.add_argument('--eta-frac', type=float, default=0.5)
    ap.add_argument('--at', default='1,100,250,500')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    xtr, ytr = load('fashion', train=True, device=dev)
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed,
                           table_dropout=a.table_dropout, clamp_mode=a.clamp)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    x, y = next(iter(loader))
    want = sorted(int(v) for v in a.at.split(','))
    print(f'arm pcA clamp {a.clamp} L={a.depth} N={a.width} tables={a.tables} T={a.T} '
          f'table_dropout={a.table_dropout} seed={a.seed}')

    sig = sigma_max_A(model, x, y, 'pcA')
    eta = a.eta_frac / max(sig ** 2, 1e-12)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows, it, step = [], iter(loader), 0
    while step <= max(want):
        if step in want:
            model.clear_dropout()                  # diagnostics see the full network
            rows.append(checkpoint(model, x, y, T=a.T, eta_h=eta, tag=str(step)))
        try:
            bx, by = next(it)
        except StopIteration:
            it = iter(loader)
            bx, by = next(it)
        model.resample_dropout(bx.shape[0])
        arm_grads(model, bx, by, 'pcA', T=a.T, eta_h=eta, alpha=1.0, rho=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.zero_grad(set_to_none=True)
        step += 1

    print('\n== (a) PATH ATTRIBUTION: per-layer cosine to BP of the interior weight update')
    print('   (same relaxed states in all three rows; only the gradient path differs)')
    for k, t in (('cos_full', 'full (both paths live)'),
                 ('cos_score_only', 'score path only (blend weights detached)'),
                 ('cos_routing_only', 'routing path only (score detached)')):
        show(rows, k, t)

    print('\n== (c) ADDRESS SEARCH FROZEN: cells held at their feedforward values through the relaxation')
    show(rows, 'cos_frozen_cells', 'frozen cells, both continuous paths live')

    print('\n== (b) DOES THE RELAXATION WALK TOWARD CELL BOUNDARIES?')
    print('   cos(update direction, direction that DECREASES the smallest margins), averaged over the')
    print('   inner loop; margin sums are per sample.')
    print(f'\n   {"step":>6s} {"cos mean":>9s} {"cos first":>10s} {"cos last":>9s} {"margin ff":>10s} '
          f'{"margin relaxed":>15s} {"relaxed/ff":>11s} {"frozen-cell relaxed":>20s}')
    for r in rows:
        print(f'   {r["tag"]:>6s} {r["cos_update_vs_margin_decrease"]:>+9.3f} {r["cos_first"]:>+10.3f} '
              f'{r["cos_last"]:>+9.3f} {r["margin_ff"]:>10.4f} {r["margin_relaxed"]:>15.4f} '
              f'{r["margin_relaxed"] / max(r["margin_ff"], 1e-12):>11.3f} '
              f'{r["margin_relaxed_frozen"]:>20.4f}')

    out = a.out or f'runs_debug/interior_paths_td{a.table_dropout}.json'
    p = os.path.join(HERE, out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'cfg': vars(a), 'rows': rows}, open(p, 'w'), indent=1)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
