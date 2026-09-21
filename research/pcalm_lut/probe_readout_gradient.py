"""Why does the readout emit almost nothing under the PC arms?

The readout is the ONLY layer whose weight gradient is well aligned with backprop (cosine +0.7 to +0.8),
yet the PC data loss sits at ~0.49 against the trivial 0.5, which is what you get when the readout emits
zero. So the direction is right and the magnitude is dead. This script takes the update apart factor by
factor at matched checkpoints.

The readout's output is  yhat = a_L * sum_t s_t * (w_0 V[t,c_t] + w_1 V[t,c'_t])  and the gradient into a
table entry is proportional to  a_L * s_t * w * r , so there are exactly three places the magnitude can
die: the fixed scale a_L, the confidence score s_t, and the error signal r. Each is measured here, in the
same units, at step 1 and at step 500, for the PC arm and for backprop on the SAME weights.

Usage: python3 probe_readout_gradient.py --arm pcA --clamp pinned --steps 500
"""
import argparse
import json
import math
import os
import sys

import torch
import torch._functorch.config as _ft_config

_ft_config.donated_buffer = False        # several grads through one graph (diagnostic only)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from train_paired import arm_grads, inner_loop, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def rms(t):
    return float(t.pow(2).mean().sqrt()) if t is not None and t.numel() else float('nan')


def readout_score(model, z):
    """The per-table confidence score s_t the readout computes at states z, and the anchor margins."""
    lut = model.f_out
    d = z[:, lut.anchor_a] - z[:, lut.anchor_b]          # [B, n_tables, nap]
    return lut.confidence_score(d), d.abs()


def snapshot(model, x, y, arm, *, T, eta_h, tag):
    out = {'tag': tag}
    pinned = model.top_pinned()

    # --- the states the arm differentiates at, and the readout's own inputs
    hs, _, _ = inner_loop(model, x, y, 'pcA' if arm == 'hybrid' else arm,
                          T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    hs = [h.detach().clone().requires_grad_(True) for h in hs]
    z = hs[-1]
    with torch.no_grad():
        s, m = readout_score(model, z)
        yhat = model.readout(z)
        r_top = (y - yhat) if pinned else (yhat - y)
        out['a_L'] = model.ai
        out['score_rms'] = rms(s)
        out['score_mean_abs'] = float(s.abs().mean())
        out['margin_min_median'] = float(m.min(-1).values.median())
        out['readout_out_rms'] = rms(yhat)
        out['target_rms'] = rms(y)
        out['r_top_rms'] = rms(r_top)
        out['h_last_rms'] = rms(z)
        out['table_rms'] = rms(model.f_out.tables)
        # the product of the three factors, in the units the table gradient is measured in
        out['factor_product'] = out['a_L'] * out['score_mean_abs'] * rms(r_top)

    # --- where the readout's gradient actually comes from
    tab = model.f_out.tables
    top_term = (y - model.readout(z)).pow(2).sum() if pinned else None
    data_term = 0.5 * (model.readout(z) - y).pow(2).sum()
    g_top = torch.autograd.grad(top_term, tab, retain_graph=True)[0] if pinned else None
    g_data = torch.autograd.grad(data_term, tab, retain_graph=True)[0]
    out['grad_from_top_residual_rms'] = rms(g_top)
    out['grad_from_data_term_rms'] = rms(g_data)
    if g_top is not None:
        cs = torch.nn.functional.cosine_similarity(g_top.flatten(), g_data.flatten(), dim=0)
        out['cos_top_vs_data'] = float(cs)
        out['top_over_data_scale'] = rms(g_top) / max(rms(g_data), 1e-30)

    # --- the gradient the arm and BP actually put on the readout, on the same weights
    arm_grads(model, x, y, arm, T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    out['arm_readout_grad_rms'] = rms(model.f_out.tables.grad)
    out['arm_interior_grad_rms'] = rms(model.f_lut[0].tables.grad)
    arm_grads(model, x, y, 'bp', T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    out['bp_readout_grad_rms'] = rms(model.f_out.tables.grad)
    out['bp_interior_grad_rms'] = rms(model.f_lut[0].tables.grad)
    model.zero_grad(set_to_none=True)
    out['readout_ratio_arm_over_bp'] = out['arm_readout_grad_rms'] / max(out['bp_readout_grad_rms'], 1e-30)
    out['interior_ratio_arm_over_bp'] = out['arm_interior_grad_rms'] / max(out['bp_interior_grad_rms'], 1e-30)

    # --- BP's own score/output at the feedforward states, for contrast with the relaxed ones
    with torch.no_grad():
        fs = model.init_states(x)
        s_ff, m_ff = readout_score(model, fs[-1])
        out['score_rms_feedforward'] = rms(s_ff)
        out['margin_min_median_feedforward'] = float(m_ff.min(-1).values.median())
        out['readout_out_rms_feedforward'] = rms(model.readout(fs[-1]))
    return out


def show(rows):
    keys = [('a_L', 'a_L (the fixed 1/sqrt(L*N) scale)'),
            ('score_mean_abs', 'mean |s_t| (readout confidence score)'),
            ('score_rms', 'RMS s_t'),
            ('score_rms_feedforward', 'RMS s_t at the FEEDFORWARD states'),
            ('margin_min_median', 'median smallest margin at the relaxed states'),
            ('margin_min_median_feedforward', 'median smallest margin, feedforward'),
            ('r_top_rms', 'RMS of the error signal r (y - yhat)'),
            ('readout_out_rms', 'RMS readout output at the relaxed states'),
            ('readout_out_rms_feedforward', 'RMS readout output, feedforward'),
            ('target_rms', 'RMS target (one-hot)'),
            ('h_last_rms', 'RMS h_{L-1} (the readout input)'),
            ('table_rms', 'RMS readout table entry'),
            ('factor_product', 'a_L * |s_t| * |r| (the three factors multiplied)'),
            ('grad_from_top_residual_rms', 'readout grad from the TOP RESIDUAL alone'),
            ('grad_from_data_term_rms', 'readout grad from a 0.5-weighted DATA TERM alone'),
            ('top_over_data_scale', '  ... their magnitude ratio'),
            ('cos_top_vs_data', '  ... their cosine'),
            ('arm_readout_grad_rms', 'ARM readout table grad RMS'),
            ('bp_readout_grad_rms', 'BP   readout table grad RMS'),
            ('readout_ratio_arm_over_bp', '  ... arm / BP'),
            ('arm_interior_grad_rms', 'ARM interior (layer 0) table grad RMS'),
            ('bp_interior_grad_rms', 'BP   interior (layer 0) table grad RMS'),
            ('interior_ratio_arm_over_bp', '  ... arm / BP')]
    w = max(len(d) for _, d in keys)
    print(f'\n   {"quantity":{w}s} ' + ' '.join(f'{r["tag"]:>14s}' for r in rows))
    for k, d in keys:
        vals = []
        for r in rows:
            v = r.get(k, float('nan'))
            vals.append(f'{v:>14.4e}' if (isinstance(v, float) and (abs(v) < 1e-3 or abs(v) > 1e4)
                                          and v == v) else f'{v:>14.4f}')
        print(f'   {d:{w}s} ' + ' '.join(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='pcA', choices=['pcA', 'pcalmB', 'hybrid'])
    ap.add_argument('--clamp', default='pinned', choices=['pinned', 'data'])
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--table-dropout', type=float, default=0.25)
    ap.add_argument('--eta-frac', type=float, default=0.5)
    ap.add_argument('--out', default='runs_pinned/readout_gradient.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    xtr, ytr = load('fashion', train=True, device=dev)
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed,
                           table_dropout=a.table_dropout, clamp_mode=a.clamp)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    x, y = next(iter(loader))
    print(f'arm {a.arm} clamp {a.clamp} L={a.depth} N={a.width} tables={a.tables} T={a.T} '
          f'dropout {a.table_dropout} | a_L = 1/sqrt(L*N) = {model.ai:.4f}')

    sig = sigma_max_A(model, x, y, 'pcA' if a.arm == 'hybrid' else a.arm)
    eta = (a.eta_frac / max(sig ** 2, 1e-12) if a.arm in ('pcA', 'hybrid')
           else 2.0 / max(sig ** 2 * 3.0, 1e-12))
    rows = []
    model.clear_dropout()
    rows.append(snapshot(model, x, y, a.arm, T=a.T, eta_h=eta, tag='step 1'))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it = iter(loader)
    for s in range(a.steps):
        try:
            bx, by = next(it)
        except StopIteration:
            it = iter(loader)
            bx, by = next(it)
        model.resample_dropout(bx.shape[0])
        arm_grads(model, bx, by, a.arm, T=a.T, eta_h=eta, alpha=1.0, rho=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.zero_grad(set_to_none=True)
    model.clear_dropout()
    rows.append(snapshot(model, x, y, a.arm, T=a.T, eta_h=eta, tag=f'step {a.steps}'))
    show(rows)

    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'cfg': vars(a), 'rows': rows}, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
