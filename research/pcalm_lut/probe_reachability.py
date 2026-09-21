"""Are the per-layer targets predictive coding proposes actually REACHABLE by the layer?

Backprop never proposes a target: its update is derived from the layer's own Jacobian, so it moves along
directions the layer can express. PC assigns each layer an explicit target state h_i and asks
f_i(h_{i-1}) to reach it. On an MLP any target is nearly reachable; on a LUT layer the read is a blend of
two rows per table, so a target proposed by gradient descent in state space may be off the manifold.

MEASUREMENT. At a checkpoint, run the relaxation, then FREEZE h_{i-1}, the target h_i, and the read
structure (addresses, confidence scores, blend weights) that h_{i-1} dictates. With those frozen the
layer output is LINEAR in the table entries, so the best achievable fit is a least-squares problem, and
it is solved exactly rather than approximated:

    LUT:  out[b, d] = a_i * sum_{t,k} coef[b, t, k] * V[idx[b, t, k], d]      (unknowns: every cell)
    MLP:  out[b, d] = a_i * sum_j W[d, j] sigma(h_{i-1})[b, j]                (unknowns: one W row)

both solved per output dimension, the LUT by conjugate gradient on the normal equations with a
matrix-free operator (the design matrix is [B, n_tables*table_size] and sparse), the MLP by lstsq.

THE OVERDETERMINATION CAVEAT, which decides whether this measures anything at all. A LUT layer has
n_tables * table_size = 8192 free entries per output dimension. With a 128-row batch the fit is
trivially exact and the measurement is vacuous. The batch here is therefore large enough that the system
is overdetermined, and the report states the unknowns-to-equations ratio for every layer so the LUT and
the MLP can be compared on equal footing rather than on appearance.

Usage: python3 probe_reachability.py --batch 16384 --steps 500
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from pcalm import ResidualMLP, constraint_sigma_max, train_step  # noqa: E402
from train_paired import arm_grads, inner_loop, sigma_max_A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------- LUT read structure ----
@torch.no_grad()
def read_structure(lut, z, a_scale):
    """The frozen read: which cells, with what coefficients, for the input z. Mirrors _blend_bag."""
    d = z[:, lut.anchor_a] - z[:, lut.anchor_b]                      # [B, T, nap]
    index = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1)      # [B, T]
    score = lut.confidence_score(d)                                  # [B, T]
    m = d.abs()
    mv, mj = m.min(dim=-1, keepdim=True)
    bits = (d > 0).to(torch.int64)
    pw = lut.powers[mj]
    bsel = torch.gather(bits, -1, mj)
    idx = torch.cat([index.unsqueeze(-1), index.unsqueeze(-1) + pw * (1 - 2 * bsel)], dim=-1)   # [B,T,2]
    logits = torch.cat([torch.zeros_like(m[..., :1]), -2.0 * mv / lut.read_tau], dim=-1)
    w = torch.softmax(logits, dim=-1)                                # [B, T, 2]
    coef = a_scale * score.unsqueeze(-1) * w                         # [B, T, 2]
    offs = (torch.arange(lut.n_tables, device=z.device) * lut.table_size).view(1, -1, 1)
    return (idx + offs).reshape(z.shape[0], -1), coef.reshape(z.shape[0], -1)   # [B, 2T] each


def lut_lstsq(flat_idx, coef, target, n_cells, iters=400, tol=1e-10, ridge=0.0):
    """min_V ||target - A V||^2 with A given by (flat_idx, coef); CG on the normal equations.

    A V     : out[b, d] = sum_k coef[b, k] * V[flat_idx[b, k], d]
    A^T u   : g[c, d]   = sum_{b, k: flat_idx[b,k]==c} coef[b, k] * u[b, d]
    Both matrix-free, so the [B, n_cells] design matrix is never formed."""
    B, K = flat_idx.shape
    D = target.shape[1]
    dev, dt = target.device, torch.float32

    def Av(V):
        return (coef.unsqueeze(-1) * V[flat_idx]).sum(1)

    def Atu(u):
        g = torch.zeros(n_cells, D, device=dev, dtype=dt)
        g.index_add_(0, flat_idx.reshape(-1), (coef.reshape(-1, 1) * u.repeat_interleave(K, 0)))
        return g

    # Tikhonov ridge, scaled to the system: even with B > n_cells the design matrix is rank-deficient in
    # practice -- rarely addressed cells contribute near-zero columns -- and unregularised CG amplifies
    # those directions until the tables blow up. lam is expressed as a fraction of the MEAN diagonal of
    # A^T A (which is exactly the per-cell sum of squared coefficients), so it is scale-free.
    lam = 0.0
    if ridge > 0:
        diag = torch.zeros(n_cells, device=dev, dtype=dt)
        diag.index_add_(0, flat_idx.reshape(-1), coef.reshape(-1) ** 2)
        lam = ridge * float(diag.mean())

    def normal_op(V):
        out = Atu(Av(V))
        return out + lam * V if lam > 0 else out

    V = torch.zeros(n_cells, D, device=dev, dtype=dt)
    r = Atu(target)                       # residual of the normal equations at V = 0
    p = r.clone()
    rs = float((r * r).sum())
    rs0 = rs
    for _ in range(iters):
        Ap = normal_op(p)
        denom = float((p * Ap).sum())
        if denom <= 0:
            break
        al = rs / denom
        V += al * p
        r -= al * Ap
        rs_new = float((r * r).sum())
        if rs_new <= tol * rs0:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    # how far CG actually got on the normal equations: a fit reported as "optimal" is only optimal if
    # this is small. Reported alongside every irreducible fraction rather than assumed.
    return V, Av(V), (rs / rs0 if rs0 > 0 else 0.0)


# ---------------------------------------------------------------------------------- the probe -------
def lut_checkpoint(model, x, y, *, T, eta_h, arm, tag, cg_iters):
    hs, _, _ = inner_loop(model, x, y, arm, T=T, eta_h=eta_h, alpha=1.0, rho=1.0)
    hs = [h.detach() for h in hs]
    rows = []
    luts = list(model.f_lut) + [model.f_out]
    n_cells = model.f_out.n_tables * model.f_out.table_size
    for i, lut in enumerate(luts):
        readout = (i == len(luts) - 1)
        z = hs[i - 1] if readout else (hs[i] if False else None)
        # layer i maps h_i -> h_{i+1} for interior blocks; the readout maps h_{L-1} -> y (pinned) or yhat
        if readout:
            zin = hs[-1]
            target = y if model.top_pinned() else model.readout(hs[-1])
            resid_target = target                                   # readout is NOT residual: y = a*LUT
        else:
            zin = hs[i]
            target = hs[i + 1]
            resid_target = target - zin                             # the block is h + a*LUT(h)
        with torch.no_grad():
            cur = model.layer(i, zin) if not readout else model.readout(zin)
            r_cur = float((target - cur).pow(2).sum())
        flat_idx, coef = read_structure(lut, zin, model.ai)
        _, fit, cg_rel = lut_lstsq(flat_idx, coef, resid_target, n_cells, iters=cg_iters)
        r_opt = float((resid_target - fit).pow(2).sum())
        base = float(resid_target.pow(2).sum())                     # the do-nothing fit (tables = 0)
        rows.append({'layer': ('readout' if readout else f'L{i}'),
                     'r_current': r_cur, 'r_opt': r_opt, 'r_zero': base,
                     'irreducible_vs_current': r_opt / max(r_cur, 1e-30),
                     'irreducible_vs_zero': r_opt / max(base, 1e-30),
                     'unknowns': n_cells, 'equations': zin.shape[0],
                     'unknowns_per_equation': n_cells / zin.shape[0], 'cg_rel_resid': cg_rel})
    return {'tag': tag, 'layers': rows}


def mlp_checkpoint(model, x, y, *, T, eta_h, mode, tag):
    from pcalm import squared_error  # noqa: F401
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    with torch.no_grad():
        lam = [torch.zeros_like(r) for r in model.residuals(x, hs, y)]
    alpha = 0.0 if mode == 'pc' else 1.0
    for _ in range(T):
        e, r = model.energy(x, y, hs, lam, 1.0, __import__('pcalm').squared_error)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
            if alpha:
                for li, ri in zip(lam, r):
                    li += alpha * ri.detach()
    hs = [h.detach() for h in hs]
    rows = []
    for i in range(model.n_hidden):
        readout = (i == model.n_hidden - 1)
        if readout:
            zin, target = hs[-1], (y if model.clamp_mode == 'pinned' else model.readout(hs[-1]))
            A = model.aL * model.act(zin)
            resid_target = target
            with torch.no_grad():
                cur = model.readout(zin)
        else:
            zin, target = hs[i], hs[i + 1]
            A = model.ai * model.act(zin)
            resid_target = target - zin
            with torch.no_grad():
                cur = model.layer(i, zin)
        with torch.no_grad():
            r_cur = float((target - cur).pow(2).sum())
            sol = torch.linalg.lstsq(A, resid_target).solution
            fit = A @ sol
            r_opt = float((resid_target - fit).pow(2).sum())
            base = float(resid_target.pow(2).sum())
        rows.append({'layer': ('readout' if readout else f'L{i}'),
                     'r_current': r_cur, 'r_opt': r_opt, 'r_zero': base,
                     'irreducible_vs_current': r_opt / max(r_cur, 1e-30),
                     'irreducible_vs_zero': r_opt / max(base, 1e-30),
                     'unknowns': A.shape[1], 'equations': A.shape[0],
                     'unknowns_per_equation': A.shape[1] / A.shape[0], 'cg_rel_resid': 0.0})
    return {'tag': tag, 'layers': rows}


def show(name, cps):
    print(f'\n== {name}')
    print(f'   {"step":>6s} {"layer":>8s} {"||r|| current":>14s} {"||r|| optimal":>14s} {"||r|| at V=0":>13s} '
          f'{"opt/current":>12s} {"opt/zero":>9s} {"unknowns/eq":>12s} {"CG resid":>10s}')
    for cp in cps:
        for r in cp['layers']:
            print(f'   {cp["tag"]:>6s} {r["layer"]:>8s} {r["r_current"]:>14.4e} {r["r_opt"]:>14.4e} '
                  f'{r["r_zero"]:>13.4e} {r["irreducible_vs_current"]:>12.4f} '
                  f'{r["irreducible_vs_zero"]:>9.4f} {r["unknowns_per_equation"]:>12.3f} '
                  f'{r["cg_rel_resid"]:>10.2e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--tables', type=int, default=32)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16384, help='the FIT batch; must overdetermine the fit')
    ap.add_argument('--train-batch', type=int, default=128)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cg-iters', type=int, default=400)
    ap.add_argument('--at', default='1,100,250,500')
    ap.add_argument('--out', default='runs_debug/reachability.json')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    want = sorted(int(v) for v in a.at.split(','))
    xtr, ytr = load('fashion', train=True, device=dev)
    fx, fy = xtr[:a.batch], ytr[:a.batch]
    out = {'cfg': vars(a), 'lut': {}, 'mlp': {}}

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
                cps.append(lut_checkpoint(model, fx, fy, T=a.T, eta_h=eta, arm=arm, tag=str(step),
                                          cg_iters=a.cg_iters))
                print(f'  [LUT {arm}] step {step} done', flush=True)
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
        out['lut'][arm] = cps
        show(f'LUT stack, {arm}', cps)

    for mode in ('pc', 'pcalm'):
        torch.manual_seed(a.seed)
        model = ResidualMLP(xtr.shape[1], 10, a.width, a.depth, device=dev, seed=a.seed)
        model.clamp_mode = 'pinned'
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = TensorLoader(xtr, ytr, batch_size=a.train_batch, seed=a.seed)
        px, py = xtr[:a.train_batch], ytr[:a.train_batch]
        sig = constraint_sigma_max(model, px, py)
        eta = 1.0 / max(sig ** 2, 1e-12)
        cps, it, step = [], iter(loader), 0
        while step <= max(want):
            if step in want:
                cps.append(mlp_checkpoint(model, fx, fy, T=a.T, eta_h=eta, mode=mode, tag=str(step)))
                print(f'  [MLP {mode}] step {step} done', flush=True)
            try:
                bx, by = next(it)
            except StopIteration:
                it = iter(loader)
                bx, by = next(it)
            train_step(model, bx, by, opt, mode, T=a.T, eta_h=eta, grad_clip=1.0)
            step += 1
        out['mlp'][mode] = cps
        show(f'plain MLP, {mode}', cps)

    p = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
