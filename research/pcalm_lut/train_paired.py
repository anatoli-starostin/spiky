"""Train the paired LUT stack under BP / PC-A (symmetric) / PC-ALM-B (constraint). See DERIVATION_v2.md.

    python train_paired.py --arm pcalmB --depth 16 --T 32 --steps 2000 --seed 0
    python train_paired.py --smoke                      # L=4, T=8, 50 steps, all three arms

Instrumentation (first-class): per-layer cosine of the arm's weight gradient to BP's on a fixed probe batch,
dead-layer count, margin quantiles per layer, log_tau trajectory, address-flip rate during relaxation, ||r||
contraction and energy monotonicity across inner steps, sigma_max drift, wall clock.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from data import TensorLoader, load  # noqa: E402
from paired import PairedLUTStack  # noqa: E402
from wandb_setup import make_tracker  # noqa: E402


# ---------------------------------------------------------------------------------------------------------
def _residuals(model, x, y, hs, arm):
    """The residual family that the arm's energy actually penalises. Arm A's energy contains BOTH the forward
    and the backward residuals (DERIVATION_v2 section 2), so its constraint operator -- and therefore the step
    size -- must cover both; arm B constrains only the forward maps. Under pinned clamping the forward family
    carries one extra member, the top residual y - readout(h_{L-1}), which changes A and hence sigma_max: the
    step size must be re-measured, not carried over."""
    rf = model.residuals_f(x, hs, y)
    if arm not in ('pcA', 'hybrid'):
        return rf
    top = y if model.top_pinned() else model.readout(hs[-1])
    return rf + model.residuals_b(hs, top)


def sigma_max_A(model, x, y, arm, iters=25, eps=1e-3):
    """sigma_max of A = d r / d h by power iteration; A v by finite differences (eps well under the median
    smallest margin ~0.15, so the probe stays inside one cell), A^T u by one backward. DERIVATION_v2 section 4."""
    hs = [h.detach().clone() for h in model.init_states(x)]
    with torch.no_grad():
        r0 = _residuals(model, x, y, hs, arm)
    v = [torch.randn_like(h) for h in hs]
    n = math.sqrt(sum(float(t.pow(2).sum()) for t in v))
    v = [t / n for t in v]
    sigma = 0.0
    for _ in range(iters):
        with torch.no_grad():
            rp = _residuals(model, x, y, [h + eps * vi for h, vi in zip(hs, v)], arm)
            Av = [(a - b) / eps for a, b in zip(rp, r0)]
        hg = [h.detach().clone().requires_grad_(True) for h in hs]
        r = _residuals(model, x, y, hg, arm)
        AtAv = torch.autograd.grad(r, hg, grad_outputs=Av)
        n = math.sqrt(sum(float(t.pow(2).sum()) for t in AtAv))
        if n == 0:
            return 0.0
        v = [t / n for t in AtAv]
        sigma = math.sqrt(n)
    return sigma


def eta_h_rule(sigma, rho, alpha):
    """eta_h = 2 / (sigma^2 (2 rho + alpha)); the stability region is eta_h sigma^2 (2 rho + alpha) < 4."""
    return 2.0 / max(sigma ** 2 * (2 * rho + alpha), 1e-12)


def inner_loop(model, x, y, arm, *, T, eta_h, alpha, rho, collect=False):
    """Relaxation. Returns (hs, lam, stats). Arm A uses energy_A (no multipliers); arm B uses energy_B."""
    if arm == 'pcalmB':
        hs = [h.detach().clone().requires_grad_(True) for h in model.init_states_warm(x, y)]
    else:
        hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    # one multiplier per CONSTRAINT, not per state: under pinned clamping the forward family has one more
    # member (the top residual) than there are free states.
    with torch.no_grad():
        lam = [torch.zeros_like(r) for r in model.residuals_f(x, hs, y)]
    a0 = model.addresses(hs) if collect else None
    trace = []
    for t in range(T):
        if arm == 'pcA':
            e, rf, rb = model.energy_A(x, y, hs)
        else:
            e, rf, rb = model.energy_B(x, y, hs, lam, rho)
        g = torch.autograd.grad(e, hs)
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi
            if arm == 'pcalmB' and alpha:
                for li, ri in zip(lam, rf):
                    li += alpha * ri.detach()
            if collect:
                rn = math.sqrt(sum(float(ri.pow(2).sum()) for ri in rf))
                trace.append({'t': t, 'energy': float(e), 'r_norm': rn})
    stats = {}
    if collect:
        with torch.no_grad():
            a1 = model.addresses(hs)
            stats['flip_frac'] = [float((p != q).float().mean()) for p, q in zip(a0, a1)]
            stats['trace'] = trace
            stats['energy_monotone'] = all(b['energy'] <= a['energy'] + 1e-6
                                           for a, b in zip(trace, trace[1:])) if len(trace) > 1 else True
            # ||r|| at the START is ~0 whenever the states are initialised at the forward pass (arm A, and
            # arm B without the warm start), so a last/first ratio is 0/0. Report the absolute norms, and a
            # ratio only against the LARGEST ||r|| reached during the loop (how much of the excursion the
            # relaxation walked back).
            stats['r_first'] = trace[0]['r_norm']
            stats['r_last'] = trace[-1]['r_norm']
            stats['r_peak'] = max(t['r_norm'] for t in trace)
            stats['r_contraction'] = (trace[-1]['r_norm'] / stats['r_peak']) if stats['r_peak'] > 1e-12 else 0.0
    return hs, lam, stats


def arm_grads(model, x, y, arm, *, T, eta_h, alpha, rho, mu=1.0):
    """Populate .grad for one arm on one batch (no optimiser step). Returns (loss_value, stats)."""
    model.zero_grad(set_to_none=True)
    if arm == 'bp':
        loss = 0.5 * (model(x) - y).pow(2).sum() / x.shape[0]
        loss.backward()
        return float(loss.detach()), {}
    if arm == 'hybrid':
        # diagnostic control: the READOUT is trained by backprop, everything else by PC (arm A). If this
        # trains normally the fault is in the readout's gradient path under PC; if it still flatlines the
        # fault is that the relaxed states h_{L-1} carry no usable signal for the readout to read.
        loss = 0.5 * (model(x) - y).pow(2).sum() / x.shape[0]
        loss.backward()
        keep = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                if n.startswith('f_out') and p.grad is not None}
        val, stats = arm_grads(model, x, y, 'pcA', T=T, eta_h=eta_h, alpha=alpha, rho=rho, mu=mu)
        for n, p in model.named_parameters():
            if n in keep:
                p.grad = keep[n]
        stats['bp_loss'] = float(loss.detach())
        return val, stats
    hs, lam, stats = inner_loop(model, x, y, arm, T=T, eta_h=eta_h, alpha=alpha, rho=rho, collect=True)
    if arm == 'pcA':
        e, rf, rb = model.energy_A(x, y, hs)
    else:
        e, rf, rb = model.energy_B(x, y, hs, lam, rho)
    (e / x.shape[0]).backward()
    if arm == 'pcalmB':
        model.recon_R(x, y, mu=mu).div(x.shape[0]).backward()          # g's own objective, f detached inside
    with torch.no_grad():
        stats['resid_rms'] = float(torch.stack([ri.pow(2).mean() for ri in rf]).mean().sqrt())
        stats['loss_at_h'] = float(0.5 * (model.readout(hs[-1]) - y).pow(2).sum() / x.shape[0])
    return float(e.detach() / x.shape[0]), stats


def dropout_fingerprint(model):
    """Identity + content of every pinned dropout mask, so a resample can be DETECTED, not assumed away."""
    fp = [(id(l._head_drop_mask), None if l._head_drop_mask is None else float(l._head_drop_mask.sum()))
          for l in model.luts()]
    for masks in (model._rmask_f, model._rmask_g):
        if masks is not None:
            fp += [(id(m), float(m.sum())) for m in masks]
    return fp


def assert_dropout_pinned(model, x, y, arm, *, T, eta_h, alpha, rho):
    """Fail LOUDLY at run start if dropout is not actually pinned for the whole update.

    Checks, in order: every LUT has a mask after resample_dropout; two forwards inside one update are
    bit-identical; and a full inner loop + weight-gradient pass leaves every mask untouched (this is the
    one that matters -- it is what 'held fixed across all T relaxation steps, identical between the f
    prediction and its vjp, and g's too' actually means)."""
    model.resample_dropout(x.shape[0])
    missing = [i for i, l in enumerate(model.luts()) if l._head_drop_mask is None]
    if model.table_dropout > 0 and missing:
        raise RuntimeError(f'table dropout {model.table_dropout} requested but {len(missing)} LUTs have '
                           f'no pinned mask (indices {missing[:5]}...)')
    if model.residual_dropout > 0 and model._rmask_f is None:
        raise RuntimeError('residual dropout requested but no residual masks were pinned')
    a, b = model(x), model(x)
    if not torch.equal(a, b):
        raise RuntimeError('two forwards inside one update differ: the dropout mask is NOT pinned')
    before = dropout_fingerprint(model)
    arm_grads(model, x, y, arm, T=T, eta_h=eta_h, alpha=alpha, rho=rho)
    model.zero_grad(set_to_none=True)
    if dropout_fingerprint(model) != before:
        raise RuntimeError('a dropout mask changed during the inner loop / backward pass: the energy is '
                           'non-stationary and the vjp differentiates a different network')
    print(f'[dropout] pin verified: table p={model.table_dropout}, residual p={model.residual_dropout}, '
          f'{len(model.luts())} LUT masks held across T={T} inner steps and the vjp', flush=True)


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    n = a.norm() * b.norm()
    return float(torch.dot(a, b) / n) if n > 0 else float('nan')


def _group(name):
    """Which gradient-RMS bucket a parameter belongs to: readout vs interior, and which kind of weight."""
    if name.startswith('g_'):
        return None
    where = 'readout' if name.startswith('f_out') else 'interior'
    if name.endswith('tables'):
        kind = 'tables'
    elif 'log_tau' in name or 'read_tau' in name:
        kind = 'log_tau'
    elif 'anchor' in name:
        kind = 'anchors'
    elif name == 'W1':
        where, kind = 'input', 'W1'
    else:
        kind = 'other'
    return f'{where}_{kind}'


def _grad_rms(names, grads):
    """RMS of the gradient per bucket, pooled over the parameters in it (sum of squares / total numel)."""
    acc = {}
    for n, g in zip(names, grads):
        k = _group(n)
        if k is None or g is None:
            continue
        s, c = acc.get(k, (0.0, 0))
        acc[k] = (s + float(g.pow(2).sum()), c + g.numel())
    return {k: math.sqrt(s / c) if c else float('nan') for k, (s, c) in acc.items()}


def alignment_vs_bp(model, x, y, arm, *, T, eta_h, alpha, rho):
    """Per-parameter cosine to BP's gradient on the same weights/batch, the dead-layer count, and the
    gradient RMS per bucket for BOTH the arm and BP -- the BP pass happens here anyway, so the
    arm/BP magnitude ratio at matched steps is free."""
    names = [n for n, _ in model.named_parameters()]
    arm_grads(model, x, y, 'bp', T=T, eta_h=eta_h, alpha=alpha, rho=rho)
    ref = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    arm_grads(model, x, y, arm, T=T, eta_h=eta_h, alpha=alpha, rho=rho)
    got = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    rows, dead, g_tot, g_live = {}, 0, 0, 0
    for n, a, b in zip(names, got, ref):
        if n.startswith('g_'):                       # g has no BP reference: count coverage instead
            g_tot += 1
            g_live += int(a is not None and float(a.abs().sum()) > 0.0)
            continue
        if a is None or b is None:
            continue
        if float(a.abs().sum()) == 0.0:
            dead += 1
            rows[n] = 0.0
        else:
            rows[n] = cos(a, b)
    rms = {'arm': _grad_rms(names, got), 'bp': _grad_rms(names, ref)}
    model.zero_grad(set_to_none=True)
    return rows, dead, (g_live / g_tot if g_tot else float('nan')), rms


# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='bp', choices=['bp', 'pcA', 'pcalmB', 'hybrid'],
                    help="hybrid: readout by BP, interior by PC-A (a diagnostic control)")
    ap.add_argument('--depth', type=int, default=16)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--tables', type=int, default=16)
    ap.add_argument('--T', type=int, default=None, help='inner steps (default 2L)')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=1.0)
    ap.add_argument('--rho-max', type=float, default=32.0)
    ap.add_argument('--rho-beta', type=float, default=2.0)
    ap.add_argument('--rho-gamma', type=float, default=0.5)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--eta-frac', type=float, default=0.5,
                    help='arm A only: eta_h = eta_frac / sigma_max^2 (the bound is < 1/sigma^2 for E = ||r||^2, whose Hessian is 2 A^T A)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--table-dropout', type=float, default=0.0,
                    help="LightMHL's own head_dropout_rate: whole tables dropped from the n_tables sum "
                         '(one Bernoulli per sample per table on the confidence score, survivors x 1/(1-p))')
    ap.add_argument('--residual-dropout', type=float, default=0.0,
                    help="standard dropout on each block's LUT output, before the a_i scaling")
    ap.add_argument('--clamp', default='pinned', choices=['pinned', 'data'],
                    help='pinned: the target is clamped as the top state, contributing one more forward '
                         'residual and NO separate data term. data: input-only clamping with the data term '
                         '(the mode every run before 2026-09-21 used).')
    ap.add_argument('--train-eval', action='store_true',
                    help='also evaluate on a fixed 2000-row TRAIN subset, for the train/test gap')
    ap.add_argument('--dataset', default='fashion')
    ap.add_argument('--sigma-every', type=int, default=100, help='re-measure sigma_max every N steps')
    ap.add_argument('--probe-every', type=int, default=50, help='alignment / margins / flips cadence')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out-dir', default='runs')
    ap.add_argument('--name', default=None)
    ap.add_argument('--tags', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = a.T or 2 * a.depth
    torch.manual_seed(a.seed)

    xtr, ytr = load(a.dataset, train=True, device=dev)
    xte, yte = load(a.dataset, train=False, device=dev)
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed,
                           table_dropout=a.table_dropout, residual_dropout=a.residual_dropout,
                           clamp_mode=a.clamp)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    probe_x, probe_y = xtr[:a.batch], ytr[:a.batch]
    name = a.name or f'{a.arm}-L{a.depth}-T{T}-s{a.seed}'
    cfg = dict(exp_name=name, arm=a.arm, depth=a.depth, width=a.width, n_tables=a.tables, T=T, steps=a.steps,
               batch=a.batch, lr=a.lr, alpha=a.alpha, rho0=a.rho, rho_max=a.rho_max, mu=a.mu, seed=a.seed,
               dataset=a.dataset, a_i=model.ai, read_tau_init=0.5, nap=8, table_size=256, read_top_n=2,
               table_dropout=a.table_dropout, residual_dropout=a.residual_dropout,
               align_layer_order='numeric',   # absent -> that run's align/f_tables_L{i} are lexicographic
               clamp=a.clamp,
               _arch_note='PC / PC-ALM over paired forward+backward LightMHL stacks (DERIVATION_v2.md). '
                          'Arm A: symmetric energy, both f and g get gradient from E. Arm B: forward maps as '
                          'constraints with multipliers lam, g outside L_rho (warm start + reconstruction R, '
                          'f detached, feedforward h). BP is the reference arm for the cosine metric.')
    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    tags = [t for t in a.tags.split(',') if t]
    tracker = make_tracker(cfg, out_dir, name=name, tags=tags)

    rho, prev_rnorm = a.rho, None
    inner_arm = 'pcA' if a.arm == 'hybrid' else a.arm
    sigma = sigma_max_A(model, probe_x, probe_y, inner_arm) if a.arm != 'bp' else float('nan')
    eta_h = (float('nan') if a.arm == 'bp' else
             (a.eta_frac / max(sigma ** 2, 1e-12) if inner_arm == 'pcA' else eta_h_rule(sigma, rho, a.alpha)))
    print(f'{name} | dev {dev} | sigma_max {sigma:.4f} eta_h {eta_h:.4e} | T {T} | stability '
          f'{eta_h * sigma ** 2 * (2 * rho + a.alpha) if a.arm != "bp" else 0:.3f} < 4', flush=True)
    if a.table_dropout > 0 or a.residual_dropout > 0:
        assert_dropout_pinned(model, probe_x, probe_y, a.arm, T=T,
                              eta_h=(0.0 if a.arm == 'bp' else eta_h), alpha=a.alpha, rho=rho)
    hist, step, t0 = [], 0, time.time()
    it = iter(loader)
    while step < a.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        step += 1
        ts = time.time()
        # ONE dropout mask for the whole update -- for the PC arms it must not move across the T inner
        # steps or between the energy's prediction and its vjp (test_dropout_mask_stable.py asserts it).
        model.resample_dropout(x.shape[0])
        fp = dropout_fingerprint(model) if (a.table_dropout or a.residual_dropout) else None
        loss, stats = arm_grads(model, x, y, a.arm, T=T, eta_h=eta_h, alpha=a.alpha, rho=rho, mu=a.mu)
        if fp is not None and dropout_fingerprint(model) != fp:
            raise RuntimeError(f'step {step}: a dropout mask changed during the update -- the mask must '
                               f'be pinned for all {T} inner steps and the vjp')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.zero_grad(set_to_none=True)
        dt = time.time() - ts
        row = {'train/loss': loss, 'train/s_per_step': dt, 'train/rho': rho, 'train/eta_h': eta_h,
               'train/sigma_max': sigma}
        if stats:
            row['train/resid_rms'] = stats.get('resid_rms', float('nan'))
            row['train/r_contraction'] = stats.get('r_contraction', float('nan'))
            row['train/r_peak'] = stats.get('r_peak', float('nan'))
            row['train/r_last'] = stats.get('r_last', float('nan'))
            row['train/energy_monotone'] = float(stats.get('energy_monotone', True))
            row['train/loss_at_h'] = stats.get('loss_at_h', float('nan'))
            fl = stats.get('flip_frac')
            if fl:
                row['flips/mean'] = sum(fl) / len(fl)
                for i, v in enumerate(fl):
                    row[f'flips/L{i}'] = v
        # rho schedule (arm B): raise rho if the residual did not contract, then re-derive eta_h
        if a.arm == 'pcalmB' and stats:
            rn = stats.get('resid_rms', float('nan'))
            if prev_rnorm is not None and rn > a.rho_gamma * prev_rnorm:
                rho = min(a.rho_beta * rho, a.rho_max)
                eta_h = eta_h_rule(sigma, rho, a.alpha)
            prev_rnorm = rn
        if a.arm != 'bp' and step % a.sigma_every == 0:
            sigma = sigma_max_A(model, probe_x, probe_y, inner_arm)
            eta_h = (a.eta_frac / max(sigma ** 2, 1e-12) if inner_arm == 'pcA'
                     else eta_h_rule(sigma, rho, a.alpha))
        if step % a.probe_every == 0 or step == 1:
            model.clear_dropout()          # every diagnostic measures the FULL network, not a dropout sample
            with torch.no_grad():
                pstates = model.init_states(probe_x)
                ms = model.margin_stats(pstates)
                taus = model.log_taus()
                tn = model.table_norms()
                br = model.branch_ratios(pstates)
            # collapse diagnostics: arm A's trivial solution is tables -> 0 with every block the identity,
            # which shows up as a monotone decay of both of these.
            for i, v in enumerate(tn):
                row[f'norm/f_tables_L{i}'] = v
            row['norm/f_tables_mean'] = sum(tn) / len(tn)
            for i, v in enumerate(br):
                row[f'ident/branch_ratio_L{i}'] = v
            row['ident/branch_ratio_mean'] = sum(br) / len(br)
            for i, m in enumerate(ms):
                row[f'margin/m_min_p50_L{i}'] = m['m_min_q'][1]
                row[f'margin/m_min_p10_L{i}'] = m['m_min_q'][0]
                row[f'margin/m_min_p90_L{i}'] = m['m_min_q'][2]
                row[f'margin/m_sum_p50_L{i}'] = m['m_sum_q'][1]
            row['margin/m_min_p50_mean'] = sum(m['m_min_q'][1] for m in ms) / len(ms)
            for i, v in enumerate(taus['f']):
                row[f'tau/f_L{i}'] = v
            for i, v in enumerate(taus['g']):
                row[f'tau/g_L{i}'] = v
            # scale of what the readout emits, against the scale it has to reach. A one-hot target has
            # RMS 1/sqrt(C); if the readout's output RMS sits far below that, the layer is emitting
            # nothing whatever its gradient DIRECTION is.
            with torch.no_grad():
                out = model(probe_x)
                row['out/readout_rms'] = float(out.pow(2).mean().sqrt())
                row['out/target_rms'] = float(probe_y.pow(2).mean().sqrt())
                row['out/readout_over_target'] = row['out/readout_rms'] / max(row['out/target_rms'], 1e-12)
            if a.arm == 'bp':
                # BP is the denominator of the magnitude comparison, so it logs the same buckets
                pnames = [n for n, _ in model.named_parameters()]
                arm_grads(model, probe_x, probe_y, 'bp', T=T, eta_h=0.0, alpha=a.alpha, rho=rho)
                for k, v in _grad_rms(pnames, [p.grad for p in model.parameters()]).items():
                    row[f'grad/{k}_rms'] = v
                model.zero_grad(set_to_none=True)
            if a.arm != 'bp':
                al, dead, gfrac, rms = alignment_vs_bp(model, probe_x, probe_y, a.arm, T=T, eta_h=eta_h,
                                                       alpha=a.alpha, rho=rho)
                for k, v in rms['arm'].items():
                    row[f'grad/{k}_rms'] = v
                for k, v in rms['bp'].items():
                    row[f'grad/{k}_rms_bp'] = v
                    if v > 0:
                        row[f'grad/{k}_ratio'] = rms['arm'].get(k, float('nan')) / v
                tabs = [v for k, v in al.items() if k.startswith('f_lut') or k.startswith('f_out')]
                gtabs = [v for k, v in al.items() if k.startswith('g_lut') or k.startswith('g_out')]
                row['align/f_mean'] = sum(tabs) / max(len(tabs), 1)
                row['align/g_mean'] = sum(gtabs) / max(len(gtabs), 1) if gtabs else float('nan')
                row['align/dead_layers'] = dead
                row['align/g_grad_frac'] = gfrac
                # NOTE: under BP the backward layers g receive NO gradient at all (they are not on the
                # forward path), so they have no reference gradient: alignment is defined for f only, and
                # g's coverage is reported by align/g_grad_frac instead.
                fk = [k for k in al if (k.startswith('f_lut') or k.startswith('f_out')) and k.endswith('tables')]
                # sort NUMERICALLY, with the readout last. Plain sorted() is lexicographic, which orders
                # f_lut.10 before f_lut.2 and silently scrambles the depth axis of this metric (runs
                # without cfg['align_layer_order'] == 'numeric' were logged that way; analyze_sweep.py
                # remaps them).
                def _depth(k):
                    return (1, 0) if k.startswith('f_out') else (0, int(k.split('.')[1]))
                for i, k in enumerate(sorted(fk, key=_depth)):
                    row[f'align/f_tables_L{i}'] = al[k]
            with torch.no_grad():
                pred = model(xte[:2000]).argmax(-1)
                row['eval/test_acc'] = float((pred == yte[:2000].argmax(-1)).float().mean())
                if a.train_eval:
                    # a FIXED train subset, the same 2000 rows every probe, so train-minus-test is a
                    # generalisation gap and not minibatch noise (needed for the dropout study)
                    ptr = model(xtr[:2000]).argmax(-1)
                    row['eval/train_acc'] = float((ptr == ytr[:2000].argmax(-1)).float().mean())
                    row['eval/gap'] = row['eval/train_acc'] - row['eval/test_acc']
                    row['eval/train_loss_full'] = float(0.5 * (model(xtr[:2000]) - ytr[:2000]).pow(2).sum(-1).mean())
            print(f'  step {step:5d} obj {loss:.4f} data {row.get("train/loss_at_h", loss):.4f} '
                  f'acc {row["eval/test_acc"]:.4f} '
                  f'{"| flips %.4f " % row.get("flips/mean", float("nan")) if stats else ""}'
                  f'| m_min_p50 {row["margin/m_min_p50_mean"]:.4f} | {dt * 1e3:.0f} ms/step', flush=True)
        hist.append({'step': step, **row})
        tracker.log(row, step=step)
    wall = time.time() - t0
    summary = {'wall_s': wall, 'final_loss': hist[-1]['train/loss'],
               'test_acc': hist[-1].get('eval/test_acc', float('nan'))}
    tracker.finish(summary=summary)
    json.dump({'cfg': cfg, 'hist': hist, 'summary': summary}, open(os.path.join(out_dir, 'run.json'), 'w'),
              indent=1)
    print(f'{name} done: {wall:.1f}s, final loss {summary["final_loss"]:.4f}, '
          f'test acc {summary["test_acc"]:.4f} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
