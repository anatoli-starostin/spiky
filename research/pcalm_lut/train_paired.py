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
def _residuals(model, x, hs, arm):
    """The residual family that the arm's energy actually penalises. Arm A's energy contains BOTH the forward
    and the backward residuals (DERIVATION_v2 section 2), so its constraint operator -- and therefore the step
    size -- must cover both; arm B constrains only the forward maps."""
    rf = model.residuals_f(x, hs)
    if arm != 'pcA':
        return rf
    return rf + model.residuals_b(hs, model.readout(hs[-1]))


def sigma_max_A(model, x, arm, iters=25, eps=1e-3):
    """sigma_max of A = d r / d h by power iteration; A v by finite differences (eps well under the median
    smallest margin ~0.15, so the probe stays inside one cell), A^T u by one backward. DERIVATION_v2 section 4."""
    hs = [h.detach().clone() for h in model.init_states(x)]
    with torch.no_grad():
        r0 = _residuals(model, x, hs, arm)
    v = [torch.randn_like(h) for h in hs]
    n = math.sqrt(sum(float(t.pow(2).sum()) for t in v))
    v = [t / n for t in v]
    sigma = 0.0
    for _ in range(iters):
        with torch.no_grad():
            rp = _residuals(model, x, [h + eps * vi for h, vi in zip(hs, v)], arm)
            Av = [(a - b) / eps for a, b in zip(rp, r0)]
        hg = [h.detach().clone().requires_grad_(True) for h in hs]
        r = _residuals(model, x, hg, arm)
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
    lam = [torch.zeros_like(h) for h in hs]
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
    hs, lam, stats = inner_loop(model, x, y, arm, T=T, eta_h=eta_h, alpha=alpha, rho=rho, collect=True)
    if arm == 'pcA':
        e, rf, rb = model.energy_A(x, y, hs)
    else:
        e, rf, rb = model.energy_B(x, y, hs, lam, rho)
    (e / x.shape[0]).backward()
    if arm == 'pcalmB':
        model.recon_R(x, mu=mu).div(x.shape[0]).backward()          # g's own objective, f detached inside
    with torch.no_grad():
        stats['resid_rms'] = float(torch.stack([ri.pow(2).mean() for ri in rf]).mean().sqrt())
        stats['loss_at_h'] = float(0.5 * (model.readout(hs[-1]) - y).pow(2).sum() / x.shape[0])
    return float(e.detach() / x.shape[0]), stats


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    n = a.norm() * b.norm()
    return float(torch.dot(a, b) / n) if n > 0 else float('nan')


def alignment_vs_bp(model, x, y, arm, *, T, eta_h, alpha, rho):
    """Per-parameter cosine to BP's gradient on the same weights/batch, plus the dead-layer count."""
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
    model.zero_grad(set_to_none=True)
    return rows, dead, (g_live / g_tot if g_tot else float('nan'))


# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='bp', choices=['bp', 'pcA', 'pcalmB'])
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
    model = PairedLUTStack(xtr.shape[1], 10, a.width, a.depth, n_tables=a.tables, device=dev, seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = TensorLoader(xtr, ytr, batch_size=a.batch, seed=a.seed)
    probe_x, probe_y = xtr[:a.batch], ytr[:a.batch]
    name = a.name or f'{a.arm}-L{a.depth}-T{T}-s{a.seed}'
    cfg = dict(exp_name=name, arm=a.arm, depth=a.depth, width=a.width, n_tables=a.tables, T=T, steps=a.steps,
               batch=a.batch, lr=a.lr, alpha=a.alpha, rho0=a.rho, rho_max=a.rho_max, mu=a.mu, seed=a.seed,
               dataset=a.dataset, a_i=model.ai, read_tau_init=0.5, nap=8, table_size=256, read_top_n=2,
               _arch_note='PC / PC-ALM over paired forward+backward LightMHL stacks (DERIVATION_v2.md). '
                          'Arm A: symmetric energy, both f and g get gradient from E. Arm B: forward maps as '
                          'constraints with multipliers lam, g outside L_rho (warm start + reconstruction R, '
                          'f detached, feedforward h). BP is the reference arm for the cosine metric.')
    out_dir = os.path.join(HERE, a.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    tags = [t for t in a.tags.split(',') if t]
    tracker = make_tracker(cfg, out_dir, name=name, tags=tags)

    rho, prev_rnorm = a.rho, None
    sigma = sigma_max_A(model, probe_x, a.arm) if a.arm != 'bp' else float('nan')
    eta_h = (float('nan') if a.arm == 'bp' else
             (a.eta_frac / max(sigma ** 2, 1e-12) if a.arm == 'pcA' else eta_h_rule(sigma, rho, a.alpha)))
    print(f'{name} | dev {dev} | sigma_max {sigma:.4f} eta_h {eta_h:.4e} | T {T} | stability '
          f'{eta_h * sigma ** 2 * (2 * rho + a.alpha) if a.arm != "bp" else 0:.3f} < 4', flush=True)
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
        loss, stats = arm_grads(model, x, y, a.arm, T=T, eta_h=eta_h, alpha=a.alpha, rho=rho, mu=a.mu)
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
            sigma = sigma_max_A(model, probe_x, a.arm)
            eta_h = (a.eta_frac / max(sigma ** 2, 1e-12) if a.arm == 'pcA'
                     else eta_h_rule(sigma, rho, a.alpha))
        if step % a.probe_every == 0 or step == 1:
            with torch.no_grad():
                ms = model.margin_stats(model.init_states(probe_x))
                taus = model.log_taus()
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
            if a.arm != 'bp':
                al, dead, gfrac = alignment_vs_bp(model, probe_x, probe_y, a.arm, T=T, eta_h=eta_h,
                                                  alpha=a.alpha, rho=rho)
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
                for i, k in enumerate(sorted(fk)):
                    row[f'align/f_tables_L{i}'] = al[k]
            with torch.no_grad():
                pred = model(xte[:2000]).argmax(-1)
                row['eval/test_acc'] = float((pred == yte[:2000].argmax(-1)).float().mean())
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
