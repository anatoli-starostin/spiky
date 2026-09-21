"""Read the L=4 pinned-clamping smoke (runs_pinned/) and the finished L=16 no-pin runs (runs/), and
answer the two questions of task abc84212:

  (a) COLLAPSE. Arm A has a trivial solution -- every residual block the identity, all tables zero. Does
      it take it under the old input-only clamping, and does pinning the target make it infeasible?
      Evidence: the table-norm trajectory and the residual-branch ratio ||a_i LUT_i(h)||/||h||.
  (b) Does the negative per-layer cosine to BP survive pinning? Collapse was the leading explanation.

The L=16 runs never logged table norms or branch ratios (the instrument did not exist yet), so for those
only ||r|| over training can be reported; the collapse verdict comes from the matched L=4 controls, which
run the same code under both clampings.

Usage: python3 analyze_pinned.py [--plot]
"""
import argparse
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = [('pin-bp-L4', 'BP pinned'), ('pin-pcA-L4', 'PC-A pinned'), ('pin-pcalmB-L4', 'PC-ALM-B pinned'),
         ('nopin-pcA-L4', 'PC-A no-pin'), ('nopin-pcalmB-L4', 'PC-ALM-B no-pin')]


def load(d, n):
    p = os.path.join(HERE, d, n, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def series(h, k):
    return [(r['step'], r[k]) for r in h if k in r and isinstance(r[k], (int, float)) and not math.isnan(r[k])]


def last(h, k, default=float('nan')):
    s = series(h, k)
    return s[-1][1] if s else default


def first(h, k, default=float('nan')):
    s = series(h, k)
    return s[0][1] if s else default


def per_layer(row, prefix):
    out, i = [], 0
    while f'{prefix}{i}' in row:
        out.append(row[f'{prefix}{i}'])
        i += 1
    return out


def headline(runs):
    print('\n== L=4 SMOKE: N=64, 32 tables, T=8, 2000 steps, seed 0, table dropout p=0.25 (mask pinned)')
    print('   data loss = 1/2||yhat-y||^2 per sample at the states the arm fits; the OBJECTIVE is not')
    print('   comparable across clampings (pinned has no data term and arm B has no objective at all).')
    print(f'\n   {"run":18s} {"clamp":7s} {"data loss":>10s} {"train acc":>10s} {"test acc":>9s} '
          f'{"gap":>7s} {"ms/step":>8s} {"objective":>11s}')
    for name, label in SMOKE:
        d = runs.get(name)
        if not d:
            print(f'   {label:18s}   (missing)')
            continue
        h = d['hist']
        print(f'   {label:18s} {d["cfg"].get("clamp", "data"):7s} '
              f'{last(h, "train/loss_at_h", last(h, "train/loss")):>10.4f} '
              f'{last(h, "eval/train_acc"):>10.4f} {last(h, "eval/test_acc"):>9.4f} '
              f'{last(h, "eval/gap"):>7.4f} '
              f'{1e3 * st.median([v for _, v in series(h, "train/s_per_step")]):>8.1f} '
              f'{last(h, "train/loss"):>11.4f}')


def collapse(runs):
    print('\n== (a) COLLAPSE: table norms and the residual-branch ratio')
    print('   tables = RMS table entry, mean over forward LUTs. branch = ||a_i LUT_i(h)||/||h||, mean over')
    print('   interior blocks. The trivial solution is BOTH decaying monotonically toward zero.')
    print(f'\n   {"run":18s} {"tables 1st":>11s} {"min":>9s} {"last":>9s} {"last/1st":>9s} {"monotone?":>10s} '
          f'{"branch 1st":>11s} {"branch last":>12s} {"last/1st":>9s}')
    for name, label in SMOKE:
        d = runs.get(name)
        if not d:
            continue
        h = d['hist']
        tn = series(h, 'norm/f_tables_mean')
        br = series(h, 'ident/branch_ratio_mean')
        if not tn:
            print(f'   {label:18s}   (no collapse instrumentation in this run)')
            continue
        v = [q for _, q in tn]
        # "monotone decay" as the trivial solution would give: fraction of probes that went down
        down = sum(1 for a, b in zip(v, v[1:]) if b < a) / max(len(v) - 1, 1)
        b0, b1 = (br[0][1], br[-1][1]) if br else (float('nan'), float('nan'))
        print(f'   {label:18s} {v[0]:>11.4f} {min(v):>9.4f} {v[-1]:>9.4f} {v[-1] / v[0]:>9.3f} '
              f'{down:>9.0%} {b0:>11.4f} {b1:>12.4f} {b1 / b0 if b0 else float("nan"):>9.3f}')


def relaxation(runs):
    print('\n== ||r|| across the inner loop, and energy monotonicity (PC arms only)')
    # r_first is computed but was never added to the logged row, so it is not available here; the shape
    # of the trajectory is read from peak vs last instead.
    print(f'   {"run":18s} {"r_peak":>10s} {"r_last":>10s} {"last/peak":>10s} '
          f'{"resid_rms":>10s} {"monotone":>9s} {"sigma 1st":>10s} {"sigma last":>11s} {"rho":>5s}')
    for name, label in SMOKE:
        d = runs.get(name)
        if not d or d['cfg']['arm'] == 'bp':
            continue
        h = d['hist']
        mono = [v for _, v in series(h, 'train/energy_monotone')]
        print(f'   {label:18s} {last(h, "train/r_peak"):>10.3e} '
              f'{last(h, "train/r_last"):>10.3e} {last(h, "train/r_contraction"):>10.3f} '
              f'{last(h, "train/resid_rms"):>10.4f} {(st.mean(mono) if mono else float("nan")):>9.2f} '
              f'{first(h, "train/sigma_max"):>10.3f} {last(h, "train/sigma_max"):>11.3f} '
              f'{last(h, "train/rho"):>5.1f}')


def routing(runs):
    print('\n== margins, tau and address flips')
    print(f'   {"run":18s} {"m_min 1st":>10s} {"m_min last":>11s} {"growth":>7s} {"tau 1st":>8s} '
          f'{"tau last":>9s} {"flips 1st":>10s} {"flips last":>11s} {"flips x":>8s}')
    for name, label in SMOKE:
        d = runs.get(name)
        if not d:
            continue
        h = d['hist']
        m = series(h, 'margin/m_min_p50_mean')
        f = series(h, 'flips/mean')
        rows = [r for r in h if 'tau/f_L0' in r]
        t0 = st.mean(per_layer(rows[0], 'tau/f_L'))
        t1 = st.mean(per_layer(rows[-1], 'tau/f_L'))
        fx = (f[-1][1] / f[0][1]) if (f and f[0][1]) else float('nan')
        print(f'   {label:18s} {m[0][1]:>10.4f} {m[-1][1]:>11.4f} {m[-1][1] / m[0][1]:>7.2f} '
              f'{t0:>8.4f} {t1:>9.4f} '
              + (f'{f[0][1]:>10.4f} {f[-1][1]:>11.4f} {fx:>8.1f}' if f
                 else f'{"n/a":>10s} {"n/a":>11s} {"n/a":>8s}'))


def alignment(runs):
    print('\n== (b) per-layer cosine to BP, input -> readout (does the anti-alignment survive pinning?)')
    for name, label in SMOKE:
        d = runs.get(name)
        if not d or d['cfg']['arm'] == 'bp':
            continue
        h = d['hist']
        rows = [r for r in h if 'align/f_tables_L0' in r]
        if not rows:
            continue
        print(f'\n   {label}')
        print(f'    {"step":>6s} {"mean":>7s} {"min":>7s} {"#neg":>5s}   per-layer')
        for r in [rows[0], rows[len(rows) // 2], rows[-1]]:
            v = per_layer(r, 'align/f_tables_L')
            print(f'    {r["step"]:>6d} {st.mean(v):>7.3f} {min(v):>7.3f} '
                  f'{sum(1 for q in v if q < 0):>5d}   ' + ' '.join(f'{q:+.2f}' for q in v))
        neg = [sum(1 for q in per_layer(r, 'align/f_tables_L') if q < 0) for r in rows]
        print(f'    layers with cos<0: first {neg[0]}, max {max(neg)}, last {neg[-1]}, '
              f'mean {st.mean(neg):.2f} of {len(per_layer(rows[0], "align/f_tables_L"))}')


def l16_norms():
    print('\n== the FINISHED L=16 no-pin runs: ||r|| over training')
    print('   (table norms and branch ratios do not exist for these -- the instrument was added after they')
    print('   ran, and no checkpoints were kept, so the collapse verdict comes from the L=4 controls above.)')
    print(f'\n   {"run":30s} {"r_peak 1st":>11s} {"r_peak last":>12s} {"r_last last":>12s} '
          f'{"contraction":>12s} {"resid_rms 1st":>14s} {"resid_rms last":>15s} {"monotone":>9s}')
    d = os.path.join(HERE, 'runs')
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n, 'run.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if r['cfg'].get('arm') in (None, 'bp') or r['cfg'].get('depth') != 16:
            continue
        h = r['hist']
        mono = [v for _, v in series(h, 'train/energy_monotone')]
        print(f'   {n:30s} {first(h, "train/r_peak"):>11.3e} {last(h, "train/r_peak"):>12.3e} '
              f'{last(h, "train/r_last"):>12.3e} {last(h, "train/r_contraction"):>12.3f} '
              f'{first(h, "train/resid_rms"):>14.4f} {last(h, "train/resid_rms"):>15.4f} '
              f'{(st.mean(mono) if mono else float("nan")):>9.2f}')


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    col = {'pin-bp-L4': '#444444', 'pin-pcA-L4': '#1f77b4', 'pin-pcalmB-L4': '#d62728',
           'nopin-pcA-L4': '#1f77b4', 'nopin-pcalmB-L4': '#d62728'}
    for name, label in SMOKE:
        d = runs.get(name)
        if not d:
            continue
        h, c = d['hist'], col[name]
        ls = '--' if name.startswith('nopin') else '-'
        for j, key in enumerate(('eval/test_acc', 'norm/f_tables_mean', 'ident/branch_ratio_mean',
                                 'margin/m_min_p50_mean')):
            s = series(h, key)
            if s:
                ax[j].plot([q for q, _ in s], [q for _, q in s], color=c, ls=ls, alpha=.85, label=label)
    for j, t in enumerate(('test accuracy', 'table norm (collapse)', 'branch ratio (identity)',
                           'median smallest margin')):
        ax[j].set(xlabel='step', title=t)
        ax[j].legend(fontsize=7)
        ax[j].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'\nwrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    runs = {n: load('runs_pinned', n) for n, _ in SMOKE}
    headline(runs)
    collapse(runs)
    relaxation(runs)
    routing(runs)
    alignment(runs)
    l16_norms()
    if a.plot:
        plot(runs, os.path.join(HERE, 'runs_pinned', 'pinned_smoke.png'))


if __name__ == '__main__':
    main()
