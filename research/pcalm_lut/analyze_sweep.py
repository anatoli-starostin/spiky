"""Read runs/<name>/run.json for the L=16 three-arm sweep and answer the two questions the sweep was
launched to answer, plus the health checks.

  chase 1  NEGATIVE MIDDLE-LAYER COSINE.  At L=4 / T=8 the per-layer cosine of the PC update to the BP
           gradient was negative in the middle of the stack (-0.59 / -0.58 / +0.98).  Does that survive
           at L=16 / T=32, is it a transient of early training, and does it decay as T grows?
  chase 2  MARGIN GROWTH / ROUTING COLLAPSE.  Margins grow under BP (the network sharpens its routing).
           If they grow the same way under the PC arms, the address-flip rate during the relaxation --
           the thing that makes the relaxation an ADDRESS SEARCH rather than a value search -- goes to
           zero and the mechanism switches itself off.  Flip rate is only defined for the PC arms (under
           BP the states never move during a step, so there is nothing to flip); margins are logged for
           all three, so margins are the shared axis.

Usage: python3 analyze_sweep.py [--plot]
"""
import argparse
import json
import math
import os
import statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
ARMS = [('bp', 'BP'), ('pcA', 'PC-A'), ('pcalmB', 'PC-ALM-B')]


def load(name):
    p = os.path.join(R, name, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else None


def series(h, key):
    return [(r['step'], r[key]) for r in h if key in r and not math.isnan(r[key])]


def last(h, key, default=float('nan')):
    s = series(h, key)
    return s[-1][1] if s else default


def per_layer(row, prefix, nl):
    return [row[f'{prefix}{i}'] for i in range(nl) if f'{prefix}{i}' in row]


def fmt(v, w=7, p=4):
    return f'{v:>{w}.{p}f}' if isinstance(v, float) and not math.isnan(v) else f'{"n/a":>{w}}'


def headline(runs):
    print('\n== HEADLINE (2000 steps, Fashion-MNIST, L=16 N=64 32 tables 256 cells, T=2L=32)')
    print(f'{"arm":10s} {"seeds":>5s} {"test acc mean+-sd":>20s} {"final loss":>11s} '
          f'{"s/step":>8s} {"wall":>8s}')
    for arm, label in ARMS:
        accs, losses, sps, walls = [], [], [], []
        for s in (0, 1, 2):
            d = runs.get((arm, s, 32))
            if not d:
                continue
            accs.append(last(d['hist'], 'eval/test_acc'))
            losses.append(last(d['hist'], 'train/loss'))
            sps.append(st.median([v for _, v in series(d['hist'], 'train/s_per_step')]))
            walls.append(d['summary']['wall_s'])
        if not accs:
            print(f'{label:10s} {"-":>5s}   (no runs yet)')
            continue
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        print(f'{label:10s} {len(accs):>5d} {st.mean(accs):>13.4f} +-{sd:.4f} '
              f'{st.mean(losses):>11.4f} {st.mean(sps):>8.3f} {st.mean(walls) / 60:>7.1f}m')


def chase1(runs, Ts):
    print('\n== CHASE 1: per-layer cosine of the PC update to the BP gradient (forward tables)')
    print('   negative = the arm moves that layer AGAINST the BP descent direction.')
    for arm, label in ARMS[1:]:
        for s in (0, 1, 2):
            d = runs.get((arm, s, 32))
            if not d:
                continue
            h = d['hist']
            nl = d['cfg']['depth']
            rows = [r for r in h if 'align/f_tables_L0' in r]
            if not rows:
                continue
            print(f'\n  {label} seed {s} (T={d["cfg"]["T"]})')
            print(f'   {"step":>6s} {"mean":>7s} {"min":>7s} {"argmin":>6s} {"#neg":>5s}   per-layer')
            picks = [rows[0], rows[len(rows) // 4], rows[len(rows) // 2], rows[-1]]
            for r in picks:
                v = per_layer(r, 'align/f_tables_L', nl)
                mn = min(v)
                print(f'   {r["step"]:>6d} {st.mean(v):>7.3f} {mn:>7.3f} {v.index(mn):>6d} '
                      f'{sum(1 for x in v if x < 0):>5d}   '
                      + ' '.join(f'{x:+.2f}' for x in v))
            neg = [sum(1 for x in per_layer(r, 'align/f_tables_L', nl) if x < 0) for r in rows]
            print(f'   layers with cos<0: first probe {neg[0]}, max {max(neg)}, last {neg[-1]}, '
                  f'mean over training {st.mean(neg):.2f} of {nl}')
    if Ts:
        print('\n  T-dependence at seed 0 (does more relaxation remove the negative cosine?)')
        print(f'   {"arm":10s} {"T":>4s} {"mean cos (last)":>16s} {"min cos (last)":>15s} '
              f'{"#neg (last)":>12s} {"#neg (mean)":>12s} {"test acc":>9s}')
        for arm, label in ARMS[1:]:
            for T in sorted(Ts):
                d = runs.get((arm, 0, T))
                if not d:
                    continue
                h, nl = d['hist'], d['cfg']['depth']
                rows = [r for r in h if 'align/f_tables_L0' in r]
                if not rows:
                    continue
                v = per_layer(rows[-1], 'align/f_tables_L', nl)
                neg = [sum(1 for x in per_layer(r, 'align/f_tables_L', nl) if x < 0) for r in rows]
                print(f'   {label:10s} {T:>4d} {st.mean(v):>16.3f} {min(v):>15.3f} '
                      f'{sum(1 for x in v if x < 0):>12d} {st.mean(neg):>12.2f} '
                      f'{last(h, "eval/test_acc"):>9.4f}')


def chase2(runs):
    print('\n== CHASE 2: margin growth vs address-flip rate')
    print('   m_min_p50 = median over batch+heads of the SMALLEST anchor margin in a layer (the margin')
    print('   that decides the neighbour blend).  flips/mean = fraction of addresses that changed during')
    print('   the relaxation, averaged over layers.  Flips are undefined for BP (states never move).')
    print(f'\n   {"arm":10s} {"seed":>4s} {"T":>4s} {"m_min first":>12s} {"m_min last":>11s} '
          f'{"growth x":>9s} {"flips first":>12s} {"flips last":>11s} {"flips x":>8s} {"corr":>6s}')
    for arm, label in ARMS:
        for key, d in sorted(runs.items()):
            if key[0] != arm:
                continue
            h = d['hist']
            m = series(h, 'margin/m_min_p50_mean')
            f = series(h, 'flips/mean')
            if not m:
                continue
            g = m[-1][1] / m[0][1] if m[0][1] else float('nan')
            if f:
                fr = f[-1][1] / f[0][1] if f[0][1] else float('nan')
                # correlation between margin and flip rate over the probes they share
                md = dict(m)
                pairs = [(md[s], v) for s, v in f if s in md]
                if len(pairs) > 3:
                    xs = [p[0] for p in pairs]
                    ys = [p[1] for p in pairs]
                    mx, my = st.mean(xs), st.mean(ys)
                    num = sum((a - mx) * (b - my) for a, b in pairs)
                    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
                    corr = num / den if den else float('nan')
                else:
                    corr = float('nan')
            else:
                fr, corr = float('nan'), float('nan')
            print(f'   {label:10s} {key[1]:>4d} {d["cfg"]["T"]:>4d} {m[0][1]:>12.4f} {m[-1][1]:>11.4f} '
                  f'{g:>9.2f} {fmt(f[0][1] if f else float("nan"), 12)} '
                  f'{fmt(f[-1][1] if f else float("nan"), 11)} {fmt(fr, 8, 2)} {fmt(corr, 6, 2)}')
    print('\n   tau (the read temperature; larger tau = softer blend = more neighbour mixing)')
    print(f'   {"arm":10s} {"seed":>4s} {"T":>4s} {"tau_f first":>12s} {"tau_f last":>11s} '
          f'{"tau_g last":>11s}')
    for arm, label in ARMS:
        for key, d in sorted(runs.items()):
            if key[0] != arm:
                continue
            h, nl = d['hist'], d['cfg']['depth']
            rows = [r for r in h if 'tau/f_L0' in r]
            if not rows:
                continue
            t0 = per_layer(rows[0], 'tau/f_L', nl)
            t1 = per_layer(rows[-1], 'tau/f_L', nl)
            tg = per_layer(rows[-1], 'tau/g_L', nl)
            print(f'   {label:10s} {key[1]:>4d} {d["cfg"]["T"]:>4d} {st.mean(t0):>12.4f} '
                  f'{st.mean(t1):>11.4f} {fmt(st.mean(tg) if tg else float("nan"), 11)}')


def health(runs):
    print('\n== HEALTH (PC arms)')
    print(f'   {"arm":10s} {"seed":>4s} {"T":>4s} {"monotone":>9s} {"r_peak":>10s} {"r_last":>10s} '
          f'{"contract":>9s} {"sigma 1st":>10s} {"sigma last":>11s} {"eta_h":>9s} {"rho":>5s} '
          f'{"dead":>5s} {"g_frac":>7s}')
    for arm, label in ARMS[1:]:
        for key, d in sorted(runs.items()):
            if key[0] != arm:
                continue
            h = d['hist']
            mono = [v for _, v in series(h, 'train/energy_monotone')]
            sg = series(h, 'train/sigma_max')
            print(f'   {label:10s} {key[1]:>4d} {d["cfg"]["T"]:>4d} '
                  f'{(st.mean(mono) if mono else float("nan")):>9.3f} '
                  f'{last(h, "train/r_peak"):>10.3e} {last(h, "train/r_last"):>10.3e} '
                  f'{last(h, "train/r_contraction"):>9.3f} {sg[0][1]:>10.3f} {sg[-1][1]:>11.3f} '
                  f'{last(h, "train/eta_h"):>9.4f} {last(h, "train/rho"):>5.1f} '
                  f'{last(h, "align/dead_layers"):>5.0f} {last(h, "align/g_grad_frac"):>7.3f}')


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {'bp': '#444444', 'pcA': '#1f77b4', 'pcalmB': '#d62728'}
    for key, d in sorted(runs.items()):
        arm, seed = key[0], key[1]
        if d['cfg']['T'] not in (32,) and arm != 'bp':
            continue
        h, c = d['hist'], colors[arm]
        lab = dict(ARMS)[arm] if seed == 0 else None
        m = series(h, 'margin/m_min_p50_mean')
        ax[0].plot([s for s, _ in m], [v for _, v in m], color=c, alpha=0.8, label=lab)
        f = series(h, 'flips/mean')
        if f:
            ax[1].plot([s for s, _ in f], [v for _, v in f], color=c, alpha=0.8, label=lab)
            md = dict(m)
            pairs = sorted((md[s], v) for s, v in f if s in md)
            ax[2].plot([p[0] for p in pairs], [p[1] for p in pairs], '.', color=c, ms=3, label=lab)
        a = series(h, 'eval/test_acc')
    ax[0].set(xlabel='step', ylabel='median smallest margin', title='margin growth')
    ax[1].set(xlabel='step', ylabel='address-flip fraction', title='address search during relaxation')
    ax[2].set(xlabel='median smallest margin', ylabel='address-flip fraction',
              title='flips vs margin (the collapse test)')
    for a_ in ax:
        a_.legend(fontsize=8)
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'\nwrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    runs, Ts = {}, set()
    for name in sorted(os.listdir(R)):
        d = load(name)
        if not d or 'arm' not in d.get('cfg', {}) or d['cfg'].get('depth') != 16:
            continue
        cfg = d['cfg']
        runs[(cfg['arm'], cfg['seed'], cfg['T'])] = d
        if cfg['arm'] != 'bp':
            Ts.add(cfg['T'])
    # the T=2L runs are the main sweep; index them by (arm, seed) too
    main_runs = {k: v for k, v in runs.items() if k[2] == 32 or k[0] == 'bp'}
    print(f'loaded {len(runs)} runs: ' + ', '.join(f'{k[0]}/s{k[1]}/T{k[2]}' for k in sorted(runs)))
    headline(main_runs)
    chase1(runs, Ts - {32})
    chase2(main_runs)
    health(main_runs)
    if a.plot:
        plot(main_runs, os.path.join(R, 'sweep_L16.png'))


if __name__ == '__main__':
    main()
