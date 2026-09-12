"""exp_g_0247 (g, beta, gamma learnable) vs exp_g_0248 (g frozen at 0): per-layer beta / gamma
trajectories side by side. Read-only; reads the two metrics.csv files.

    python compare_gamma_0247_0248.py [--png out.png]

Prints, per layer: the value at selected eval steps for both runs, the final difference 0248 - 0247,
the step at which gamma first exceeded 1.1, the late drift (14K->16K and 12K->14K), and the correlation
of the two runs' gamma trajectories across eval steps. Plots gamma and beta overlaid.
"""
import csv
import os
import sys

RC = os.path.expanduser('~/projects/spiky/experiments/ffn_replacement/runs_corrected')
RUNS = {'0247 (g learnable)': 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1',
        '0248 (g frozen)': 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1'}
L = 6
SHOW = (1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)

data = {}
for name, rd in RUNS.items():
    rows = list(csv.DictReader(open(os.path.join(RC, rd, 'metrics.csv'))))
    data[name] = {'step': [int(r['step']) for r in rows],
                  **{k: [[float(r[f'lm_{k}_L{i}']) for r in rows] for i in range(L)] for k in ('g', 'beta', 'gamma')}}
a, b = data['0247 (g learnable)'], data['0248 (g frozen)']
steps = [s for s in a['step'] if s in b['step']]
ia = {s: j for j, s in enumerate(a['step'])}
ib = {s: j for j, s in enumerate(b['step'])}


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((u - mx) ** 2 for u in x) ** .5
    sy = sum((v - my) ** 2 for v in y) ** .5
    return sum((u - mx) * (v - my) for u, v in zip(x, y)) / (sx * sy) if sx and sy else float('nan')


for k in ('gamma', 'beta'):
    print('=' * 110)
    print(f'{k}: 0247 | 0248 at eval steps, per layer')
    print('   ' + f'{"step":>6} ' + ' '.join(f'{"L" + str(i) + " 0247":>11} {"0248":>7}' for i in range(L)))
    for s in SHOW:
        if s in ia and s in ib:
            print('   ' + f'{s:>6} ' + ' '.join(f'{a[k][i][ia[s]]:>11.4f} {b[k][i][ib[s]]:>7.4f}' for i in range(L)))
    fin = steps[-1]
    print('   final 0248-0247 ' + ' '.join(f'{b[k][i][ib[fin]] - a[k][i][ia[fin]]:>+9.4f}' for i in range(L)))
    for lab, s0, s1 in (('drift 12K->14K', 12000, 14000), ('drift 14K->16K', 14000, 16000)):
        if s0 in ib and s1 in ib:
            print(f'   {lab} 0247 ' + ' '.join(f'{a[k][i][ia[s1]] - a[k][i][ia[s0]]:>+8.4f}' for i in range(L)))
            print(f'   {lab} 0248 ' + ' '.join(f'{b[k][i][ib[s1]] - b[k][i][ib[s0]]:>+8.4f}' for i in range(L)))
    print('   trajectory corr  ' + ' '.join(
        f'{corr([a[k][i][ia[s]] for s in steps], [b[k][i][ib[s]] for s in steps]):>+9.3f}' for i in range(L)))
    if k == 'gamma':
        def first(run, i, idx):
            return next((s for s in run['step'] if run[k][i][idx[s]] > 1.1), None)
        print('   first step gamma > 1.1: 0247 ' + ' '.join(f'{str(first(a, i, ia)):>7}' for i in range(L))
              + ' | 0248 ' + ' '.join(f'{str(first(b, i, ib)):>7}' for i in range(L)))
print('=' * 110)
print(f'0248 g (must be exactly 0 at every eval): max |g| = {max(abs(v) for i in range(L) for v in b["g"][i]):.3e}')

if '--png' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = sys.argv[sys.argv.index('--png') + 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, k, init in zip(axes, ('gamma', 'beta'), (1.0, 2.0)):
        for i in range(L):
            ax.plot(a['step'], a[k][i], color=colors[i], lw=1.6, label=f'L{i} 0247 (g learnable)')
            ax.plot(b['step'], b[k][i], color=colors[i], lw=1.6, ls='--', label=f'L{i} 0248 (g frozen)')
        ax.axhline(init, color='0.5', ls=':', lw=1)
        ax.set(xlabel='step', title=f'learned {k}: solid 0247, dashed 0248')
        ax.grid(True, alpha=.3)
    axes[0].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print('wrote', out)
