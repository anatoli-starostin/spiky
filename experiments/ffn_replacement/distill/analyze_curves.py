"""Per-layer curves + convergence analysis for a distill_ffn.py sweep.

Makes one 3-panel figure and prints (and saves) the answers to two questions:
  * at what step does each layer's held-out error plateau, and
  * does the RANKING of layers by error early on match the ranking at the end?
which together decide whether future sweeps can be cut short.

Works on a finished or an in-progress run: "final" means the latest eval step present, and the
output says which.

SCHEDULE CAVEAT, printed with the results. The sweep's cosine LR schedule is tied to its full
`--steps`, so an early point on this curve was taken at HIGH learning rate. A sweep natively
configured to that shorter length would anneal and land lower at the same step. Truncating a
long curve therefore OVERSTATES how much a short sweep loses; ranking stability is the more
schedule-robust signal.

Colour: Okabe-Ito (published CVD-safe categorical palette), fixed order by layer index.

    python analyze_curves.py runs/sweep_0238arch_16k
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator          # noqa: E402

OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9',
             '#F0E442', '#000000']
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'
TOL = 0.02

run = sys.argv[1]
run = run if os.path.isabs(run) else os.path.join(os.path.dirname(os.path.abspath(__file__)), run)
man = json.load(open(os.path.join(run, 'manifest.json')))
layers, N = man['layers'], man['steps']
rows = list(csv.DictReader(open(os.path.join(run, 'curves.csv'))))
done = os.path.exists(os.path.join(run, 'results.json'))


def series(col):
    return {li: {int(r['step']): float(r[col]) for r in rows if int(r['layer']) == li}
            for li in layers}


fvu, rel = series('val_fvu'), series('val_rel_err')
evals = sorted(s for s in {int(r['step']) for r in rows} if s > 0)
last = evals[-1]
status = 'final' if done else f'IN PROGRESS — latest eval step {last:,} of {N:,}'


# ---- plateau: first eval step after which the layer STAYS within TOL of its latest value --
def plateau(s):
    fin = s[last]
    for st in evals:
        if all(s[t] <= fin * (1 + TOL) for t in evals if t >= st):
            return st
    return last


def ranking(s_by_layer, st):
    """Hardest first (highest error)."""
    return sorted(layers, key=lambda li: -s_by_layer[li][st])


def kendall(a, b):
    pos = {li: i for i, li in enumerate(b)}
    n, conc = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            conc += 1 if pos[a[i]] < pos[a[j]] else -1
    return conc / (n * (n - 1) / 2)


checks = [s for s in (2000, 4000) if s in evals and s < last] + [last]
report = {'run': os.path.basename(run), 'status': status, 'latest_step': last, 'total_steps': N,
          'tolerance': TOL, 'plateau_step': {}, 'rankings_fvu': {}, 'rankings_rel_err': {},
          'kendall_tau_vs_latest': {}, 'fvu_at': {}, 'fvu_ratio_to_latest': {}}
for li in layers:
    report['plateau_step'][li] = plateau(fvu[li])
for st in checks:
    report['rankings_fvu'][st] = ranking(fvu, st)
    report['rankings_rel_err'][st] = ranking(rel, st)
    report['kendall_tau_vs_latest'][st] = kendall(ranking(fvu, st), ranking(fvu, last))
    report['fvu_at'][st] = {li: fvu[li][st] for li in layers}
    report['fvu_ratio_to_latest'][st] = {li: fvu[li][st] / fvu[li][last] for li in layers}

# ---- text ------------------------------------------------------------------------------------
lin = {li: man['linear_baseline'][str(li)]['fvu'] for li in layers}
fmt = lambda order: ' > '.join(f'L{li}' for li in order)   # noqa: E731
print(f'{report["run"]}: {status}\n')
print(f'PLATEAU (first eval after which FVU stays within {TOL:.0%} of its step-{last:,} value)')
for li in layers:
    print(f'  L{li}: step {report["plateau_step"][li]:>6,}   '
          f'FVU@{last:,} = {fvu[li][last]:.4f}   (linear baseline {lin[li]:.4f})')
print('\nRANKING by held-out FVU, hardest first')
for st in checks:
    tag = 'latest' if st == last else f'{st:,}'
    print(f'  step {tag:>7}: {fmt(report["rankings_fvu"][st])}   '
          f'Kendall tau vs latest = {report["kendall_tau_vs_latest"][st]:+.3f}')
print('\nRANKING by relative error ||e||^2/||o_t||^2, hardest first')
for st in checks:
    tag = 'latest' if st == last else f'{st:,}'
    print(f'  step {tag:>7}: {fmt(report["rankings_rel_err"][st])}')
print('\nFVU at early steps as a multiple of the latest value (how far from converged)')
for st in checks[:-1]:
    print(f'  step {st:,}: ' + '  '.join(f'L{li}={report["fvu_ratio_to_latest"][st][li]:.2f}x'
                                      for li in layers))
print('\nCAVEAT: the LR schedule is cosine over all', f'{N:,}', 'steps, so early points were taken'
      ' at high LR. A sweep natively configured that short would anneal and land lower at the'
      ' same step; truncating this curve overstates what a short sweep loses.')
json.dump(report, open(os.path.join(run, 'analysis.json'), 'w'), indent=2)

# ---- figure ----------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 5.4))


def curve_panel(ax, s_by_layer, ylabel, title):
    for i, li in enumerate(layers):
        pts = sorted(s_by_layer[li].items())
        pts = [(x, y) for x, y in pts if x > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=OKABE_ITO[i], lw=2,
                label=f'L{li}')
    for st in (2000, 4000):
        if st < last:
            ax.axvline(st, color=INK2, lw=1, ls=':', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # the 2K/4K comparison steps are x tick labels, not in-plot text (which hit the legend)
    ax.set_xticks([t for t in (500, 1000, 2000, 4000, 8000, 16000) if t <= last])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}K' if v >= 1000
                                               else f'{v:g}'))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.set_xlabel('distillation step (log scale)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', color=INK)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc='upper right')
    ax.grid(color=GRID, lw=0.8, which='major')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


curve_panel(a1, fvu, 'held-out FVU (log scale)', 'Fraction of variance unexplained')
curve_panel(a2, rel, 'held-out relative error ||e||²/||o||² (log scale)', 'Relative error')

ys, yl = [fvu[li][last] for li in layers], [lin[li] for li in layers]
for name, vals, col in (('LUT student', ys, OKABE_ITO[0]), ('linear baseline', yl, INK2)):
    a3.plot(layers, vals, marker='o', color=col, lw=2, ms=8, label=name,
            markeredgecolor='white', markeredgewidth=2,
            ls='-' if name == 'LUT student' else '--')
    a3.annotate(name, (layers[-1], vals[-1]), xytext=(8, 0), textcoords='offset points',
                va='center', color=INK2, fontsize=9)
for li, y, ylin in zip(layers, ys, yl):
    left = y > ylin
    a3.annotate(f'{y:.3f}', (li, y), xytext=(-9 if left else 9, 7), textcoords='offset points',
                ha='right' if left else 'left', color=INK, fontsize=8)
a3.set_xticks(layers)
a3.set_xticklabels([f'L{li}' for li in layers])
a3.set_xlim(layers[0] - 0.3, layers[-1] + 1.3)
a3.set_ylim(bottom=0)
a3.set_xlabel('layer')
a3.set_ylabel('held-out FVU')
a3.set_title(f'Complexity by layer (step {last:,})', loc='left', color=INK)
a3.legend(frameon=False, loc='lower right', fontsize=9)
a3.grid(axis='y', color=GRID, lw=0.8)
a3.set_axisbelow(True)
for sp in ('top', 'right'):
    a3.spines[sp].set_visible(False)

fig.suptitle(f'Per-layer FFN distillation — {report["run"]} ({status}) · student '
             f'{man["student_params_per_layer"][str(layers[0])]:,} params/layer, '
             f'{man["eval_tokens"]:,} held-out tokens', color=INK2, fontsize=10, y=0.995)
fig.tight_layout()
out = os.path.join(run, 'curves_by_layer.png')
fig.savefig(out, dpi=140, facecolor='white')
print(f'\nwrote {out}')
