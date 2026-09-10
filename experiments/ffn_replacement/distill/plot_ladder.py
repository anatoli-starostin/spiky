"""Capacity ladder: per-layer FVU across student sizes, for distill_ffn.py runs.

Two views, and the answers to the ladder's questions, printed and saved to ladder.json:
  (a) FVU by layer, one line per student size, linear least-squares baseline dashed. Is the
      per-layer complexity ORDERING the same at every capacity?
  (b) FVU vs student params (log x), one line per layer: the scaling curve of each layer.
Plus the output-rank floor (variance outside the top H*inner_out principal directions) where
known, because at small H it bounds what any table can reach.

Uses each run's results.json when present, else its latest eval step (marked in the legend).

    python plot_ladder.py runs/ladder4k_H4 runs/ladder4k_H8 runs/ladder4k_H16 runs/ladder4k_H32
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator          # noqa: E402

OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9',
             '#F0E442', '#000000']
# Sizes are ORDERED, so panel (a) uses one sequential hue light -> dark (Blues), not categories.
SIZE_RAMP = ['#9ecae1', '#4292c6', '#08519c', '#08306b']      # evenly stepped lightness
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'
HERE = os.path.dirname(os.path.abspath(__file__))


def load(run):
    run = run if os.path.isabs(run) else os.path.join(HERE, run)
    man = json.load(open(os.path.join(run, 'manifest.json')))
    layers = man['layers']
    res_p = os.path.join(run, 'results.json')
    if os.path.exists(res_p):
        res = json.load(open(res_p))
        fvu = {int(k): v['fvu'] for k, v in res['final'].items()}
        step, done = res['steps'], True
    else:
        rows = list(csv.DictReader(open(os.path.join(run, 'curves.csv'))))
        step = max(int(r['step']) for r in rows)
        fvu = {int(r['layer']): float(r['val_fvu']) for r in rows if int(r['step']) == step}
        done = False
    ov = man['student_overrides']
    base = json.load(open(os.path.join(os.path.dirname(HERE), man['student_config'])))
    cfg = {**base, **(ov if isinstance(ov, dict) else {})}
    return {'name': os.path.basename(run), 'layers': layers, 'fvu': fvu, 'step': step,
            'steps': man['steps'], 'done': done, 'cfg': cfg,
            'params': man['student_params_per_layer'][str(layers[0])],
            'lin': {li: man['linear_baseline'][str(li)]['fvu'] for li in layers},
            'rank': cfg['lut_n_heads'] * cfg['lut_inner_out_dim']}


runs = sorted((load(r) for r in sys.argv[1:]), key=lambda r: r['params'])
layers = runs[0]['layers']
lin = runs[0]['lin']
p0 = runs[0]['params']


def label(r):
    c = r['cfg']
    tag = f'H={c["lut_n_heads"]} · {r["params"] / 1e6:.1f}M ({r["params"] / p0:.0f}×)'
    return tag if r['done'] else f'{tag} — step {r["step"]:,}/{r["steps"]:,}'


# ---- answers ---------------------------------------------------------------------------------
fmt = lambda order: ' > '.join(f'L{li}' for li in order)   # noqa: E731
out = {'runs': [], 'reference_ranking': None}
ref = sorted(layers, key=lambda li: -runs[0]['fvu'][li])
out['reference_ranking'] = ref
print(f'reference (smallest) ranking, hardest first: {fmt(ref)}\n')
for r in runs:
    rk = sorted(layers, key=lambda li: -r['fvu'][li])
    beats = {li: r['fvu'][li] < lin[li] for li in layers}
    out['runs'].append({'name': r['name'], 'params': r['params'], 'done': r['done'],
                        'step': r['step'], 'fvu': r['fvu'], 'ranking': rk,
                        'same_ranking_as_reference': rk == ref, 'beats_linear': beats})
    print(f'{label(r)}')
    print('   FVU  ' + '  '.join(f'L{li}={r["fvu"][li]:.4f}' for li in layers))
    print(f'   rank {fmt(rk)}   {"SAME" if rk == ref else "DIFFERENT"} as reference')
    print(f'   L0 vs linear: {r["fvu"][0]:.4f} vs {lin[0]:.4f} -> '
          f'{"BEATS linear" if beats[0] else "does not beat linear"}')
if len(runs) > 1:
    print('\nFVU ratio smallest -> largest, per layer (lower = more gain from capacity)')
    print('   ' + '  '.join(f'L{li}={runs[-1]["fvu"][li] / runs[0]["fvu"][li]:.3f}'
                             for li in layers))
    out['fvu_ratio_largest_over_smallest'] = {li: runs[-1]['fvu'][li] / runs[0]['fvu'][li]
                                              for li in layers}

# ---- figure ----------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))

ramp = SIZE_RAMP[-len(runs):] if len(runs) <= len(SIZE_RAMP) else SIZE_RAMP[-1:] * len(runs)
for r, col in zip(runs, ramp):
    a1.plot(layers, [r['fvu'][li] for li in layers], marker='o', color=col, lw=2, ms=7,
            markeredgecolor='white', markeredgewidth=1.5, label=label(r),
            ls='-' if r['done'] else (0, (4, 2)))
a1.plot(layers, [lin[li] for li in layers], marker='o', color=INK2, lw=1.8, ms=6, ls='--',
        markeredgecolor='white', markeredgewidth=1.5, label='linear least-squares baseline')
a1.set_xticks(layers)
a1.set_xticklabels([f'L{li}' for li in layers])
a1.set_xlabel('layer')
a1.set_ylabel('held-out FVU')
a1.set_ylim(0, max(max(lin.values()), max(max(r['fvu'].values()) for r in runs)) * 1.3)
a1.set_title('FVU by layer, one line per student size', loc='left', color=INK)
a1.legend(frameon=False, fontsize=8.5, loc='upper right')   # headroom above the L2 peak
a1.grid(axis='y', color=GRID, lw=0.8)
a1.set_axisbelow(True)

for i, li in enumerate(layers):
    xs = [r['params'] for r in runs]
    ys = [r['fvu'][li] for r in runs]
    a2.plot(xs, ys, marker='o', color=OKABE_ITO[i], lw=2, ms=7, markeredgecolor='white',
            markeredgewidth=1.5, label=f'L{li}')
a2.set_xscale('log')
a2.set_yscale('log')
a2.set_xticks([r['params'] for r in runs])
a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e6:.1f}M'))
a2.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
a2.yaxis.set_major_locator(LogLocator(base=10, subs=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)))
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
a2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
# Direct end labels, nudged apart where lines finish close together (L2 and L3 end ~3% apart,
# which would otherwise print one label on top of the other).
lo, hi = (math.log10(v) for v in a2.get_ylim())
ends = sorted(((math.log10(runs[-1]['fvu'][li]) - lo) / (hi - lo), li) for li in layers)
placed = []
for frac, _ in ends:
    placed.append(max(frac, placed[-1] + 0.045) if placed else frac)
for (_, li), frac in zip(ends, placed):
    a2.annotate(f'L{li}', (runs[-1]['params'], frac), xycoords=('data', 'axes fraction'),
                xytext=(7, 0), textcoords='offset points', va='center', color=INK2, fontsize=9)
a2.set_xlabel('student params per layer (log scale)')
a2.set_ylabel('held-out FVU (log scale)')
a2.set_title('Scaling curve per layer', loc='left', color=INK)
a2.legend(frameon=False, ncol=3, fontsize=9, loc='lower left')
a2.grid(color=GRID, lw=0.8, which='major')
a2.set_axisbelow(True)
for ax in (a1, a2):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

steps = sorted({r['steps'] for r in runs})
fig.suptitle(f'Per-layer FFN distillation capacity ladder · {"/".join(f"{s:,}" for s in steps)}'
             f' native steps · lut_n_heads scaled, tph/nap/inner fixed at exp_n_0238', color=INK2,
             fontsize=10, y=0.995)
fig.tight_layout()
dst = os.path.join(HERE, 'runs', 'ladder4k_heads.png') if len(sys.argv) > 1 else None
fig.savefig(dst, dpi=140, facecolor='white')
json.dump(out, open(os.path.join(HERE, 'runs', 'ladder4k_heads.json'), 'w'), indent=2)
print(f'\nwrote {dst}')
