"""Capacity ladder (plus other single configs) for distill_ffn.py runs: per-layer FVU vs size.

Two views, and the answers, printed and saved as JSON next to the figure:
  (a) FVU by layer, one line per run, linear least-squares baseline dashed. Is the per-layer
      complexity ORDERING the same at every capacity?
  (b) FVU vs student params (log-log), one line per layer through the LADDER runs. Any other
      config is a hollow marker in its layer's colour, and is compared numerically with the
      ladder trend interpolated (log-log, piecewise) to its own param count.
A run is a LADDER run when its overrides touch nothing but `lut_n_heads`.

Uses each run's results.json when present, else its latest eval step (marked in the legend).

    python plot_ladder.py runs/ladder4k_H4 runs/ladder4k_H8 runs/ladder4k_H16 \
        [runs/<other config>] [--out runs/<name>.png]
"""
import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator          # noqa: E402

OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9',
             '#F0E442', '#000000']
# Ladder sizes are ORDERED, so panel (a) gives them one sequential hue, light -> dark.
SIZE_RAMP = ['#9ecae1', '#4292c6', '#08519c', '#08306b']      # evenly stepped lightness
EXTRA_COLORS = ['#D55E00', '#AA3377']                           # other configs, panel (a)
EXTRA_MARKERS = ['D', 's']
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'
HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = [(2, 3), (4,), (1,), (5,), (0,)]                        # hardest first


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
    C = round(math.sqrt(man['teacher_ffn_params_per_layer'] / 8))        # 2*C*4C
    out = cfg['lut_inner_out_dim']
    return {'name': os.path.basename(run), 'layers': layers, 'fvu': fvu, 'step': step,
            'steps': man['steps'], 'done': done, 'cfg': cfg,
            'ladder': isinstance(ov, dict) and set(ov) <= {'lut_n_heads'},
            'params': man['student_params_per_layer'][str(layers[0])],
            'lin': {li: man['linear_baseline'][str(li)]['fvu'] for li in layers},
            # no decompress -> the output is a sum of C-dim table rows: full rank
            'rank': C if out == -1 else min(C, cfg['lut_n_heads'] * out)}


def describe(r):
    c = r['cfg']
    out = 'full' if c['lut_inner_out_dim'] == -1 else c['lut_inner_out_dim']
    return (f'H={c["lut_n_heads"]} tph={c["lut_tables_per_head"]} nap={c["lut_n_anchor_pairs"]} '
            f'd_in={c["lut_inner_in_dim"]} d_out={out}')


ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--out', default='runs/ladder4k_heads.png')
a = ap.parse_args()
dst = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)

allruns = [load(r) for r in a.runs]
ladder = sorted((r for r in allruns if r['ladder']), key=lambda r: r['params'])
extras = [r for r in allruns if not r['ladder']]
layers = allruns[0]['layers']
lin = allruns[0]['lin']
p0 = ladder[0]['params'] if ladder else allruns[0]['params']


def label(r):
    tag = (f'H={r["cfg"]["lut_n_heads"]} · {r["params"] / 1e6:.1f}M ({r["params"] / p0:.0f}×)'
           if r['ladder'] else f'{describe(r)} · {r["params"] / 1e6:.2f}M')
    return tag if r['done'] else f'{tag} — step {r["step"]:,}/{r["steps"]:,}'


def tiers_hold(fvu):
    return all(min(fvu[l] for l in hi) > max(fvu[l] for l in lo)
               for hi, lo in zip(TIERS, TIERS[1:]))


def ladder_trend(li, params):
    """Ladder FVU for layer li at `params`, piecewise-linear in log-log (end segments extend)."""
    xs = [math.log(r['params']) for r in ladder]
    ys = [math.log(r['fvu'][li]) for r in ladder]
    if len(xs) == 1:
        return math.exp(ys[0])
    lp = math.log(params)
    i = max(0, min(len(xs) - 2, sum(1 for x in xs if x <= lp) - 1))
    t = (lp - xs[i]) / (xs[i + 1] - xs[i])
    return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))


# ---- answers ---------------------------------------------------------------------------------
fmt = lambda order: ' > '.join(f'L{li}' for li in order)   # noqa: E731
report = {'runs': [], 'tiers': [list(t) for t in TIERS]}
check_tiers = sorted(layers) == list(range(6))
for r in ladder + extras:
    rk = sorted(layers, key=lambda li: -r['fvu'][li])
    row = {'name': r['name'], 'config': describe(r), 'ladder': r['ladder'], 'params': r['params'],
           'output_rank': r['rank'], 'done': r['done'], 'step': r['step'], 'fvu': r['fvu'],
           'ranking': rk, 'beats_linear': {li: r['fvu'][li] < lin[li] for li in layers}}
    if check_tiers:
        row['tiers_hold'] = tiers_hold(r['fvu'])
    print(f'{label(r)}   [output rank {r["rank"]}]')
    print('   FVU  ' + '  '.join(f'L{li}={r["fvu"][li]:.4f}' for li in layers))
    print(f'   rank {fmt(rk)}' + (f'   tiers {{L2,L3}}>L4>L1>L5>L0 '
                                  f'{"HOLD" if row["tiers_hold"] else "BROKEN"}' if check_tiers else ''))
    print(f'   L0 vs linear: {r["fvu"][0]:.4f} vs {lin[0]:.4f}')
    if not r['ladder'] and len(ladder) > 1:
        trend = {li: ladder_trend(li, r['params']) for li in layers}
        near = min(ladder, key=lambda q: abs(math.log(q['params'] / r['params'])))
        row['ladder_trend_at_params'] = trend
        row['ratio_to_trend'] = {li: r['fvu'][li] / trend[li] for li in layers}
        row['nearest_ladder_point'] = near['name']
        row['ratio_to_nearest'] = {li: r['fvu'][li] / near['fvu'][li] for li in layers}
        print(f'   ladder trend at {r["params"] / 1e6:.2f}M: ' +
              '  '.join(f'L{li}={trend[li]:.4f}' for li in layers))
        print('   actual / trend: ' + '  '.join(f'L{li}={row["ratio_to_trend"][li]:.3f}'
                                              for li in layers))
        print(f'   actual / {near["name"]} ({near["params"] / 1e6:.2f}M): ' +
              '  '.join(f'L{li}={row["ratio_to_nearest"][li]:.3f}' for li in layers))
    report['runs'].append(row)

# ---- figure ----------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID})
ratio_panel = bool(extras) and len(ladder) > 1
if ratio_panel:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 5.8),
                                     gridspec_kw={'width_ratios': [1.15, 1.15, 0.8]})
else:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.8))
    a3 = None

ramp = SIZE_RAMP[-len(ladder):] if len(ladder) <= len(SIZE_RAMP) else SIZE_RAMP[-1:] * len(ladder)
for r, col in zip(ladder, ramp):
    a1.plot(layers, [r['fvu'][li] for li in layers], marker='o', color=col, lw=2, ms=7,
            markeredgecolor='white', markeredgewidth=1.5, label=label(r),
            ls='-' if r['done'] else (0, (4, 2)))
for k, r in enumerate(extras):
    a1.plot(layers, [r['fvu'][li] for li in layers], marker=EXTRA_MARKERS[k % 2],
            color=EXTRA_COLORS[k % 2], lw=2, ms=7, markeredgecolor='white', markeredgewidth=1.5,
            label=label(r), ls=(0, (6, 2, 1, 2)))
a1.plot(layers, [lin[li] for li in layers], marker='o', color=INK2, lw=1.8, ms=6, ls='--',
        markeredgecolor='white', markeredgewidth=1.5, label='linear least-squares baseline')
a1.set_xticks(layers)
a1.set_xticklabels([f'L{li}' for li in layers])
a1.set_xlabel('layer')
a1.set_ylabel('held-out FVU')
a1.set_ylim(0, max(max(lin.values()), max(max(r['fvu'].values()) for r in allruns)) * 1.35)
a1.set_title('FVU by layer, one line per student', loc='left', color=INK)
a1.legend(frameon=False, fontsize=8.5, loc='upper right')   # headroom above the L2 peak
a1.grid(axis='y', color=GRID, lw=0.8)
a1.set_axisbelow(True)

for i, li in enumerate(layers):
    if ladder:
        a2.plot([r['params'] for r in ladder], [r['fvu'][li] for r in ladder], marker='o',
                color=OKABE_ITO[i], lw=2, ms=7, markeredgecolor='white', markeredgewidth=1.5,
                label=f'L{li}')
    for k, r in enumerate(extras):
        a2.plot([r['params']], [r['fvu'][li]], marker=EXTRA_MARKERS[k % 2], ls='none', ms=10,
                markerfacecolor='white', markeredgecolor=OKABE_ITO[i], markeredgewidth=2.2,
                zorder=5)
a2.set_xscale('log')
a2.set_yscale('log')
a2.set_xticks([r['params'] for r in ladder] or [r['params'] for r in allruns])
a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e6:.1f}M'))
a2.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
a2.yaxis.set_major_locator(LogLocator(base=10, subs=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)))
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
a2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
# Direct end labels on the ladder lines, nudged apart where lines finish close together.
if ladder:
    lo, hi = (math.log10(v) for v in a2.get_ylim())
    ends = sorted(((math.log10(ladder[-1]['fvu'][li]) - lo) / (hi - lo), li) for li in layers)
    placed = []
    for frac, _ in ends:
        placed.append(max(frac, placed[-1] + 0.045) if placed else frac)
    for (_, li), frac in zip(ends, placed):
        a2.annotate(f'L{li}', (ladder[-1]['params'], frac), xycoords=('data', 'axes fraction'),
                    xytext=(7, 0), textcoords='offset points', va='center', color=INK2, fontsize=9)
handles, labels = a2.get_legend_handles_labels()
for k, r in enumerate(extras):
    handles.append(Line2D([], [], marker=EXTRA_MARKERS[k % 2], ls='none', ms=9,
                          markerfacecolor='white', markeredgecolor=INK2, markeredgewidth=2))
    labels.append(describe(r))
a2.legend(handles, labels, frameon=False, ncol=4, fontsize=8.5, loc='lower left')
a2.set_xlabel('student params per layer (log scale)')
a2.set_ylabel('held-out FVU (log scale)')
a2.set_title('Scaling curve per layer' + (' (lines: H ladder; hollow: other configs)'
                                          if extras else ''), loc='left', color=INK)
a2.grid(color=GRID, lw=0.8, which='major')
a2.set_axisbelow(True)

# (c) The question panel (b) cannot answer by eye when an extra config lands on top of a ladder
# point: per layer, FVU divided by the ladder trend at the SAME param count. 1.0 = on trend.
if a3 is not None:
    rows = {row['name']: row for row in report['runs']}
    a3.axhline(1.0, color=INK2, lw=1.2, ls='--')
    # centred over the middle layers: the right end collided with the L5 marker
    a3.annotate('on the ladder trend', ((layers[0] + layers[-1]) / 2, 1.0), xytext=(0, 5),
                textcoords='offset points', ha='center', color=INK2, fontsize=8.5)
    for k, r in enumerate(extras):
        ratio = rows[r['name']]['ratio_to_trend']
        a3.plot(layers, [ratio[li] for li in layers], marker=EXTRA_MARKERS[k % 2],
                color=EXTRA_COLORS[k % 2], lw=1.5, ms=8, markeredgecolor='white',
                markeredgewidth=1.5, label=describe(r))
        for li in layers:
            a3.annotate(f'{ratio[li]:.2f}', (li, ratio[li]), xytext=(0, 9 if ratio[li] >= 1 else -14),
                        textcoords='offset points', ha='center', color=INK, fontsize=8)
    a3.set_yscale('log')
    vals = [rows[r['name']]['ratio_to_trend'][li] for r in extras for li in layers]
    span = max(max(vals), 1 / min(vals)) * 1.15
    a3.set_ylim(1 / span, span)
    a3.yaxis.set_major_locator(LogLocator(base=10, subs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25,
                                                         1.5, 2.0, 3.0)))
    a3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
    a3.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    a3.set_xticks(layers)
    a3.set_xticklabels([f'L{li}' for li in layers])
    a3.set_xlabel('layer')
    a3.set_ylabel('FVU ÷ ladder trend at equal params (log scale)')
    a3.set_title('Above or below the ladder trend?', loc='left', color=INK)
    a3.text(0.02, 0.02, 'below 1 = better than the H ladder at the same size',
            transform=a3.transAxes, color=INK2, fontsize=8.5)
    a3.grid(axis='y', color=GRID, lw=0.8)
    a3.set_axisbelow(True)
for ax in (a1, a2) + ((a3,) if a3 is not None else ()):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

steps = sorted({r['steps'] for r in allruns})
fig.suptitle(f'Per-layer FFN distillation · {"/".join(f"{s:,}" for s in steps)} native steps · '
             'ladder scales lut_n_heads at exp_n_0238 tph/nap/inner', color=INK2, fontsize=10,
             y=0.995)
fig.tight_layout()
fig.savefig(dst, dpi=140, facecolor='white')
json.dump(report, open(os.path.splitext(dst)[0] + '.json', 'w'), indent=2)
print(f'\nwrote {dst}')
