"""Plot a distill_ffn.py sweep: per-layer FVU (LUT student vs linear baseline) + curves.

Works on a finished run (reads results.json) or an in-progress one (falls back to the latest
eval step in curves.csv and says so in the title), so it can be checked before a sweep ends.

Colour: Okabe-Ito, a published colour-vision-deficiency-safe categorical palette, assigned in
FIXED order by layer index / series identity, never by rank. Each panel has exactly one y-axis.

    python plot_distill.py runs/sweep_0238arch_16k
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9',
             '#F0E442', '#000000']
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'

run = sys.argv[1]
run = run if os.path.isabs(run) else os.path.join(os.path.dirname(os.path.abspath(__file__)), run)
man = json.load(open(os.path.join(run, 'manifest.json')))
layers = man['layers']
rows = list(csv.DictReader(open(os.path.join(run, 'curves.csv'))))
res_p = os.path.join(run, 'results.json')

if os.path.exists(res_p):
    res = json.load(open(res_p))
    fin = {int(k): v for k, v in res['final'].items()}
    student_fvu = [fin[li]['fvu'] for li in layers]
    at_step, status = res['steps'], 'final'
else:
    at_step = max(int(r['step']) for r in rows)
    by = {int(r['layer']): float(r['val_fvu']) for r in rows if int(r['step']) == at_step}
    student_fvu = [by[li] for li in layers]
    status = f'IN PROGRESS, step {at_step:,}/{man["steps"]:,}'
lin_fvu = [man['linear_baseline'][str(li)]['fvu'] for li in layers]

plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))

# ---- panel 1: the complexity curve across depth ----------------------------------------
series = [('LUT student (exp_n_0238 arch)', student_fvu, OKABE_ITO[0]),
          ('linear least-squares baseline', lin_fvu, OKABE_ITO[1])]
for name, ys, col in series:
    a1.plot(layers, ys, '-o', color=col, lw=2, ms=8, label=name,
            markeredgecolor='white', markeredgewidth=2)
    # direct label at the right end (2 series, so every series is labelled)
    a1.annotate(name.split(' (')[0], (layers[-1], ys[-1]), xytext=(8, 0),
                textcoords='offset points', va='center', color=INK2, fontsize=9)
for li, y, yl in zip(layers, student_fvu, lin_fvu):
    # Place the value label on the side the OTHER series doesn't pass through. Where the
    # student sits above the linear baseline (e.g. L0), the linear line rises through the
    # up-right slot, so go up-left there; everywhere else up-right is clear.
    left = y > yl
    a1.annotate(f'{y:.3f}', (li, y), xytext=(-9 if left else 9, 7),
                textcoords='offset points', ha='right' if left else 'left',
                color=INK, fontsize=8)
a1.set_xticks(layers)
a1.set_xticklabels([f'L{li}' for li in layers])
a1.set_xlabel('layer')
a1.set_ylabel('FVU  (fraction of teacher FFN variance unexplained)')
a1.set_ylim(bottom=0)
a1.set_xlim(layers[0] - 0.3, layers[-1] + 1.4)
a1.set_title(f'Per-layer FFN distillation error ({status})', loc='left', color=INK)
a1.legend(frameon=False, loc='lower right', fontsize=9)   # upper-left collided with the L2 peak
a1.grid(axis='y', color=GRID, lw=0.8)
a1.set_axisbelow(True)
for s in ('top', 'right'):
    a1.spines[s].set_visible(False)

# ---- panel 2: held-out FVU during training, one line per layer ---------------------------
for i, li in enumerate(layers):
    pts = sorted((int(r['step']), float(r['val_fvu'])) for r in rows
                 if int(r['layer']) == li and int(r['step']) > 0)
    if pts:
        a2.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=OKABE_ITO[i], lw=2,
                label=f'L{li}')
a2.set_yscale('log')
# plain decimals, not 6x10^-1 -- read directly against the FVU numbers in panel 1
from matplotlib.ticker import FuncFormatter, LogLocator   # noqa: E402
a2.yaxis.set_major_locator(LogLocator(base=10, subs=(0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)))
a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
a2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
a2.set_xlabel('distillation step')
a2.set_ylabel('held-out FVU  (log scale)')
a2.set_title('Student training curves', loc='left', color=INK)
a2.legend(frameon=False, ncol=3, fontsize=9)
a2.grid(color=GRID, lw=0.8, which='both', alpha=0.7)
a2.set_axisbelow(True)
for s in ('top', 'right'):
    a2.spines[s].set_visible(False)

fig.suptitle(f'Teacher {man["teacher_cfg_name"]} · student {man["student_params_per_layer"][str(layers[0])]:,} '
             f'params/layer (teacher FFN {man["teacher_ffn_params_per_layer"]:,}) · '
             f'{man["tokens_per_step"]:,} tokens/step · eval {man["eval_tokens"]:,} held-out tokens',
             color=INK2, fontsize=9, y=0.995)
fig.tight_layout()
out = os.path.join(run, 'distill_per_layer.png')
fig.savefig(out, dpi=140, facecolor='white')
print(out)
