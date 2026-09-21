"""Figures for experiments 2 and 3, built from the committed run artifacts only.

Every series is loaded from runs_debug/altmin_ridge_d0.3.json and runs_debug/ls_direction.json; nothing
is retyped from a summary. Points are drawn as markers because the data exists only at the sampled
probe steps (outer 0,4,...,40 for experiment 2; training steps 1/100/250/500 for experiment 3), so a
dense line would imply resolution the runs do not have.

Palette: Okabe-Ito, a published colourblind-safe set. The dataviz skill's own validator
(scripts/validate_palette.js) is not present on this machine any more, so it could not be run; a
pre-validated published palette is the substitute, and hues are assigned in fixed order per entity
rather than cycled.

Usage: python3 make_plots.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs_debug', 'plots')
AM = os.path.join(HERE, 'runs_debug', 'altmin_ridge_d0.3.json')
LS = os.path.join(HERE, 'runs_debug', 'ls_direction.json')

# Okabe-Ito, assigned by entity and never cycled
C = {'probe_test': '#0072B2', 'probe_train': '#D55E00', 'model': '#009E73',
     'readout': '#0072B2', 'interior': '#D55E00', 'viol': '#CC79A7',
     'opt': '#0072B2', 'composed': '#D55E00', 'ref': '#555555'}
INK = '#222222'
plt.rcParams.update({'axes.edgecolor': '#BBBBBB', 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.titlesize': 11,
                     'font.size': 9, 'axes.grid': True, 'grid.color': '#E4E4E4',
                     'grid.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def finite(rows, key):
    """Sampled points only, dropping any the run did not produce (arm A NaNs out partway)."""
    return [(r['step'], r[key]) for r in rows
            if key in r and isinstance(r[key], (int, float)) and not math.isnan(r[key])]


def tidy(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def fig1_and_2(am):
    """Headline: the interior never gets above its value at initialisation. LUT beside the MLP control.

    Two rows on purpose. The top row is the honest full-range view; the bottom row is the same data
    zoomed, because the control's improvement is 2.7 accuracy points and is invisible on a 0-1 axis.
    The zoom is labelled as a zoom rather than passed off as the default view."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    for j, (key, label) in enumerate([('lut-pcalmB', 'LUT stack, PC-ALM-B'),
                                      ('mlp-pc', 'plain MLP, PC (control)')]):
        rows = am['arms'][key]
        init = dict(finite(rows, 'probe_test_acc'))[0]
        a = axes[0][j]
        a.axhline(init, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
        a.annotate(f'interior at initialisation: {init:.4f}', xy=(40, init), xytext=(-4, 5),
                   textcoords='offset points', ha='right', va='bottom', fontsize=8.5, color=C['ref'])
        for k, lbl, col in (('probe_train_acc', 'interior, optimal readout (train)', C['probe_train']),
                            ('probe_test_acc', 'interior, optimal readout (test)', C['probe_test']),
                            ('test_acc', 'model as trained (test)', C['model'])):
            s = finite(rows, k)
            a.plot([p for p, _ in s], [v for _, v in s], 'o-', ms=5, lw=1.8, color=col, label=lbl,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        tidy(a, '', 'accuracy' if j == 0 else '', label)
        a.set_ylim(0.0, 1.0)
        a.legend(loc='lower right', frameon=False, fontsize=8.5)

        z = axes[1][j]
        z.axhline(init, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
        for k, col in (('probe_train_acc', C['probe_train']), ('probe_test_acc', C['probe_test']),
                       ('test_acc', C['model'])):
            sp = [q for q in finite(rows, k) if q[0] > 0]      # drop the untrained point so the zoom fits
            z.plot([q for q, _ in sp], [v for _, v in sp], 'o-', ms=5, lw=1.8, color=col,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        tidy(z, 'outer step (exact least-squares weight solve)',
             'accuracy (zoom)' if j == 0 else '', 'zoom, from outer step 4')
        z.set_ylim(0.70, 0.95)
    axes[1][0].annotate('train probe climbs 0.911 -> 0.924 while the test probe\n'
                        'sits on its value at initialisation: memorisation',
                        xy=(20, 0.924), xytext=(8, 0.845), fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle='->', color='#888888', lw=1))
    axes[1][1].annotate('control interior really does improve\n0.730 -> 0.740 on test',
                        xy=(40, 0.7401), xytext=(10, 0.80), fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle='->', color='#888888', lw=1))
    fig.tight_layout()
    p = os.path.join(OUT, 'exp2_interior_vs_control.png')
    fig.savefig(p, dpi=170)
    return p


def fig3(am):
    """Readout loss collapses; interior loss barely moves."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for j, (key, label) in enumerate([('lut-pcalmB', 'LUT stack, PC-ALM-B'),
                                      ('mlp-pc', 'plain MLP, PC (control)')]):
        rows = am['arms'][key]
        a = ax[j]
        a.axhline(0.5, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
        a.annotate('0.50 = emitting nothing', xy=(0, 0.5), xytext=(6, 4), textcoords='offset points',
                   ha='left', va='bottom', fontsize=8.5, color=C['ref'])
        for k, lbl, col in (('readout_data_loss', 'readout data loss (the model)', C['readout']),
                            ('interior_data_loss', 'interior data loss (readout re-solved)', C['interior'])):
            s = finite(rows, k)
            a.plot([p for p, _ in s], [v for _, v in s], 'o-', ms=5, lw=1.8, color=col, label=lbl,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        tidy(a, 'outer step', '1/2||yhat - y||^2 per sample' if j == 0 else '', label)
        a.set_ylim(0, 0.56)
        a.legend(loc='upper right', frameon=False, fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(OUT, 'exp2_losses.png')
    fig.savefig(p, dpi=170)
    return p


def fig4(am):
    """Interior constraint violation, spike included and unclipped."""
    rows = am['arms']['lut-pcalmB']
    s = finite(rows, 'interior_viol')
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(1.0, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate('1.0 = as violated as doing nothing', xy=(40, 1.0), xytext=(-4, 5),
                textcoords='offset points', ha='right', va='bottom', fontsize=8.5, color=C['ref'])
    ax.plot([p for p, _ in s], [v for _, v in s], 'o-', ms=5, lw=1.8, color=C['viol'],
            markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    pk = max(s, key=lambda t: t[1])
    ax.annotate(f'spike to {pk[1]:.2f}', xy=pk, xytext=(-10, -18), textcoords='offset points',
                fontsize=8.5, color=INK, ha='right',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1))
    # linear rather than log: the whole range is 0.23 to 1.81, well inside one decade, and linear keeps
    # the plateau and the spike equally legible without pretending at orders of magnitude.
    tidy(ax, 'outer step', 'interior constraint violation  r_cur / r_zero',
         'LUT PC-ALM-B: the relaxation targets ARE being met')
    ax.set_ylim(0, 2.0)
    fig.tight_layout()
    p = os.path.join(OUT, 'exp2_violation.png')
    fig.savefig(p, dpi=170)
    return p


def fig5(ls):
    """Individually reachable vs jointly compatible."""
    labels, opt, comp = [], [], []
    for arm in ('pcA', 'pcalmB'):
        cp = [c for c in ls['arms'][arm] if c['tag'] == '500'][0]
        for r in cp['layers']:
            labels.append(f'{"A" if arm == "pcA" else "B"}\n{r["layer"]}')
            opt.append(r['r_opt_frozen'] / r['r_cur'])
            comp.append(r['r_composed'] / r['r_cur'])
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    w = 0.38
    ax.axhline(1.0, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate('1.0 = the residual PC currently leaves', xy=(len(labels) - 0.5, 1.0), xytext=(-4, 5),
                textcoords='offset points', ha='right', va='bottom', fontsize=8.5, color=C['ref'])
    ax.bar([x - w / 2 - 0.01 for x in xs], opt, w, color=C['opt'], label='fitted layer by layer',
           edgecolor='white', linewidth=1.5, zorder=3)
    ax.bar([x + w / 2 + 0.01 for x in xs], comp, w, color=C['composed'],
           label='all layers installed, forward pass recomputed', edgecolor='white', linewidth=1.5,
           zorder=3)
    for x, v in zip(xs, opt):
        ax.annotate(f'{v:.2f}', xy=(x - w / 2, v), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=C['opt'])
    for x, v in zip(xs, comp):
        ax.annotate(f'{v:.2f}', xy=(x + w / 2, v), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=C['composed'])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    tidy(ax, 'arm and layer (at training step 500)', 'residual after the fit, / current residual',
         'Reachable layer by layer; only partly compatible together')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', frameon=False, fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(OUT, 'exp3_reachability.png')
    fig.savefig(p, dpi=170)
    return p


def fig6(ls):
    """Near-orthogonality, on an axis that shows what 1.0 would look like."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    styles = {'L0': 'o-', 'L1': 's-', 'readout': '^-'}
    cols = {'pcA': '#0072B2', 'pcalmB': '#D55E00'}
    for arm in ('pcA', 'pcalmB'):
        for layer in ('L0', 'L1', 'readout'):
            pts = []
            for cp in ls['arms'][arm]:
                for r in cp['layers']:
                    if r['layer'] == layer:
                        pts.append((int(cp['tag']), r['cos_pc_update_vs_ls']))
            pts.sort()
            ax.plot([p for p, _ in pts], [v for _, v in pts], styles[layer], ms=5, lw=1.6,
                    color=cols[arm], alpha=0.9, markeredgecolor='white', markeredgewidth=0.8,
                    label=f'{"PC-A" if arm == "pcA" else "PC-ALM-B"}, {layer}', zorder=3)
    ax.axhline(0.0, color=C['ref'], lw=1.2, zorder=1)
    ax.axhline(1.0, color=C['ref'], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate('1.0 = stepping straight at the solvable answer', xy=(500, 1.0), xytext=(-4, -13),
                textcoords='offset points', ha='right', fontsize=8.5, color=C['ref'])
    ax.annotate('0 = orthogonal', xy=(500, 0.0), xytext=(-4, 5), textcoords='offset points',
                ha='right', fontsize=8.5, color=C['ref'])
    tidy(ax, 'training step', 'cosine(PC weight update, least-squares direction)',
         'PC steps almost at right angles to the answer to its own subproblem')
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc='center left', frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, 'exp3_cosine.png')
    fig.savefig(p, dpi=170)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    am = json.load(open(AM))
    ls = json.load(open(LS))
    paths = [fig1_and_2(am), fig3(am), fig4(am), fig5(ls), fig6(ls)]
    # what could NOT be recovered from disk, stated rather than reconstructed
    a_rows = am['arms']['lut-pcA']
    dead = [r['step'] for r in a_rows
            if isinstance(r['readout_data_loss'], float) and math.isnan(r['readout_data_loss'])]
    print(f'LUT arm A: its losses are NaN from outer step {min(dead)} onward, after which its accuracies '
          f'are the finite-but-meaningless 0.10 that argmax of a NaN output produces. It is left off the '
          f'experiment-2 panels rather than drawn as a curve that looks like a result.')
    for p in paths:
        print('wrote', os.path.relpath(p, HERE))


if __name__ == '__main__':
    main()
