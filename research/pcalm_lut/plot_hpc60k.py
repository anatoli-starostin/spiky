"""Plots for the 60K hierarchical-predictive-coding run. Okabe-Ito palette, no dual axes."""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_ae')
RUN = 'cmhl-L4-hpc-diag-s60000'
OUT = os.path.join(R, 'plots_hpc60k')
FLOOR = 0.06877
HPC10K, BASE10K = 0.09102, 0.14576
BLOCK = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
GRID = dict(color='#DDDDDD', linewidth=0.6)


def style(ax, title, ylab, xlab='training step'):
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#999999')
    ax.tick_params(labelsize=8, color='#999999')


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def main():
    os.makedirs(OUT, exist_ok=True)
    j = json.load(open(os.path.join(R, RUN, 'run.json')))
    H, S = j['hist'], j['summary']
    nb = S['n_blocks']
    step = [r['step'] for r in H]

    def s(key):
        return [r[key] for r in H if key in r]

    nl = sum(1 for k in H[-1] if k.startswith('hpc/level'))

    # 1 -- per-level training loss, plus the headline
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i in range(nl):
        ax.plot(step, s(f'hpc/level{i}'), color=BLOCK[i], linewidth=1.4, label=f'level {i} train')
    ax.plot(step, s('eval/test_mse'), color='#000000', linewidth=1.8,
            label='summed reconstruction, TEST')
    ax.axhline(FLOOR, color='#000000', linestyle=':', linewidth=1.4,
               label=f'linear 784-128-784 floor ({FLOOR:.5f})')
    ax.set_yscale('log')
    style(ax, 'Per-level training loss and the headline test MSE\n'
              'levels 1-3 lying on top of level 0 means they explain none of its residual',
          'MSE, standardised (log)')
    ax.legend(fontsize=8, frameon=False)
    save(fig, '1_levels_and_headline.png')

    # 2 -- headline test vs the 10K reference points
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(step, s('eval/test_mse'), color='#0072B2', linewidth=1.8, label='HPC 60K, test')
    ax.plot(step, s('eval/train_mse'), color='#0072B2', linewidth=1.1, linestyle='--',
            label='HPC 60K, train')
    for y, c, lab in ((FLOOR, '#000000', f'linear floor {FLOOR:.5f}'),
                      (HPC10K, '#009E73', f'HPC at 10K {HPC10K:.5f}'),
                      (BASE10K, '#D55E00', f'baseline at 10K {BASE10K:.5f}')):
        ax.axhline(y, color=c, linestyle=':', linewidth=1.3, label=lab)
    ax.set_yscale('log')
    style(ax, 'Headline: test MSE of the summed reconstruction', 'MSE, standardised (log)')
    ax.legend(fontsize=8, frameon=False)
    save(fig, '2_headline.png')

    # 3 -- utilisation: participation ratio, entropy, dead fraction
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.6), sharex=True)
    for key, ax, (name, ylab) in zip(
            ('util/pr_mean', 'util/entropy_mean', 'util/dead_frac'), axes,
            (('Participation ratio 1/sum(p^2): effective rows in use, of 256',
              'effective rows'),
             ('Usage entropy, nats (uniform maximum log 256 = 5.545)', 'nats'),
             ('Dead-row fraction: rows untouched by the 512-sample probe batch', 'fraction'))):
        for b in range(nb):
            ax.plot(step, s(f'{key}_b{b}'), color=BLOCK[b], linewidth=1.5, label=f'block {b}')
        style(ax, name, ylab, xlab='' if key != 'util/dead_frac' else 'training step')
        ax.legend(fontsize=8, frameon=False, ncol=4)
    if 'util/entropy_max_b0' in H[-1]:
        axes[1].axhline(H[-1]['util/entropy_max_b0'], color='#000000', linestyle=':', linewidth=1.2)
    save(fig, '3_utilisation.png')

    # 4 -- gradient norms
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for k, c, lab in (('grad/enc_weight', '#0072B2', 'encoder'),
                      ('grad/b0_compress', '#D55E00', 'block-0 compress'),
                      ('grad/dec_weight', '#009E73', 'decoder'),
                      ('grad/tables_b0', '#CC79A7', 'tables, block 0')):
        if k in H[-1]:
            ax.plot(step, s(k), color=c, linewidth=1.5, label=lab)
    ax.set_yscale('log')
    style(ax, 'Gradient norms', 'gradient L2 norm (log)')
    ax.legend(fontsize=8, frameon=False)
    save(fig, '4_gradients.png')

    # 5 -- readout norm per block
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for b in range(nb):
        ax.plot(step, s(f'norm/t5_block_out_mean_b{b}'), color=BLOCK[b], linewidth=1.5,
                label=f'block {b} readout')
    style(ax, 'Readout norm after each block (mean per-sample L2)', 'mean ||h||')
    ax.legend(fontsize=8, frameon=False)
    save(fig, '5_readout_norms.png')

    # 6 -- table row norm, the geometry side of the collapse
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for b in range(nb):
        ax.plot(step, s(f'util/row_norm_mean_b{b}'), color=BLOCK[b], linewidth=1.5,
                label=f'block {b}')
    style(ax, 'Mean table row norm: a collapsed block has nothing left in its table', 'mean ||row||')
    ax.legend(fontsize=8, frameon=False)
    save(fig, '6_row_norms.png')

    m01 = S['test_mse'] * 0.3081 ** 2
    print(f'\nfinal test {S["test_mse"]:.5f}  MSE[0,1] {m01:.6f}  PSNR {10*math.log10(1/m01):.2f} dB')


if __name__ == '__main__':
    main()
