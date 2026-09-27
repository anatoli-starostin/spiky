"""The dedup-vs-G figure on its own, so it can be produced before the benchmark finishes.

Okabe-Ito hues, one y-axis, legend present, recessive grid. The dataviz palette validator
is not installed on this machine, so the palette is the hand-checked Okabe-Ito set.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, 'artifacts')
OUT = os.path.join(HERE, 'figs')


def main():
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(os.path.join(ART, 'index_stats.json')))
    d = st['dedup']
    g = [x['G'] for x in d]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax.plot(g, [x['empirical_dedup'] for x in d], color='#0072B2', linewidth=2.0,
            marker='o', markersize=6, label='measured, real trained indices, B=24,576')
    ax.plot(g, [x['birthday_dedup'] for x in d], color='#D55E00', linewidth=1.6,
            linestyle='--', marker='s', markersize=5,
            label='birthday bound, independent uniform indices')
    ax.set_xscale('log', base=2)
    ax.set_xticks(g)
    ax.set_xticklabels([str(x) for x in g])
    for a, t in ((ax, 'Intra-thread dedup: hits per emitted atomic, thread owning row r '
                      'across G tables'),
                 (ax2, 'Row-usage histogram of each table (head 0), 256 tables overlaid')):
        a.set_title(t, fontsize=10, pad=8)
        a.grid(True, color='#DDDDDD', linewidth=0.6)
        a.set_axisbelow(True)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a.spines[s].set_color('#999999')
        a.tick_params(labelsize=8, color='#999999')
    ax.set_xlabel('tables per thread G (log2)', fontsize=9)
    ax.set_ylabel('dedup factor (hits / atomics)', fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax.annotate('measured sits ON the independence bound:\nhot rows are not aligned across '
                'tables,\nso G > 1 buys at most 1.61x even at G = 256',
                xy=(0.52, 0.28), xycoords='axes fraction', fontsize=8.5, color='#333333')

    import torch
    h = torch.load(os.path.join(ART, 'row_hist.pt'))['row_hist_head0']
    p = (h / h.sum(1, keepdim=True)).numpy()
    for i in range(0, p.shape[0], 4):
        ax2.plot(p[i], color='#0072B2', linewidth=0.4, alpha=0.18)
    ax2.plot(p.mean(0), color='#000000', linewidth=1.8, label='mean over the 256 tables')
    ax2.axhline(1.0 / p.shape[1], color='#D55E00', linestyle=':', linewidth=1.5,
                label=f'uniform, 1/256 = {1/256:.4f}')
    ax2.set_xlabel('row index (0..255)', fontsize=9)
    ax2.set_ylabel('share of the 24,576 tokens', fontsize=9)
    ax2.legend(fontsize=8.5, frameon=False)

    fig.suptitle('Real index distribution of blocks.0.ffn.lut_light in exp_n_0196, top-1, '
                 'B = 24,576 real tokens', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    q = os.path.join(OUT, 'dedup_and_row_usage.png')
    fig.savefig(q, dpi=175, bbox_inches='tight')
    print('wrote', q)


if __name__ == '__main__':
    main()
