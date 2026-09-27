"""Tokens/s vs batch size for every kernel arm, and the L2 footprint curve.

Conventions, as in the rest of this work: Okabe-Ito colourblind-safe hues (yellow
dropped, it fails against white), one y-axis per panel and never a second scale, a legend
plus a direct label on the last point of each series, recessive grid and axes, and no
sharpening or rescaling of anything. The dataviz palette validator is NOT installed on
this machine, so the palette is the hand-checked Okabe-Ito set rather than a
script-validated one.
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

# Okabe-Ito, yellow dropped
COLOR = {'gather': '#000000', 'v0': '#D55E00', 'v1': '#CC79A7',
         'v2': '#0072B2', 'v3': '#009E73'}
NAME = {'gather': 'gather (baseline)', 'v0': 'cell-stationary v0 (naive)',
        'v1': 'cell-stationary v1 (as specified)',
        'v2': 'cell-stationary v2 (conflict-free reduction)',
        'v3': 'cell-stationary v3 (G tables/thread)'}
SHAPE = {'B': 'paper reference config: 256 tables x 256 rows x 1024 lanes (64 MiB int8)',
         'A': 'repo real shape: one head of blocks.0.ffn.lut_light, 256 x 256 x 48 (3 MiB)'}


def style(ax, title, xlab, ylab):
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, color='#DDDDDD', linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#999999')
    ax.tick_params(labelsize=8, color='#999999')


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(os.path.join(ART, 'bench.json')))
    rows = res['ladder']

    for use_coef in (True, False):
        shapes = [s for s in ('B', 'A') if any(r['shape'] == s for r in rows)]
        if not shapes:
            continue
        fig, axes = plt.subplots(1, len(shapes), figsize=(6.6 * len(shapes), 5.0))
        if len(shapes) == 1:
            axes = [axes]
        for ax, sh in zip(axes, shapes):
            sel = [r for r in rows if r['shape'] == sh and r['use_coef'] == use_coef]
            fams = [f for f in ('gather', 'v0', 'v1', 'v2', 'v3')
                    if any(r['family'] == f for r in sel)]
            for f in fams:
                pts = sorted((r['B'], r['tokens_per_s']) for r in sel if r['family'] == f)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=COLOR[f], linewidth=1.8, marker='o', markersize=4,
                        label=NAME[f])
                ax.annotate(f.replace('gather', 'gather'), (xs[-1], ys[-1]),
                            textcoords='offset points', xytext=(6, 0), fontsize=7.5,
                            color=COLOR[f], va='center')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            style(ax, SHAPE[sh], 'batch size B (tokens, log2)', 'tokens/s (log)')
            ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        fig.suptitle(f'One LUT-Core evaluation per token, top-1, int8 tables, '
                     f'coefficient = {"the trained margin score" if use_coef else "1"}'
                     f'\nRTX 5090, warmed clocks, median of >=50 iterations',
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        p = os.path.join(OUT, f'tokens_per_s_c{"real" if use_coef else "1"}.png')
        fig.savefig(p, dpi=175, bbox_inches='tight')
        plt.close(fig)
        print('wrote', p)

    # ---- L2 footprint curve ----
    lp = os.path.join(ART, 'l2.json')
    if os.path.exists(lp):
        d = json.load(open(lp))
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        xs = [o['footprint_bytes'] / 2 ** 20 for o in d]
        ys = [o['request_gbs'] for o in d]
        ax.plot(xs, ys, color='#0072B2', linewidth=1.9, marker='o', markersize=5,
                label='gather baseline, requested table bytes / time')
        ax.axhline(1792, color='#000000', linestyle=':', linewidth=1.4,
                   label='HBM peak, 1,792 GB/s')
        ax.axvline(96, color='#D55E00', linestyle='--', linewidth=1.4,
                   label='L2 capacity, 96 MB')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        style(ax, 'The baseline reads its tables out of L2, not HBM\n'
                  'request rate above the HBM ceiling is cache amortisation, measured '
                  'without ncu',
              'table-set footprint (MiB, log2)', 'requested bytes/s (GB/s, log)')
        ax.legend(fontsize=8, frameon=False)
        p = os.path.join(OUT, 'l2_footprint.png')
        fig.savefig(p, dpi=175, bbox_inches='tight')
        plt.close(fig)
        print('wrote', p)

    # ---- dedup vs G ----
    sp = os.path.join(ART, 'index_stats.json')
    if os.path.exists(sp):
        st = json.load(open(sp))
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        g = [d['G'] for d in st['dedup']]
        ax.plot(g, [d['empirical_dedup'] for d in st['dedup']], color='#0072B2',
                linewidth=1.9, marker='o', markersize=5,
                label='measured, real trained indices at B=24,576')
        ax.plot(g, [d['birthday_dedup'] for d in st['dedup']], color='#D55E00',
                linewidth=1.6, linestyle='--', marker='s', markersize=4,
                label='birthday bound for independent uniform indices')
        ax.set_xscale('log', base=2)
        style(ax, 'Intra-thread dedup: hits per emitted atomic for a thread owning row r '
                  'across G tables\nthe measured curve sits on the independence bound, so '
                  'hot rows are not aligned across tables',
              'tables per thread G (log2)', 'dedup factor')
        ax.legend(fontsize=8, frameon=False)
        p = os.path.join(OUT, 'dedup_vs_g.png')
        fig.savefig(p, dpi=175, bbox_inches='tight')
        plt.close(fig)
        print('wrote', p)


if __name__ == '__main__':
    main()
