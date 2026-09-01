#!/usr/bin/env python3
"""Render the FFN-slot phase-breakdown figure from measured data.

    python paper_timings/phase_split.py --load-checkpoint   # writes results.json
    python paper_timings/make_figure.py                     # -> ffn_phase_split.png

Reads results.json rather than hardcoded numbers, so re-running the timings and
re-running this reproduces the committed figure with the new measurements. The
committed results.json is the RTX 5090 run described in README.md.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# One colour per phase, stable across models so the eye can compare rows. Warm hues for
# the LUT pipeline, cool for the dense baseline, neutral grey for unaccounted glue.
COLORS = {'Linear 384->1536 + GELU': '#4C72B0', 'Linear 1536->384': '#94AFD4',
          'compress 384->192': '#DD8452', 'routing + gather (fused)': '#C44E52',
          'decompress 192->384': '#EFC08A', 'other': '#BBBBBB'}
ORDER = ['vanilla', 'exp_n_0126', 'exp_n_0127', 'exp_n_0128']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'ffn_phase_split.png'))
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    with open(args.results) as fh:
        R = json.load(fh)
    models = R['models']
    names = [n for n in ORDER if n in models] + \
            [n for n in models if n not in ORDER]

    rows = []
    for n in names:
        m = models[n]
        if n == 'vanilla':
            lbl = f'vanilla dense\n({m["total"]:.3f} ms)'
        else:
            lbl = f'{n}\n({m["total"]:.3f} ms, {m["vs_vanilla"]:.2f}x)'
        rows.append((lbl, m['phases']))

    total_max = max(m['total'] for m in models.values())
    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    seen = []
    for y, (lbl, phases) in enumerate(rows):
        left = 0.0
        for pname, ms in phases:
            ax.barh(y, ms, left=left, height=0.62,
                    color=COLORS.get(pname, '#888888'),
                    edgecolor='white', linewidth=1.4,
                    label=pname if pname not in seen else None)
            if pname not in seen:
                seen.append(pname)
            if ms > total_max * 0.075:      # label only segments wide enough to read
                ax.text(left + ms / 2, y, f'{ms:.3f}', ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
            left += ms

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'FFN slot time (ms)  —  {R["gpu"].replace("NVIDIA GeForce ", "")}, '
                  f'batch {R["batch"]} x seq {R["seq"]} = {R["tokens"]:,} tokens, '
                  f'{R["dtype"]}', fontsize=9)
    ax.set_xlim(0, total_max * 1.16)
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), fontsize=8.5,
              frameon=False, ncol=3)
    ax.set_title('FFN-slot phase breakdown: vanilla dense vs LUT models (fused path)',
                 fontsize=10.5, pad=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
