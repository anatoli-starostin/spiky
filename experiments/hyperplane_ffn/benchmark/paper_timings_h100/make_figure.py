#!/usr/bin/env python3
"""Portable generator for the H100 FFN-slot phase-breakdown figure.

Reads a paper_timing.json (produced by the H100 timing run) and emits a stacked bar
chart of the per-phase timing across vanilla dense and the three routed CompressionMHL
variants (fused_v2 best path). No machine-specific paths.

Usage:  python make_phase_figure.py [paper_timing.json] [out.png]
Defaults: ./paper_timing.json -> ./h100_phase_breakdown.png
"""
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', os.environ.get('MPLCONFIGDIR', '/tmp/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, 'results.json')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_here, 'ffn_phase_split_h100.png')
J = json.load(open(src))
cond = J['conditions']
v = J['vanilla']
routed = J['routed']

C_PROJ_IN = '#3b6fb0'    # up-proj / compress
C_PROJ_OUT = '#8fb8e0'   # down-proj / decompress
C_SPECIAL = '#e08a3b'    # routing+gather (fused) — the LUT-specific cost
C_GELU = '#4caf7d'       # vanilla GELU

labels, stacks = [], []   # each stack: list of (height, color, seg_label)
labels.append('vanilla\n(dense bf16)')
stacks.append([(v['up'], C_PROJ_IN, 'up 384->1536'),
               (v['gelu'], C_GELU, 'GELU'),
               (v['down'], C_PROJ_OUT, 'down 1536->384')])
for m in ('0126', '0127', '0128'):
    r = routed[m]
    labels.append(f'{m}\nnap{r["nap"]}/tph{r["tph"]}')
    stacks.append([(r['compress'], C_PROJ_IN, 'compress 384->192'),
                   (r['fused_route_gather'], C_SPECIAL, 'routing+gather (fused)'),
                   (r['decompress'], C_PROJ_OUT, 'decompress 192->384')])

fig, ax = plt.subplots(figsize=(9, 5.5))
xs = range(len(labels))
seen = set()
for i, stack in enumerate(stacks):
    bottom = 0.0
    for h, col, lab in stack:
        lg = lab if lab not in seen else None
        seen.add(lab)
        ax.bar(i, h, bottom=bottom, color=col, edgecolor='white', width=0.62, label=lg)
        bottom += h
    ax.text(i, bottom + 0.004, f'{bottom:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

van_total = v['up'] + v['gelu'] + v['down']
ax.axhline(van_total, ls='--', lw=1, color='#666',
           label=f'vanilla total ({van_total:.3f} ms)')
ax.set_xticks(list(xs))
ax.set_xticklabels(labels)
ax.set_ylabel('FFN-slot time (ms/call)')
ax.set_title(f'H100 FFN-slot phase breakdown — batch {cond["batch"]}x{cond["seq"]} '
             f'({cond["tokens"]:,} tok), {cond["gpu"].split("NVIDIA ")[-1]}\n'
             f'routed = fused_v2 (routing+gather fused, index in shared); '
             f'ratios are the citable comparison')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=140)
print(f'wrote {out}')
for m in ('0126', '0127', '0128'):
    r = routed[m]
    print(f'{m}: slot {r["slot"]:.4f} ms = {r["vs_vanilla"]:.2f}x vanilla '
          f'(compress {r["compress"]:.4f} + fused {r["fused_route_gather"]:.4f} + decompress {r["decompress"]:.4f})')
