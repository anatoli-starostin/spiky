#!/usr/bin/env python3
"""Paper exp_n_0127 + vanilla (published, and re-measured today) next to the int8 / config-B rows, at BOTH
float32_matmul_precision settings. Layout and colours follow paper/make_bench_figs.py (bench_combined.pdf): horizontal
stacked bars per stage, white stage values, bold slot total after the bar, ratio under the label, "other" glue not
drawn (the total annotation is the true measured slot total).

Reads only measured data:
  ../results.json                          the paper's published 5090 run (phase_split.py, precision 'high')
  ../verify_out/fig0127v2_{high,highest}_r{1,2,3}.json p2int8_paper0127_v2.py, 3 launches per precision (after the four
                                           library kernel steps; config B BEFORE = kernel_before_step0/, AFTER = the library)
The pre-step-4 version of this figure and its script are kept as *_pre_step4.*.
Bars = mean over the 3 launches of the per-launch interleaved medians (P1, harness timeit).

    python make_fig_0127_vs_int8.py      -> fig_0127_vs_int8.pdf / .png
"""
import json
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.patches as mp          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
VO = os.path.join(PT, 'verify_out')

# paper colours (make_bench_figs.py), plus two stages the paper never had
COL = {'Linear 384->1536 + GELU': '#4C72B0', 'Linear 1536->384': '#A6C0E0',
       'compress 384->192': '#DD8452', 'routing + gather (fused)': '#C44E52',
       'decompress 192->384': '#EFC08A', 'int8 read (kernel)': '#8172B3', 'dtype casts': '#9A9A9A'}
LEGEND = ['Linear 384->1536 + GELU', 'Linear 1536->384', 'compress 384->192', 'routing + gather (fused)',
          'int8 read (kernel)', 'decompress 192->384', 'dtype casts']
FP32_MATMUL = {'V32', 'P127_32', 'Q32', 'T16'}          # rows whose compress/decompress/MLP matmuls run fp32
FP32_MATMUL_STAGES = {'Linear 384->1536 + GELU', 'Linear 1536->384', 'compress 384->192', 'decompress 192->384'}
LABEL = {'V16': 'vanilla dense, bf16', 'V32': 'vanilla dense, fp32',
         'P127_16': 'exp_n_0127, paper hybrid bf16', 'P127_32': 'exp_n_0127, fp32',
         'Q32': 'int8 read (abl_48), fp32, current kernel',
         'B0': 'BEFORE  int8 read, config B, pre-optimisation kernel',
         'B4': 'AFTER  int8 read, config B, bf16 in/out, current kernel',
         'T16': "int8 read, fp32 module in bf16 (casts), current kernel"}
ROWS = ['V16', 'P127_16', 'B0', 'B4', 'V32', 'P127_32', 'Q32', 'T16']
FADED = {'B0'}
# geometry, verified from the run configs and checkpoints (exp_n_0127: FastMultiHeadLut, lut_n_anchor_pairs 7, tables
# [512, 128, 48], lut_forward_mode hard, no read_top_n / confidence keys; abl_48: lut_impl light, lut_n_anchor_pairs 8,
# tables [512, 256, 48], lut_read_top_n 2, confidence learned_margin, lut_quant_mode p2_int8; abl_10 / exp_n_0135: dense)
G_DENSE = 'dense FFN, no LUT'
G_0127 = 'LUT nap 7: 128 cells/table, 4x128 tables, 1 hard cell read per table, no score'
G_INT8 = 'LUT nap 8: 256 cells/table, 4x128 tables, 2 scored cells read per table, int8'
GEOM = {'V16': G_DENSE, 'V32': G_DENSE, 'P127_16': G_0127, 'P127_32': G_0127,
        'Q32': G_INT8, 'B0': G_INT8, 'B4': G_INT8, 'T16': G_INT8}


def load(prec):
    runs = [json.load(open(os.path.join(VO, f'fig0127v2_{prec}_r{i}.json'))) for i in (1, 2, 3)]
    for r in runs:
        assert r['precision_asserted'] == prec and r['checks']['gate_pass']
    med = lambda k: statistics.mean(r['median_ms'][k]['P1']['median'] for r in runs)   # noqa: E731
    out = {}
    for row in ROWS:
        stages = []
        for k in runs[0]['median_ms']:
            if k.startswith(row + '|'):
                s = k.split('|', 1)[1]
                stages.append(('dtype casts' if s.startswith('dtype casts') else s, med(k)))
        out[row] = (stages, med(row))
    return out


def published():
    """The paper's published rows, then the same rows re-run today with the paper's own phase_split.py (randn input,
    3 launches; ../verify_out/paper_repro_phase_split_r{1,2,3}.json)."""
    d = json.load(open(os.path.join(PT, 'results.json')))['models']
    rows = []
    for k, lbl, gm in (('vanilla', 'vanilla dense, bf16', G_DENSE), ('exp_n_0127', 'exp_n_0127, paper hybrid bf16', G_0127)):
        rows.append((f'PUBLISHED  {lbl}\n[{gm}]', [(p, v) for p, v in d[k]['phases'] if p != 'other'], d[k]['total'],
                     d[k]['total'] / d['vanilla']['total'], False, True))
    rep = [json.load(open(os.path.join(VO, f'paper_repro_phase_split_r{i}.json')))['models'] for i in (1, 2, 3)]
    rv = statistics.mean(r['vanilla']['total'] for r in rep)
    for k, lbl, gm in (('vanilla', 'vanilla dense, bf16', G_DENSE), ('exp_n_0127', 'exp_n_0127, paper hybrid bf16', G_0127)):
        ph = [(p, statistics.mean(r[k]['phases'][i][1] for r in rep)) for i, (p, _) in enumerate(rep[0][k]['phases'])
              if p != 'other']
        tot = statistics.mean(r[k]['total'] for r in rep)
        rows.append((f'REPRODUCED today, paper harness (randn)  {lbl}\n[{gm}]', ph, tot, tot / rv, False, True))
    return rows


def draw(ax, bars, title, xmax):
    ys = list(range(len(bars)))[::-1]
    ylabels = []
    for y, (name, stages, total, ratio, tf32, pub) in zip(ys, bars):
        left = 0.0
        for st, w in stages:
            hatch_tf32 = tf32 and st in FP32_MATMUL_STAGES
            ax.barh(y, w, left=left, height=0.60, color=COL[st], edgecolor='white', linewidth=0.7,
                    hatch='///' if hatch_tf32 else None, alpha=0.55 if pub else 1.0)
            if w > 0.095 * xmax:
                ax.text(left + w / 2, y, f'{w:.3f}', ha='center', va='center', fontsize=10, color='white',
                        fontweight='bold')
            left += w
        ax.text(max(left, total) + 0.008 * xmax, y, f'{total:.3f}', ha='left', va='center', fontsize=11,
                fontweight='bold', color='#333')
        ylabels.append(f'{name}\n(time {ratio:.2f}$\\times$ vanilla bf16)')
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels(ylabels[::-1], fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('FFN-slot time (ms/call)', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', ls=':', lw=0.5, alpha=0.5)


def panel_bars(data, prec, with_published):
    bars = published() if with_published else []
    base = data['V16'][1]
    for row in ROWS:
        stages, total = data[row]
        tf32 = prec == 'high' and row in FP32_MATMUL
        name = LABEL[row] + ('  [TF32]' if tf32 else '') + ('  (today)' if with_published else '') + f'\n[{GEOM[row]}]'
        bars.append((name, stages, total, total / base, tf32, row in FADED))
    return bars


def main():
    hi, hst = load('high'), load('highest')
    xmax = 1.14 * max(max(t for _, t in hi.values()), max(t for _, t in hst.values()))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(19.5, 10.5))
    draw(axL, panel_bars(hi, 'high', True), "RTX 5090, float32_matmul_precision='high'\n"
         "(the paper harness's setting: TF32 ON for fp32 matmuls; hatched = TF32 matmul)", xmax)
    draw(axR, panel_bars(hst, 'highest', False), "RTX 5090, float32_matmul_precision='highest'\n"
         "(the trainer's setting: true fp32 matmuls)", xmax)
    handles = [mp.Patch(color=COL[s], label=s) for s in LEGEND]
    handles.append(mp.Patch(facecolor='white', edgecolor='#555', hatch='///', label='fp32 matmul run as TF32'))
    handles.append(mp.Patch(color='#888', alpha=0.55, label='faded = paper rows (published / reproduced) and the BEFORE row'))
    fig.legend(handles=handles, fontsize=10.5, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.055), framealpha=0.95)
    fig.suptitle('FFN-slot phase breakdown (batch 48 x 512 = 24,576 tokens): paper exp_n_0127 vs the int8 read, '
                 'at both matmul precisions', fontsize=14)
    fig.text(0.5, 0.030, 'NOT GEOMETRY-MATCHED: the int8 rows read TWO scored cells per table from a 256-cell table (nap 8), '
             "exp_n_0127 reads ONE hard cell per table from a 128-cell table (nap 7), so the int8 rows do strictly more work per token.",
             ha='center', va='bottom', fontsize=11, color='#8B0000', fontweight='bold')
    fig.text(0.5, 0.005, 'All "today" rows, including BEFORE / AFTER, measured in one interleave per precision (3 fresh launches, '
             'mean of per-launch medians); every stage segment is a measured stage timing, none inferred.',
             ha='center', va='bottom', fontsize=9.5, color='#444')
    fig.tight_layout(rect=[0, 0.17, 1, 0.95])
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_0127_vs_int8.{ext}'), bbox_inches='tight', pad_inches=0.08, dpi=150)
    print('saved fig_0127_vs_int8.pdf / .png')
    for prec, d in (('high', hi), ('highest', hst)):
        print(prec, {r: round(t, 4) for r, (_, t) in d.items()})


if __name__ == '__main__':
    main()
