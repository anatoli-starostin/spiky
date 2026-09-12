"""sharp_margin arms vs the references: matched-step eval curves, final deltas in noise units, and the
preregistered bins (doc/research/lut_ablation/notes_sharp_margin.md). Read-only.

    python cmp_sharp.py [--png out.png]
"""
import csv
import json
import os
import sys

RC = '/home/astarostin/projects/spiky/experiments/ffn_replacement/runs_corrected'
RUNS = {
    '0193 margin': 'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1',
    '0243 min_margin': 'exp_g_0243_B16k_light_minmargin_gain37p4_tph128_seed1',
    '0244 tanh_margin': 'exp_g_0244_B16k_light_tanhmargin_a2_gain1_tph128_seed1',
    '0195 n=2': 'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1',
    '0245 sharp g1.75': 'exp_g_0245_B16k_light_sharpmargin_g1p75_gain3p9_tph128_seed1',
    '0246 sharp g3': 'exp_g_0246_B16k_light_sharpmargin_g3_gain25p4_tph128_seed1',
    '0247 learned': 'exp_g_0247_B16k_light_learnedmargin_tph128_seed1',
    '0248 frozen-g': 'exp_g_0248_B16k_light_learnedmargin_frozeng_tph128_seed1',
    '0249 frozen-g TV10': 'exp_g_0249_B16k_light_learnedmargin_frozeng_tv10_tph128_seed1',
}
LEARNED_REFS = ('0193 margin', '0245 sharp g1.75', '0244 tanh_margin', '0243 min_margin', '0195 n=2')
LEARNED_RUNS = {'0247 learned': LEARNED_REFS, '0248 frozen-g': ('0247 learned',) + LEARNED_REFS,
                '0249 frozen-g TV10': ('0248 frozen-g', '0247 learned', '0193 margin')}
PARTNER = {'0245 sharp g1.75': '0243 min_margin', '0246 sharp g3': '0244 tanh_margin'}
NOISE = [('vanilla 2-seed range', 0.00335), ('budget-law resid sd', 0.0035), ('4K LUT 3-seed sd (lower bnd)', 0.009642)]
BIN = 0.0035


def curve(name):
    p = os.path.join(RC, RUNS[name], 'metrics.csv')
    if not os.path.exists(p):
        return {}
    return {int(r['step']): float(r['val_bpb']) for r in csv.DictReader(open(p)) if r.get('val_bpb')}


def final(name):
    p = os.path.join(RC, RUNS[name], 'summary.json')
    return json.load(open(p)) if os.path.exists(p) else None


curves = {k: curve(k) for k in RUNS}
present = [k for k in RUNS if curves[k]]
steps = sorted(set().union(*[set(curves[k]) for k in present]))
print('eval-step curve (corrected in-run eval, bs48x100 skip-12)')
print(f'{"step":>6} ' + ' '.join(f'{k:>17}' for k in present))
for s in steps:
    print(f'{s:>6} ' + ' '.join(f'{curves[k][s]:>17.6f}' if s in curves[k] else f'{"-":>17}' for k in present))

print()
for arm, partner in PARTNER.items():
    sm = final(arm)
    if not sm:
        print(f'{arm}: no summary.json yet')
        continue
    b = sm['final_val_bpb']
    ref = {k: final(k)['final_val_bpb'] for k in ('0193 margin', '0243 min_margin', '0244 tanh_margin', '0195 n=2')}
    print(f'{arm}: final_val_bpb {b:.6f}  ({sm["training_time_hours"]} h, {sm["total_params"]:,} params)')
    for k, v in ref.items():
        dd = b - v
        print(f'   vs {k:<17} {v:.6f}: {dd:+.6f} = ' + ', '.join(f'{dd / u:+.2f} x {n}' for n, u in NOISE))
    m, p = ref['0193 margin'], ref[partner]
    lo, hi = min(m, p), max(m, p)
    if abs(b - m) <= BIN:
        verdict = 'NEAR MARGIN -> continuity costs performance'
    elif abs(b - p) <= BIN:
        verdict = f'NEAR ITS PARTNER ({partner}) -> sharp gating is simply worse'
    elif m + BIN < b < p - BIN or p + BIN < b < m - BIN:
        verdict = 'INTERMEDIATE -> both contribute'
    else:
        verdict = 'OUTSIDE both references (not forced into a bin)'
    print(f'   bins: margin [{m - BIN:.4f}, {m + BIN:.4f}]  partner [{p - BIN:.4f}, {p + BIN:.4f}]  -> {verdict}')
    print(f'   position (bpb - margin) / (partner - margin) = {(b - m) / (p - m):.3f}')
    shared = [s for s in sorted(curves[arm]) if s in curves['0193 margin'] and s in curves[partner]]
    if shared:
        worse_m = sum(curves[arm][s] > curves['0193 margin'][s] for s in shared)
        better_p = sum(curves[arm][s] < curves[partner][s] for s in shared)
        print(f'   matched steps: worse than margin at {worse_m}/{len(shared)}, better than partner at '
              f'{better_p}/{len(shared)}')

for run, refs in LEARNED_RUNS.items():
    sm = final(run)
    if not sm:
        continue
    b = sm['final_val_bpb']
    print(f'{run}: final_val_bpb {b:.6f}  ({sm["training_time_hours"]} h, {sm["total_params"]:,} params)')
    for k in refs:
        v = final(k)['final_val_bpb']
        dd = b - v
        print(f'   vs {k:<17} {v:.6f}: {dd:+.6f} = ' + ', '.join(f'{dd / u:+.2f} x {n}' for n, u in NOISE))
    for refname in ('0193 margin', '0247 learned'):
        if refname == run or not curves[refname]:
            continue
        ref = curves[refname]
        shared = [s for s in sorted(curves[run]) if s in ref]
        deltas = [curves[run][s] - ref[s] for s in shared]
        print(f'   matched steps vs {refname}: worse at {sum(d > 0 for d in deltas)}/{len(shared)}; '
              f'delta at 500 {deltas[0]:+.6f}, 4000 {deltas[shared.index(4000)]:+.6f}, '
              f'8000 {deltas[shared.index(8000)]:+.6f}, 16000 {deltas[-1]:+.6f}')

if '--png' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = sys.argv[sys.argv.index('--png') + 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for k in present:
        xs = sorted(curves[k])
        a1.plot(xs, [curves[k][s] for s in xs], label=k, lw=1.6)
        ref = curves['0193 margin']
        xs2 = [s for s in xs if s in ref]
        a2.plot(xs2, [curves[k][s] - ref[s] for s in xs2], label=k, lw=1.6)
    a1.set(xlabel='step', ylabel='val bpb', title='Corrected eval curve', ylim=(1.14, 1.40))
    a2.axhspan(-BIN, BIN, color='0.85', label='+/-0.0035 bin')
    a2.set(xlabel='step', ylabel='bpb - exp_g_0193', title='Matched-step delta vs margin (0193)')
    for a in (a1, a2):
        a.grid(True, alpha=.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print('wrote', out)
