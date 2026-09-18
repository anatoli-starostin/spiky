"""Analyse the matrix-menu tph sweep (run_menu_tph_sweep.sh) against runs/ladder4k_H8.

  python analyze_menu_tph_sweep.py   ->  runs/menu_tph_sweep_fvu.png, runs/menu_tph_sweep_usage.png,
                                         runs/menu_tph_sweep_summary.json (and prints the tables)

FVU: final (step 4000) held-out FVU per layer from each run's results.json -- the harness's own metric.
Menu usage (menu runs, from the saved students): per head, over that head's tph x 256 cells, the argmax menu item
of each cell; reported as #items used (of 64), usage entropy and its exponential (effective #items). Also the
TOKEN-weighted version on the harness's held-out val slab: the score-weighted hard menu distribution a[n,h,:]
summed over tokens (what the forward actually applies), same statistics.
"""
import json, math, os, sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_ffn as D  # noqa: E402

R = os.path.join(HERE, 'runs')
RUNS = [('ladder4k_H8', 'Light tph128 (ref)'), ('light4k_H8_tph32', 'Light tph32'), ('light4k_H8_tph8', 'Light tph8'),
        ('menu4k_H8_tph128', 'menu tph128'), ('menu4k_H8_tph32', 'menu tph32'), ('menu4k_H8_tph16', 'menu tph16'),
        ('menu4k_H8_tph8', 'menu tph8')]


def usage_stats(counts):
    c = counts.double()
    p = c / c.sum()
    nz = p[p > 0]
    H = -(nz * nz.log()).sum().item()
    return {'used': int((c > 0).sum()), 'entropy_nats': H, 'eff_items': math.exp(H)}


summary = {}
for run, label in RUNS:
    p = os.path.join(R, run, 'results.json')
    if not os.path.exists(p):
        print('missing', run); continue
    j = json.load(open(p))
    summary[run] = {'label': label, 'fvu': [j['final'][str(li)]['fvu'] for li in range(6)],
                    'wall_min': j['wall_clock_s'] / 60, 'peak_gib': j.get('peak_gpu_mem_gib'),
                    'train_mse': [], 'val_mse': [j['final'][str(li)]['mse'] for li in range(6)]}
    summary[run]['mean_fvu'] = sum(summary[run]['fvu']) / 6
    import csv
    rows = list(csv.DictReader(open(os.path.join(R, run, 'curves.csv'))))
    last = {int(r['layer']): r for r in rows if int(r['step']) == 4000}
    summary[run]['train_mse'] = [float(last[li]['train_mse']) for li in range(6)]
    nonfinite = [r for r in rows if not all(math.isfinite(float(r[k])) for k in ('val_mse', 'val_fvu'))]
    summary[run]['nonfinite_evals'] = len(nonfinite)
    # step-to-step: any eval where FVU rose > 10% vs the previous eval (odd behaviour flag)
    rises = []
    for li in range(6):
        seq = [(int(r['step']), float(r['val_fvu'])) for r in rows if int(r['layer']) == li and int(r['step']) >= 1000]
        for (s0, f0), (s1, f1) in zip(seq, seq[1:]):
            if f1 > f0 * 1.10:
                rises.append((li, s1, round(f0, 4), round(f1, 4)))
    summary[run]['fvu_rises_gt10pct'] = rises

print(f'{"run":22s} ' + ' '.join(f'{"L" + str(i):>7s}' for i in range(6)) + f' {"mean":>7s} {"wall":>7s}')
for run, s in summary.items():
    print(f'{s["label"]:22s} ' + ' '.join(f'{v:7.4f}' for v in s['fvu']) + f' {s["mean_fvu"]:7.4f} {s["wall_min"]:6.1f}m'
          + (f'  RISES {s["fvu_rises_gt10pct"]}' if s['fvu_rises_gt10pct'] else '')
          + (f'  NONFINITE {s["nonfinite_evals"]}' if s['nonfinite_evals'] else ''))

# ---- menu usage ------------------------------------------------------------------------------------------------
dev = 'cuda'
base = json.load(open(D.DEF_STUDENT))
tok = D.RustBPETokenizer.from_directory(os.path.join(D.get_base_dir(), 'tokenizer'))
teacher, tcfg = D.load_teacher(D.DEF_TEACHER, tok.get_vocab_size(), dev)
val_idx = D.take_rows(D.tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val', device=dev),
                      32, skip_rows=12)
io = D.teacher_ffn_io(teacher, val_idx, list(range(6)))
usage = {}
for run, label in RUNS:
    if not run.startswith('menu') or run not in summary:
        continue
    man = json.load(open(os.path.join(R, run, 'manifest.json')))
    ov = man['student_overrides']
    usage[run] = {}
    for li in range(6):
        s = D.build_student({**base, **ov}, 384, 6, li, dev)
        s.load_state_dict(torch.load(os.path.join(R, run, f'student_L{li}.pt'), map_location=dev))
        s.eval()
        lut = s.lut_light
        H, T, M = lut.n_heads, lut.tables_per_head, lut.menu_size
        am = lut.menu_logits.argmax(-1).view(H, T * lut.table_size)             # per head, all its cells
        cell = [usage_stats(torch.bincount(am[h], minlength=M)) for h in range(H)]
        with torch.no_grad():
            h_in = io[li][0]
            a = torch.cat([lut.menu_weights(s.compress(h_in[i:i + 4096]).view(-1, H, lut.input_dim))
                           for i in range(0, h_in.shape[0], 4096)])               # [N, H, M] hard, score-weighted
        tokw = [usage_stats(a[:, h].abs().sum(0)) for h in range(H)]
        tau = lut.menu_tau.tolist()
        # did the logits move? rebuild the untrained student (same seeds -> identical init) and compare
        s0 = D.build_student({**base, **ov}, 384, 6, li, dev)
        L0, L1 = s0.lut_light.menu_logits.detach(), lut.menu_logits.detach()
        drift = {'argmax_same_frac': (L0.argmax(-1) == L1.argmax(-1)).float().mean().item(),
                 'logit_std_init': L0.std().item(), 'logit_std_final': L1.std().item(),
                 'max_abs_change': (L1 - L0).abs().max().item(),
                 'rms_change': (L1 - L0).pow(2).mean().sqrt().item(),
                 'menu_std_final': lut.menu.std().item()}
        usage[run][li] = {'cell': cell, 'token': tokw, 'tau': tau, 'drift': drift,
                          'hist_tokens': a.abs().sum(0).cpu().tolist()}
        del s, s0
    print(f'--- {summary[run]["label"]}: per layer, mean over 8 heads: cell-argmax used/64, eff items (cell), '
          f'eff items (token-weighted), tau range')
    for li in range(6):
        u = usage[run][li]
        mc = sum(x['used'] for x in u['cell']) / 8
        ec = sum(x['eff_items'] for x in u['cell']) / 8
        et = sum(x['eff_items'] for x in u['token']) / 8
        mn = min(x['used'] for x in u['cell'])
        dr = u['drift']
        print(f'  L{li}: used {mc:5.1f}/64 (min head {mn}) | eff(cell) {ec:5.1f} | eff(token) {et:5.1f} | '
              f'tau {min(u["tau"]):.4f}-{max(u["tau"]):.4f} | argmax unchanged {dr["argmax_same_frac"]:.4f} | '
              f'logit rms change {dr["rms_change"]:.2e} (max {dr["max_abs_change"]:.2e}) | menu std {dr["menu_std_final"]:.2e}')
summary['_usage'] = {r: {li: {k: v for k, v in d.items() if k != 'hist_tokens'} for li, d in u.items()}
                     for r, u in usage.items()}
json.dump(summary, open(os.path.join(R, 'menu_tph_sweep_summary.json'), 'w'), indent=1)

# ---- charts ----------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})
colors = {'ladder4k_H8': '#1f3b73', 'light4k_H8_tph32': '#4a78c2', 'light4k_H8_tph8': '#9bbbe8',
          'menu4k_H8_tph128': '#8c1d18', 'menu4k_H8_tph32': '#c9453b', 'menu4k_H8_tph16': '#e8836f',
          'menu4k_H8_tph8': '#f4b9a8'}
runs = [r for r, _ in RUNS if r in summary]
w = 0.8 / len(runs)
for k, run in enumerate(runs):
    xs = [li + (k - (len(runs) - 1) / 2) * w for li in range(6)]
    axs[0].bar(xs, summary[run]['fvu'], width=w, color=colors[run], label=summary[run]['label'],
               edgecolor='white', linewidth=0.5)
axs[0].set_xticks(range(6)); axs[0].set_xticklabels([f'L{i}' for i in range(6)])
axs[0].set_ylabel('held-out FVU at step 4000 (lower is better)')
axs[0].set_title('Per-layer FVU: Light (blue) vs matrix menu M=64 (red), H8, by tables/head')
axs[0].legend(fontsize=9, ncol=2); axs[0].grid(axis='y', ls=':', alpha=0.5)
for fam, marker, keys in (('Light', 'o', ['ladder4k_H8', 'light4k_H8_tph32', 'light4k_H8_tph8']),
                          ('menu', 's', ['menu4k_H8_tph128', 'menu4k_H8_tph32', 'menu4k_H8_tph16', 'menu4k_H8_tph8'])):
    pts = [(int(summary[r]['label'].split('tph')[1].split()[0]), summary[r]['mean_fvu']) for r in keys if r in summary]
    pts.sort()
    axs[1].plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, lw=2,
                color='#1f3b73' if fam == 'Light' else '#8c1d18', label=fam)
    for x, y in pts:
        axs[1].annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(4, 6), fontsize=8)
axs[1].set_xscale('log', base=2); axs[1].set_xticks([8, 16, 32, 128]); axs[1].set_xticklabels(['8', '16', '32', '128'])
axs[1].set_xlabel('tables per head (tph)'); axs[1].set_ylabel('mean FVU over L0-L5')
axs[1].set_title('Mean FVU vs tph'); axs[1].legend(); axs[1].grid(ls=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(R, 'menu_tph_sweep_fvu.png'), dpi=130)

if usage:
    ur = [r for r in ('menu4k_H8_tph128', 'menu4k_H8_tph8') if r in usage]
    fig, axs = plt.subplots(len(ur), 6, figsize=(18, 3.2 * len(ur)), squeeze=False)
    for i, run in enumerate(ur):
        for li in range(6):
            hist = torch.tensor(usage[run][li]['hist_tokens'])        # [H, M]
            hist = hist / hist.sum(1, keepdim=True)
            srt = hist.sort(dim=1, descending=True).values
            ax = axs[i][li]
            for h in range(hist.shape[0]):
                ax.plot(range(1, 65), srt[h].numpy(), lw=1, alpha=0.7)
            et = sum(x['eff_items'] for x in usage[run][li]['token']) / 8
            ax.set_title(f'{summary[run]["label"]} L{li}: eff {et:.1f}/64', fontsize=9)
            ax.set_yscale('log'); ax.set_ylim(1e-5, 1)
            if li == 0:
                ax.set_ylabel('token-weighted share (sorted)')
            ax.set_xlabel('menu item rank')
    fig.suptitle('Menu usage per head (8 lines per panel): token-weighted hard menu share on the held-out slab, sorted')
    fig.tight_layout()
    fig.savefig(os.path.join(R, 'menu_tph_sweep_usage.png'), dpi=120)
print('wrote runs/menu_tph_sweep_fvu.png, runs/menu_tph_sweep_usage.png, runs/menu_tph_sweep_summary.json')
