"""Tables from probe_attention_sinks.py's measurements. Reads the JSON, prints, renders one figure.

THE CRITERION, stated before looking at the numbers so it cannot be fitted to them. A key position counts
as a sink when all three hold:
  mass      its mean attention mass is >= 5x uniform (uniform = 1/196 = 0.0051 for self-attention)
  ranking   >= 50% of queries put it top-1
  invariance its coefficient of variation across queries is <= 0.5
Mass alone is an informative token; mass plus query-invariance is a sink.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs_vit')
GRID = 14
MASS_X, TOP1_MIN, CV_MAX = 5.0, 0.5, 0.5


def rowcol(t):
    return divmod(t, GRID)


def border(t):
    r, c = rowcol(t)
    return r in (0, GRID - 1) or c in (0, GRID - 1)


def head_rows(layer, st):
    mean, var = np.array(st['mean']), np.array(st['var'])
    top1, vn = np.array(st['top1_frac']), np.array(st['value_norm'])
    hn = np.array(st['hidden_norm']) if 'hidden_norm' in st else None
    n_keys = mean.shape[1]
    out = []
    for h in range(mean.shape[0]):
        k = int(mean[h].argmax())
        cv = float(np.sqrt(max(var[h, k], 0)) / max(mean[h, k], 1e-12))
        out.append(dict(
            layer=layer, head=h, key=k, n_keys=n_keys,
            ratio=float(mean[h, k] * n_keys), top1=float(top1[h, k]), cv=cv,
            entropy=float(st['entropy'][h]), maxw=float(st['max_w'][h]),
            vnorm_rel=float(vn[k] / vn.mean()),
            hnorm_rel=float(hn[k] / hn.mean()) if hn is not None else float('nan'),
            top5=[(int(t), float(mean[h, t] * n_keys)) for t in mean[h].argsort()[::-1][:5]],
            sink=bool(mean[h, k] * n_keys >= MASS_X and top1[h, k] >= TOP1_MIN and cv <= CV_MAX)))
    return out


def main():
    rep = json.load(open(os.path.join(R, 'attention_sinks.json')))
    summary = {}
    for label, run in rep.items():
        heads = run.pop('_heads')
        print(f'\n{"=" * 118}\n{label}   ({heads} heads/layer)\n{"=" * 118}')
        print(f'{"layer":<11}{"h":>2} {"top key":>8} {"(r,c)":>8} {"brdr":>5} {"mass/unif":>10} '
              f'{"top1 frac":>10} {"CV":>7} {"entropy":>8} {"mean top1 w":>12} {"|v|/avg":>9} '
              f'{"|h|/avg":>9}  sink')
        rows = []
        for layer in [k for k in run if k.startswith(('enc[', 'dec['))]:
            rows += head_rows(layer, run[layer])
        for r in rows:
            rc = rowcol(r['key'])
            ent_max = np.log(r['n_keys'])
            print(f'{r["layer"]:<11}{r["head"]:>2} {r["key"]:>8} {str(rc):>8} '
                  f'{"yes" if border(r["key"]) else "-":>5} {r["ratio"]:>10.1f} {r["top1"]:>10.3f} '
                  f'{r["cv"]:>7.2f} {r["entropy"]:>5.2f}/{ent_max:.2f} {r["maxw"]:>12.3f} '
                  f'{r["vnorm_rel"]:>9.2f} {r["hnorm_rel"]:>9.2f}  {"SINK" if r["sink"] else ""}')
        for cl in ['enc_cross', 'dec_cross']:
            if cl not in run:
                continue
            st = run[cl]
            mean, top1 = np.array(st['mean']), np.array(st['top1_frac'])
            var, vn = np.array(st['var']), np.array(st['value_norm'])
            n_keys = mean.shape[1]
            print(f'\n{cl}: {mean.shape[0]} heads over {n_keys} keys, uniform = {1/n_keys:.4f}')
            for h in range(mean.shape[0]):
                k = int(mean[h].argmax())
                cv = float(np.sqrt(max(var[h, k], 0)) / max(mean[h, k], 1e-12))
                per = ' '.join(f'{v:.3f}' for v in mean[h]) if n_keys <= 12 else ''
                print(f'  head {h}: top key {k:>3}  mass/unif {mean[h,k]*n_keys:>5.2f}  '
                      f'top1 {top1[h,k]:.3f}  CV {cv:.2f}  entropy {st["entropy"][h]:.2f}/'
                      f'{np.log(n_keys):.2f}  |v|/avg {vn[k]/vn.mean():.2f}'
                      + (f'\n            per-key mass: {per}' if per else ''))
        sinks = [r for r in rows if r['sink']]
        summary[label] = dict(n_heads_total=len(rows), n_sinks=len(sinks),
                              worst=max((r['ratio'] for r in rows), default=0),
                              min_entropy=min((r['entropy'] for r in rows), default=0))
        print(f'\n  -> {len(sinks)} of {len(rows)} self-attention heads meet the sink criterion; '
              f'highest mass concentration {max(r["ratio"] for r in rows):.1f}x uniform')

    print(f'\n{"=" * 60}\nacross checkpoints:')
    for k, v in summary.items():
        print(f'  {k:<18} sinks {v["n_sinks"]:>2}/{v["n_heads_total"]}  max mass '
              f'{v["worst"]:>5.1f}x  min entropy {v["min_entropy"]:.2f}')

    # The figure shows the one place anything sink-like happens: the FIRST encoder layer, where three of
    # four heads drift onto corner patches as training proceeds. Rows are heads, columns checkpoints;
    # each panel is where that head's attention mass lands on the 14x14 patch grid, relative to uniform.
    rep2 = json.load(open(os.path.join(R, 'attention_sinks.json')))
    labels = [k for k in rep2 if k.startswith('d128')] + [k for k in rep2 if k.startswith('d64')]
    maps = {lab: np.array(rep2[lab]['enc[0]']['mean']) for lab in labels}
    t1 = {lab: np.array(rep2[lab]['enc[0]']['top1_frac']) for lab in labels}
    n_h = maps[labels[0]].shape[0]
    fig, ax = plt.subplots(n_h, len(labels), figsize=(1.55 * len(labels), 1.72 * n_h + 0.9))
    vmax = max(float((m * m.shape[1]).max()) for m in maps.values())
    for j, lab in enumerate(labels):
        for h in range(n_h):
            a = ax[h][j]
            a.imshow(maps[lab][h].reshape(GRID, GRID) * maps[lab].shape[1], cmap='magma',
                     vmin=0, vmax=vmax, interpolation='nearest')
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlabel(f'top-1 {t1[lab][h].max():.2f}', fontsize=7, labelpad=1.5)
            if j == 0:
                a.set_ylabel(f'head {h}', fontsize=8)
        ax[0][j].set_title(lab.replace(' @ ', '\n'), fontsize=7.5, pad=4)
    fig.suptitle('Encoder layer 0: where each head\'s attention mass lands on the 14x14 patch grid\n'
                 '(shared colour scale; below each panel, the fraction of queries ranking its peak '
                 'patch first)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (1.72 * n_h + 0.9)))
    path = os.path.join(R, 'plots', 'attention_mass_maps.png')
    fig.savefig(path, dpi=200)
    print('\nwrote', path)


if __name__ == '__main__':
    sys.exit(main())
