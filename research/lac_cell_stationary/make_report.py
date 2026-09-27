"""Turn the artifacts into the markdown result tables and splice them into README.md
below the <!-- RESULTS --> marker. Regenerating is idempotent."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, 'artifacts')
FAMS = ['gather', 'v0', 'v1', 'v2', 'v3']
SHAPES = {'B': 'paper reference (NT=256, R=256, N=1024, 64 MiB int8, synthetic uniform '
               'indices)',
          'A': 'repo real (one head of blocks.0.ffn.lut_light, NT=256, R=256, N=48, '
               '3 MiB int8, real trained indices and margin coefficients)'}


def load(name):
    p = os.path.join(ART, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ladder_table(rows, shape, use_coef):
    sel = [r for r in rows if r['shape'] == shape and r['use_coef'] == use_coef]
    if not sel:
        return ''
    bs = sorted({r['B'] for r in sel})
    fams = [f for f in FAMS if any(r['family'] == f for r in sel)]
    out = ['| B | ' + ' | '.join(f'`{f}` ms' for f in fams) +
           ' | best cell-stationary vs baseline |',
           '|---:|' + '---:|' * (len(fams) + 1)]
    for B in bs:
        cells, cs_best, base = [], None, None
        for f in fams:
            m = [r for r in sel if r['B'] == B and r['family'] == f]
            if not m:
                cells.append('-')
                continue
            r = min(m, key=lambda x: x['median_ms'])
            cells.append(f'{r["median_ms"]:.4f}' + ('*' if r.get('capped') else ''))
            if f == 'gather':
                base = r['median_ms']
            elif cs_best is None or r['median_ms'] < cs_best:
                cs_best = r['median_ms']
        ratio = f'{cs_best/base:.1f}x slower' if (base and cs_best) else '-'
        out.append(f'| {B} | ' + ' | '.join(cells) + f' | {ratio} |')
    out.append('')
    out.append('Configurations used (the per-family winner of the large-batch sweep):')
    out.append('')
    for f in fams:
        m = [r for r in sel if r['family'] == f]
        if m:
            lab = min(m, key=lambda x: (x['B'] != max(bs), x['median_ms']))['label']
            out.append(f'- `{f}`: {lab}')
    out.append('')
    out.append('`*` = one launch only; the configuration exceeded the per-config time cap.')
    return '\n'.join(out)


def tokens_table(rows, shape, use_coef):
    sel = [r for r in rows if r['shape'] == shape and r['use_coef'] == use_coef]
    if not sel:
        return ''
    bs = sorted({r['B'] for r in sel})
    fams = [f for f in FAMS if any(r['family'] == f for r in sel)]
    out = ['| B | ' + ' | '.join(f'`{f}` tok/s' for f in fams) + ' |',
           '|---:|' + '---:|' * len(fams)]
    for B in bs:
        cells = []
        for f in fams:
            m = [r for r in sel if r['B'] == B and r['family'] == f]
            cells.append(f'{max(x["tokens_per_s"] for x in m):.3e}' if m else '-')
        out.append(f'| {B} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main():
    b = load('bench.json')
    st = load('index_stats.json')
    l2 = load('l2.json')
    pw = load('power.json')
    real = None
    parts = ['## Results', '']

    if b:
        parts += [f'Device: {b["device"]}, torch {b["torch"]}. '
                  f'Clocks after the run: {b.get("clocks_after", "n/a")}.', '']
        for shape in ('B', 'A'):
            for use_coef in (True, False):
                t = ladder_table(b['ladder'], shape, use_coef)
                if not t:
                    continue
                parts += [f'### Latency ladder -- shape {shape}: {SHAPES[shape]}',
                          '',
                          f'Coefficient mode: **{"the trained margin score" if use_coef else "c = 1"}**.',
                          '', t, '',
                          'Tokens per second, same runs:', '',
                          tokens_table(b['ladder'], shape, use_coef), '']

    if st:
        parts += ['### Index statistics of the real layer', '',
                  f'`{st["run"]}` / `{st["layer"]}`, B={st["B"]} real tokens, '
                  f'{st["n_heads"]} heads x {st["tables_per_head"]} tables x {st["R"]} rows.',
                  '',
                  '| head | participation ratio (of 256) | entropy (nats, max 5.545) | '
                  'dead rows | top-1 row share | rows 0+255 share |',
                  '|---:|---:|---:|---:|---:|---:|']
        for h in st['heads']:
            parts.append(f'| {h["head"]} | {h["pr_mean"]:.1f} '
                         f'(min {h["pr_min"]:.1f}, max {h["pr_max"]:.1f}) | '
                         f'{h["entropy_mean"]:.3f} | {h["dead_frac"]:.4f} | '
                         f'{h["top1_share"]:.4f} | {h["edge_share"]:.4f} |')
        parts += ['',
                  '**Intra-thread dedup vs the birthday bound** -- a thread owning row r '
                  'across G tables:', '',
                  '| G | measured dedup | birthday bound | measured / bound | '
                  'atomic groups per token | reduction vs G=1 |',
                  '|---:|---:|---:|---:|---:|---:|']
        for d in st['dedup']:
            parts.append(f'| {d["G"]} | {d["empirical_dedup"]:.4f} | '
                         f'{d["birthday_dedup"]:.4f} | {d["ratio"]:.3f} | '
                         f'{d["atomic_groups_per_token"]:.1f} | '
                         f'{d["reduction_vs_G1"]:.3f} |')
        a = st['alignment']
        parts += ['',
                  f'Cross-table hot-row alignment: the per-table argmax row takes '
                  f'{a["argmax_distinct"]} distinct values across the {st["tables_per_head"]} '
                  f'tables, the most common (row {a["argmax_mode_row"]}) shared by only '
                  f'{a["argmax_mode_count"]} of them; {a["argmax_row0"]} tables peak at row 0 '
                  f'and {a["argmax_rowlast"]} at row 255, so there is no saturating-comparison '
                  f'signature.', '',
                  '**Distinct rows touched per table per token tile** (the amortisation '
                  'headroom for a bucketed / accumulator-stationary kernel):', '',
                  '| tile | distinct rows (of 256) | as a fraction of the tile | '
                  'read reduction vs one row per token |',
                  '|---:|---:|---:|---:|']
        for t in st['tiles']:
            parts.append(f'| {t["tile"]} | {t["distinct_mean"]:.1f} | '
                         f'{t["frac_of_tile"]:.3f} | {t["read_reduction_vs_TS"]:.2f}x |')
        parts.append('')

    if l2:
        parts += ['### L2 amortisation of the baseline, measured without ncu', '',
                  '| N | table footprint (MiB) | per-token read (KiB) | median ms | '
                  'requested GB/s | x HBM peak (1,792 GB/s) | fits in L2 |',
                  '|---:|---:|---:|---:|---:|---:|:--|']
        for o in l2:
            parts.append(f'| {o["N"]} | {o["footprint_bytes"]/2**20:.0f} | '
                         f'{o["per_token_bytes"]/1024:.0f} | {o["median_ms"]:.4f} | '
                         f'{o["request_gbs"]:.0f} | {o["x_hbm_peak"]:.2f} | '
                         f'{"yes" if o["l2_resident"] else "no"} |')
        parts.append('')

    if pw:
        parts += ['### Energy per LUT-Core evaluation', '',
                  f'Whole-board power while the kernel loops; idle draw '
                  f'{pw["idle_w"]:.1f} W.', '',
                  '| kernel | per launch (ms) | board W | uJ per evaluation | '
                  'M evaluations/s |', '|:--|---:|---:|---:|---:|']
        for r in pw['rows']:
            parts.append(f'| `{r["label"]}` | {r["ms"]:.3f} | {r["watts"]:.0f} | '
                         f'{r["uj_per_eval"]:.2f} | {r["mevals_per_s"]:.2f} |')
        parts.append('')

    txt = '\n'.join(parts)
    rp = os.path.join(HERE, 'README.md')
    src = open(rp).read()
    head = src.split('<!-- RESULTS -->')[0]
    open(rp, 'w').write(head + '<!-- RESULTS -->\n\n' + txt)
    print(f'spliced {len(txt)} chars of results into README.md')


if __name__ == '__main__':
    main()
