"""Consolidated report over the 16,000-step corrected runs only.

The 48k/144k long runs (0151, 0152, 0155, 0156, 0157, 0158) are excluded — they are a
different training budget and are reported in SWEEP_RESULTS.md's own section.

Table parameters are defined per family, since the families differ in what a table row emits:

  CompressionMHL (both projections)  rows are d_out wide      ->  6 * H * tph * cells * d_out
  raw FastMHL    (no projections)    rows are n_embd wide     ->  6 * H * tph * cells * 384
  input-projection only (d_out = -1) rows are n_embd wide     ->  6 * H * tph * cells * 384

so a run with output compression OFF is credited the full-width table budget it actually has.

    python report_16k.py            # grouped table + family summaries
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = 1.1651468950008814          # exp_n_0135 corrected — the matched 16k vanilla
N_EMBD, DEPTH = 384, 6

FAMILY = {'dense': 'a) dense / vanilla anchor',
          'raw': 'b) raw FastMHL — no projections',
          'compression': 'c) CompressionMHL — both projections',
          'inproj': 'd) input projection only — no output compression'}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'exp_n_*', 'corrected_score.json'))):
        s = json.load(open(p))
        d = os.path.dirname(p)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        if cfg.get('n_steps') != 16000:
            continue
        sp = os.path.join(d, 'summary.json')
        summ = json.load(open(sp)) if os.path.exists(sp) else {}
        raw = cfg.get('ffn_lut_kind') == 'fastmhl_raw'
        dense = cfg.get('ffn_type') == 'dense'
        nap = cfg.get('raw_nap') if raw else cfg.get('lut_n_anchor_pairs')
        H = cfg.get('raw_n_heads') if raw else cfg.get('lut_n_heads')
        tph = cfg.get('raw_tph') if raw else cfg.get('lut_tables_per_head')
        d_in = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
        d_out = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
        fam = ('dense' if dense else 'raw' if raw else
               'inproj' if d_out == -1 else 'compression')
        cells = (2 ** nap) if nap is not None else None
        row_width = None if dense else (N_EMBD if fam in ('raw', 'inproj') else d_out)
        rows.append(dict(
            tag=os.path.basename(d)[6:10], run=os.path.basename(d), fam=fam,
            H=H, tph=tph, cells=cells, d_in=d_in, d_out=d_out, row_width=row_width,
            params=s.get('total_params') or summ.get('total_params') or 0,
            tables=(DEPTH * H * tph * cells * row_width) if row_width else None,
            dbs=cfg.get('device_batch_size'),
            orig=s.get('originally_reported_bpb') or summ.get('final_val_bpb'),
            corr=s['corrected_val_bpb']))
    for r in rows:
        r['delta'] = (r['corr'] - r['orig']) if r['orig'] else None
        r['native'] = r['delta'] is not None and abs(r['delta']) < 1e-12
        r['vs_van'] = r['corr'] - ANCHOR
    return rows


def shape(r):
    if r['fam'] == 'dense':
        return 'dense 4x MLP'
    if r['fam'] == 'raw':
        return f"raw H{r['H']} tph{r['tph']} c{r['cells']} (rows 384)"
    if r['fam'] == 'inproj':
        return f"H{r['H']} tph{r['tph']} c{r['cells']} d_in{r['d_in']} / no d_out"
    return f"H{r['H']} tph{r['tph']} c{r['cells']} d{r['d_in']}/{r['d_out']}"


def main():
    rows = sorted(load(), key=lambda r: r['corr'])
    print(f"ALL 16,000-STEP RUNS SCORED UNDER THE CORRECTED PROTOCOL ({len(rows)})")
    print("evaluate_bpb_fixed: bs48 x 100, skip 12, 2,451,456 val tokens of shard_06542.")
    print(f"Vanilla anchor exp_n_0135 = {ANCHOR:.6f}. "
          "The 48k/144k long runs are excluded — separate budget, separate section.\n")
    hdr = (f"{'run':<6}{'shape':<42}{'params':>13}{'tables':>14}{'dbs':>5}"
           f"{'orig':>10}{'corrected':>11}{'correction':>12}{'vs vanilla':>12}")
    print(hdr)
    for r in rows:
        star = '*' if r['native'] else ' '
        tp = f"{r['tables']:,}" if r['tables'] else '-'
        print(f"{r['tag']:<6}{shape(r):<42}{r['params']:>13,}{tp:>14}{star}{(r['dbs'] or 0):>4}"
              f"{(r['orig'] or 0):>10.6f}{r['corr']:>11.6f}{(r['delta'] or 0):>+12.6f}"
              f"{r['vs_van']:>+12.6f}")
    print("  * = trained ON the corrected protocol, so its zero correction is by construction")

    print("\nBY FAMILY")
    for key in ('dense', 'raw', 'compression', 'inproj'):
        g = [r for r in rows if r['fam'] == key]
        if not g:
            print(f"   {FAMILY[key]:<52} (no scored run yet)")
            continue
        b = min(g, key=lambda r: r['corr'])
        print(f"   {FAMILY[key]:<52} n={len(g)}  best {b['tag']} {b['corr']:.6f}  "
              f"gap to vanilla {b['vs_van']:+.6f}")

    print("\nIS CORRECTED bpb MONOTONE IN TABLE PARAMETERS?")
    g = sorted((r for r in rows if r['tables']), key=lambda r: r['tables'])
    print(f"   {'tables':>14}  {'run':<6}{'family':<14}{'bpb':>10}")
    prev = None
    breaks = []
    for r in g:
        mark = ''
        if prev is not None and r['corr'] > prev['corr'] and r['tables'] > prev['tables']:
            mark = '   <- more tables, WORSE than the previous row'
            breaks.append((prev['tag'], r['tag']))
        print(f"   {r['tables']:>14,}  {r['tag']:<6}{r['fam']:<14}{r['corr']:>10.6f}{mark}")
        prev = r
    print(f"   strictly monotone: {not breaks}")


if __name__ == '__main__':
    main()
