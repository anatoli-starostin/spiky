"""Table and analysis over the FULL-LENGTH corrected runs (the historical grid + anchors).

Distinct from summarise_all.py, which covers the 4k-step proxy sweeps. These are the real
16k / 48k / 144k runs re-scored under the corrected protocol, so they ARE mutually comparable
and they are the cross-check on what the proxy sweeps concluded.

Reads every corrected_score.json in runs_corrected/ that is not a proxy-sweep run.

    python summarise_grid.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = 1.1651468950008814          # exp_n_0135 corrected — the matched 16k vanilla
NAIVE = 1.1929264025964097           # exp_n_0136 corrected


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, '*', 'corrected_score.json'))):
        s = json.load(open(p))
        if s.get('proxy_sweep'):
            continue
        d = os.path.dirname(p)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        sp = os.path.join(d, 'summary.json')
        summ = json.load(open(sp)) if os.path.exists(sp) else {}
        nap = cfg.get('lut_n_anchor_pairs')
        raw = cfg.get('ffn_lut_kind') == 'fastmhl_raw'
        rows.append(dict(
            run=os.path.basename(d),
            short=os.path.basename(d).split('_')[2] if os.path.basename(d).startswith('exp_n_')
                  else os.path.basename(d),
            tag=os.path.basename(d)[6:10],
            kind=('dense' if cfg.get('ffn_type') == 'dense' else
                  'raw' if raw else 'compression'),
            H=cfg.get('raw_n_heads') if raw else cfg.get('lut_n_heads'),
            tph=cfg.get('raw_tph') if raw else cfg.get('lut_tables_per_head'),
            cells=(2 ** cfg['raw_nap']) if raw else ((2 ** nap) if nap is not None else None),
            d_in=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
            d_out=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')),
            params=s.get('total_params') or summ.get('total_params') or 0,
            n_steps=cfg.get('n_steps'),
            dbs=cfg.get('device_batch_size'),
            old_tokens=(cfg.get('device_batch_size') or 0) * cfg['seq_len']
                       * (cfg.get('eval_steps') or 0),
            orig=s.get('originally_reported_bpb') or summ.get('final_val_bpb'),
            best=s.get('originally_reported_best_bpb') or summ.get('best_val_bpb'),
            corr=s['corrected_val_bpb']))
    for r in rows:
        r['delta'] = (r['corr'] - r['orig']) if r['orig'] else None
        r['vs_van'] = r['corr'] - ANCHOR
    return rows


def main():
    rows = sorted(load(), key=lambda r: r['corr'])
    print(f"CORRECTED FULL-LENGTH RUNS ({len(rows)}) — evaluate_bpb_fixed, bs48 x 100, skip 12, "
          f"2,451,456 val tokens.\nThese ARE mutually comparable. Vanilla 16k anchor "
          f"exp_n_0135 = {ANCHOR:.6f}, naive LUT exp_n_0136 = {NAIVE:.6f}.\n")
    print(f"{'run':<10}{'shape':<28}{'params':>12}{'steps':>7}{'dbs':>5}{'orig bpb':>11}"
          f"{'corrected':>11}{'correction':>12}{'vs vanilla':>12}")
    for r in rows:
        shape = ('dense 4x MLP' if r['kind'] == 'dense' else
                 f"raw H{r['H']} tph{r['tph']} c{r['cells']}" if r['kind'] == 'raw' else
                 f"H{r['H']} tph{r['tph']} c{r['cells']} d{r['d_in']}/{r['d_out']}")
        print(f"{r['tag']:<10}{shape:<28}{r['params']:>12,}{(r['n_steps'] or 0):>7}"
              f"{(r['dbs'] or 0):>5}{(r['orig'] or 0):>11.6f}{r['corr']:>11.6f}"
              f"{(r['delta'] or 0):>+12.6f}{r['vs_van']:>+12.6f}")

    print("\nCORRECTION SIZE vs ORIGINAL device_batch_size "
          "(the batch-coupled window: dbs x 512 x eval_steps tokens)")
    by = {}
    for r in rows:
        if r['delta'] is not None:
            by.setdefault(r['dbs'], []).append(r)
    for dbs in sorted(by):
        ds = [r['delta'] for r in by[dbs]]
        toks = by[dbs][0]['old_tokens']
        print(f"   dbs {dbs:>2}  n={len(ds):<2}  old window {toks:>7,} tok  "
              f"correction mean {sum(ds)/len(ds):+.6f}  "
              f"range [{min(ds):+.6f}, {max(ds):+.6f}]")
        print(f"            runs: " + ", ".join(r['tag'] for r in by[dbs]))

    print("\nRANK CHECK — original order vs corrected order (16k runs only, "
          "the only ones on a common training budget)")
    k16 = [r for r in rows if r['n_steps'] == 16000 and r['orig']]
    o = sorted(k16, key=lambda r: r['orig'])
    c = sorted(k16, key=lambda r: r['corr'])
    print("   by ORIGINAL:  " + " < ".join(r['tag'] for r in o))
    print("   by CORRECTED: " + " < ".join(r['tag'] for r in c))
    print(f"   ordering preserved: {[r['tag'] for r in o] == [r['tag'] for r in c]}")
    print("\n   sign vs the vanilla anchor, before and after:")
    for r in sorted(k16, key=lambda r: r['corr']):
        if r['kind'] == 'dense':
            continue
        was = r['orig'] - 1.20144
        now = r['vs_van']
        flip = '  <- FLIPS' if (was < 0) != (now < 0) else ''
        print(f"     {r['tag']}  was {was:+.5f} vs 1.20144   now {now:+.5f} vs "
              f"{ANCHOR:.5f}{flip}")


if __name__ == '__main__':
    main()
