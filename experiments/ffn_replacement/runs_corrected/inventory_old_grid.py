"""Enumerate every experiment in ../runs/ and classify it for corrected re-scoring.

Reads each run's config.json and summary.json, records the shape and the originally reported
numbers, and sorts every directory into:

  (a) already re-scored — a corrected_score.json exists in runs_corrected/
  (b) in scope         — a real trained run whose checkpoint we need
  (c) not scoreable    — no summary.json / no trained result, a smoke or aborted run,
                         or a config we cannot rebuild

Writes old_grid_inventory.json next to this file. No network, no heavy work.

    python inventory_old_grid.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
RUNS = os.path.join(FR, 'runs')
sys.path.insert(0, os.path.join(FR, 'tools'))

ALREADY = {  # re-scored earlier; leave alone
    'exp_n_0118_ffnsw_S2a_nap9_FULL16k', 'exp_n_0133_grid_H4d48_nap10_tph128',
    'exp_n_0129_grid_H4d48_nap8_tph256', 'exp_n_0135_untied_vanilla_baseline_16k',
    'exp_n_0136_fastmhl_raw_H4_nap8_tph128', 'exp_n_0160_fastmhl_raw_H1_nap9_tph512',
    'exp_n_0161_fastmhl_raw_H1_nap10_tph256',
}


def shape_of(cfg):
    if cfg.get('ffn_type') == 'dense':
        return dict(kind='dense', H=None, tph=None, cells=None, d_in=None, d_out=None)
    if cfg.get('ffn_lut_kind') == 'fastmhl_raw':
        return dict(kind='fastmhl_raw', H=cfg.get('raw_n_heads'), tph=cfg.get('raw_tph'),
                    cells=2 ** cfg['raw_nap'] if 'raw_nap' in cfg else None,
                    d_in=None, d_out=None)
    nap = cfg.get('lut_n_anchor_pairs')
    return dict(kind='compression', H=cfg.get('lut_n_heads'),
                tph=cfg.get('lut_tables_per_head'),
                cells=(2 ** nap) if nap is not None else None,
                d_in=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
                d_out=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')))


def main():
    rows = []
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d):
            continue
        cfgp, sump = os.path.join(d, 'config.json'), os.path.join(d, 'summary.json')
        r = dict(run=name, has_config=os.path.exists(cfgp), has_summary=os.path.exists(sump),
                 has_metrics=os.path.exists(os.path.join(d, 'metrics.csv')),
                 local_checkpoint=os.path.exists(os.path.join(d, 'checkpoint.pt')))
        if r['has_config']:
            cfg = json.load(open(cfgp))
            r.update(shape_of(cfg))
            for k in ('ffn_type', 'n_steps', 'device_batch_size', 'total_batch_size',
                      'eval_steps', 'seq_len', 'tie_unembedder', 'lr'):
                r[k] = cfg.get(k)
            r['old_eval_tokens'] = (cfg.get('device_batch_size', 0) * cfg.get('seq_len', 0)
                                    * cfg.get('eval_steps', 0)) or None
        if r['has_summary']:
            s = json.load(open(sump))
            r['final_val_bpb'] = s.get('final_val_bpb')
            r['best_val_bpb'] = s.get('best_val_bpb')
            r['total_params'] = s.get('total_params')
            r['training_time_hours'] = s.get('training_time_hours')
        # classify
        if name in ALREADY or os.path.exists(
                os.path.join(HERE, name, 'corrected_score.json')):
            r['bucket'] = 'a_already_rescored'
        elif not r['has_config']:
            r['bucket'] = 'c_unscoreable'
            r['why'] = 'no config.json — cannot rebuild the model'
        elif not r['has_summary'] or r.get('final_val_bpb') is None:
            r['bucket'] = 'c_unscoreable'
            r['why'] = 'no summary.json / no final_val_bpb — never completed a trained result'
        elif r.get('total_params') is None:
            r['bucket'] = 'c_unscoreable'
            r['why'] = 'summary.json has no total_params to verify a checkpoint against'
        else:
            r['bucket'] = 'b_in_scope'
        rows.append(r)

    buckets = {}
    for r in rows:
        buckets.setdefault(r['bucket'], []).append(r)
    for b in sorted(buckets):
        print(f"\n=== {b}  ({len(buckets[b])})")
        for r in buckets[b]:
            if b == 'c_unscoreable':
                print(f"   {r['run']:<52} {r['why']}")
            else:
                print(f"   {r['run']:<52} {str(r.get('kind')):<13} "
                      f"H{r.get('H')} tph{r.get('tph')} c{r.get('cells')} "
                      f"in{r.get('d_in')} out{r.get('d_out')}  "
                      f"{(r.get('total_params') or 0):>12,}  "
                      f"steps {r.get('n_steps')}  dbs {r.get('device_batch_size')}  "
                      f"final {r.get('final_val_bpb')}")
    with open(os.path.join(HERE, 'old_grid_inventory.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\ntotals: " + "  ".join(f"{b}={len(v)}" for b, v in sorted(buckets.items())))
    print(f"wrote {HERE}/old_grid_inventory.json")


if __name__ == '__main__':
    main()
