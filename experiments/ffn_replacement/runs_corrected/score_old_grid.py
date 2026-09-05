"""Re-score one historical ffn_replacement run under the corrected protocol.

Same code path as the earlier re-scorings (tools/fixed_eval.evaluate_bpb_fixed via
tools/model_build.build_model) so the numbers are directly comparable to exp_n_0118 / 0129 /
0133 and to nebius's exp_n_0135 / 0136 / 0160 / 0161. Nothing is re-implemented here; this
only wraps the result in the corrected_score.json format the folder uses and adds the
originally-reported numbers for the delta.

Refuses to publish a number if the checkpoint does not load cleanly (0 missing / 0 unexpected
keys) or if the rebuilt parameter count disagrees with the run's own summary.json.

    python score_old_grid.py <run_name> [--checkpoint PATH] [--keep]

`--checkpoint` defaults to runs_corrected/<run>/checkpoint.pt. Without `--keep` the checkpoint
is DELETED after scoring when it lives inside runs_corrected/ (disk is limited; the score file
is what we keep). A checkpoint outside runs_corrected/ is never deleted.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))

from nanochat.common import get_base_dir                          # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes  # noqa: E402
from model_build import build_model                               # noqa: E402
from fixed_eval import evaluate_bpb_fixed, eval_config            # noqa: E402

VANILLA_FFN_MACS = 2 * 384 * 1536
ANCHOR = 1.1651468950008814          # exp_n_0135, corrected
NAIVE = 1.1929264025964097           # exp_n_0136, corrected


def shape_of(cfg):
    if cfg.get('ffn_type') == 'dense':
        return dict(kind='dense', H=None, tph=None, cells=None, d_in=None, d_out=None)
    if cfg.get('ffn_lut_kind') == 'fastmhl_raw':
        return dict(kind='fastmhl_raw', H=cfg.get('raw_n_heads'), tph=cfg.get('raw_tph'),
                    cells=2 ** cfg['raw_nap'], d_in=None, d_out=None)
    nap = cfg.get('lut_n_anchor_pairs')
    return dict(kind='compression', H=cfg.get('lut_n_heads'),
                tph=cfg.get('lut_tables_per_head'), cells=2 ** nap,
                d_in=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
                d_out=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()

    d = os.path.join(HERE, a.run)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    summ = json.load(open(os.path.join(d, 'summary.json')))
    ck = a.checkpoint or os.path.join(d, 'checkpoint.pt')
    if not os.path.exists(ck):
        print(f'SKIP {a.run}: no checkpoint at {ck}')
        sys.exit(2)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=dev)
    ec = eval_config(cfg)
    model = build_model(cfg, tok.get_vocab_size(), device=dev)
    total_params = sum(p.numel() for p in model.parameters())

    recorded = summ.get('total_params')
    if recorded is not None and total_params != recorded:
        print(f'STOP {a.run}: rebuilt params {total_params:,} != summary.json '
              f'{recorded:,} — not publishing a number for a model we cannot reproduce')
        sys.exit(3)
    missing, unexpected = model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    if missing or unexpected:
        print(f'STOP {a.run}: checkpoint load is not clean — {len(missing)} missing, '
              f'{len(unexpected)} unexpected keys')
        print(f'   missing[:5]={list(missing)[:5]}  unexpected[:5]={list(unexpected)[:5]}')
        sys.exit(4)
    model.eval()
    bpb = evaluate_bpb_fixed(model, tok, token_bytes, cfg['seq_len'], dev, **ec)
    del model
    torch.cuda.empty_cache()

    sh = shape_of(cfg)
    cf = (sh['H'] * 384 * sh['d_in']) if sh['d_in'] else None
    df = (sh['H'] * 384 * sh['d_out']) if sh['d_out'] else None
    orig = summ.get('final_val_bpb')
    out = {
        'run': cfg.get('exp_name', a.run),
        'description': (
            f"Historical ffn_replacement run re-scored under the FIXED validation protocol "
            f"(see ../../FIXED_EVAL.md): batch size 48 x 100 eval steps, leading 12 rows "
            f"skipped, batch-size-INDEPENDENT, on the held-out val shard "
            f"(shard_06542.parquet) from token 0. Rebuilt from config.json "
            f"(ffn_type={cfg.get('ffn_type')}) and loaded from checkpoint.pt with 0 missing / "
            f"0 unexpected keys; rebuilt parameter count matches summary.json exactly. The "
            f"original number was produced by the batch-coupled eval at device_batch_size "
            f"{cfg.get('device_batch_size')} x eval_steps {cfg.get('eval_steps')} = "
            f"{(cfg.get('device_batch_size') or 0) * cfg['seq_len'] * (cfg.get('eval_steps') or 0):,} "
            f"val tokens."),
        'eval_protocol': {
            'eval_batch_size': ec['eval_batch_size'], 'eval_steps': ec['eval_steps'],
            'skip_rows': ec['skip_rows'],
            'val_tokens_scored': (ec['eval_batch_size'] * ec['eval_steps'] - ec['skip_rows'])
                                 * cfg['seq_len'],
            'batch_size_independent': True, 'val_shard': 'shard_06542.parquet',
        },
        'shape': sh,
        'total_params': total_params,
        'n_steps': cfg.get('n_steps'),
        'training_device_batch_size': cfg.get('device_batch_size'),
        'original_eval_window': {
            'device_batch_size': cfg.get('device_batch_size'),
            'eval_steps': cfg.get('eval_steps'),
            'val_tokens_scored': (cfg.get('device_batch_size') or 0) * cfg['seq_len']
                                 * (cfg.get('eval_steps') or 0),
            'batch_coupled': True,
        },
        'compress_projection_flops': cf,
        'decompress_projection_flops': df,
        'projection_flops_total': (cf + df) if (cf and df) else None,
        'projection_flops_ratio_vs_vanilla_ffn': ((cf + df) / VANILLA_FFN_MACS)
                                                 if (cf and df) else None,
        'corrected_val_bpb': bpb,
        'originally_reported_bpb': orig,
        'originally_reported_best_bpb': summ.get('best_val_bpb'),
        'correction': (bpb - orig) if orig else None,
        'matched_vanilla_16k_baseline': {
            'run': 'exp_n_0135_untied_vanilla_baseline_16k',
            'corrected_val_bpb': ANCHOR, 'originally_reported_bpb': 1.20144},
        'naive_lut_reference': {'run': 'exp_n_0136_fastmhl_raw_H4_nap8_tph128',
                                'corrected_val_bpb': NAIVE},
        'delta_vs_vanilla_corrected': bpb - ANCHOR,
        'delta_vs_naive_lut_corrected': bpb - NAIVE,
        'training_time_hours': summ.get('training_time_hours'),
        'load_missing_keys': 0, 'load_unexpected_keys': 0,
    }
    with open(os.path.join(d, 'corrected_score.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ('run', 'total_params', 'corrected_val_bpb', 'originally_reported_bpb',
                       'correction', 'delta_vs_vanilla_corrected')}, indent=2))

    inside = os.path.abspath(ck).startswith(os.path.abspath(HERE) + os.sep)
    if inside and not a.keep:
        os.remove(ck)
        print(f'removed {ck} (score kept; disk is limited)')


if __name__ == '__main__':
    main()
