"""Score one proxy-sweep run on the corrected protocol and write its corrected_score.json.

Same measurement as `tools/score_checkpoint.py` — it calls the identical
`evaluate_bpb_fixed` (bs48 x 100, skip 12, 2,451,456 val tokens) and rebuilds the model from
the run's own config.json — but wraps the result in the corrected_score.json format the rest
of runs_corrected/ uses, with the sweep's comparability warning and the delta against S0
(this sweep's own vanilla zero-line) filled in.

    python score_sweep.py sweep_s01_tied_H4_tph256_c256_din32_dout32
"""
import glob
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

S0 = 'sweep_s00_vanilla_dense'
WARNING = (
    'PROXY BUDGET — NOT COMPARABLE TO THE 16k ANCHORS. This run is 4,000 steps at an '
    'effective batch of 24 sequences (12,288 tokens), one of 11 runs sharing that budget so '
    'they can be compared to each other cheaply. The EVAL is the full corrected protocol '
    '(evaluate_bpb_fixed: bs48 x 100 batches, leading 12 rows skipped, 2,451,456 val tokens '
    'of the held-out shard_06542.parquet, batch-size independent), identical for all 11 — '
    'only its frequency was reduced to every 500 steps. Because the TRAINING budget is 1/8 '
    'of the long runs, these bpb values sit far above them and must NEVER be compared with '
    'exp_n_0135 (1.165147), exp_n_0136 (1.192926), exp_n_0118 (1.164939) or exp_n_0129 '
    '(1.170961). The sweep\'s own zero-line is sweep_s00_vanilla_dense.')


def main():
    run = sys.argv[1]
    d = os.path.join(HERE, run)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    ck = os.path.join(d, 'checkpoint.pt')
    assert os.path.exists(ck), f'no checkpoint: {ck}'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=dev)
    ec = eval_config(cfg)
    model = build_model(cfg, tok.get_vocab_size(), device=dev)
    missing, unexpected = model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    model.eval()
    bpb = evaluate_bpb_fixed(model, tok, token_bytes, cfg['seq_len'], dev, **ec)
    total_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()

    # the run lives in one of the sweep manifests (sweep_manifest.json, sweep2_, sweep3_, ...)
    me = man = None
    for p in sorted(glob.glob(os.path.join(HERE, 'sweep*_manifest.json'))):
        m = json.load(open(p))
        hit = next((r for r in m['runs'] if r['run'] == run), None)
        if hit is not None:
            me, man = hit, m
            break
    assert me is not None, f'{run} is in neither sweep manifest'
    summ = json.load(open(os.path.join(d, 'summary.json')))

    s0_path = os.path.join(HERE, S0, 'corrected_score.json')
    s0 = json.load(open(s0_path)) if os.path.exists(s0_path) else None

    out = {
        'run': cfg['exp_name'],
        'description': cfg['_arch_note'],
        'proxy_sweep': True,
        'comparability_warning': WARNING,
        'training_budget': {
            'n_steps': cfg['n_steps'],
            'effective_batch_sequences': man['effective_batch_sequences'],
            'effective_batch_tokens': cfg['total_batch_size'],
            'device_batch_size': cfg['device_batch_size'],
            'grad_accum': cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len']),
            'eval_every': cfg['eval_every'],
        },
        'eval_protocol': {
            'eval_batch_size': ec['eval_batch_size'], 'eval_steps': ec['eval_steps'],
            'skip_rows': ec['skip_rows'],
            'val_tokens_scored': (ec['eval_batch_size'] * ec['eval_steps'] - ec['skip_rows'])
                                 * cfg['seq_len'],
            'batch_size_independent': True, 'val_shard': 'shard_06542.parquet',
        },
        'shape': {k: me[k] for k in ('H', 'tph', 'cells', 'd_in', 'd_out')},
        'compress_projection_flops': me['compress_flops'],
        'compress_projection_flops_ratio_vs_vanilla': me['compress_flops_ratio'],
        # sweep 2 records the decompress side too; derive it for sweep-1 runs so the FLOPs
        # accounting is complete and symmetric across all 17
        'decompress_projection_flops': me.get(
            'decompress_flops',
            (me['H'] * 384 * me['d_out']) if me['H'] else None),
        'projection_flops_total': me.get(
            'projection_flops_total',
            (me['H'] * 384 * (me['d_in'] + me['d_out'])) if me['H'] else None),
        'projection_flops_ratio_vs_vanilla_ffn': me.get(
            'projection_flops_ratio_vs_vanilla_ffn',
            (me['H'] * 384 * (me['d_in'] + me['d_out']) / (2 * 384 * 1536))
            if me['H'] else 1.0),
        'total_params': total_params,
        'expected_params': me['expected'],
        'param_deviation_vs_brief': me['deviation'],
        'proxy_val_bpb': bpb,
        'training_time_hours': summ.get('training_time_hours'),
        'best_val_bpb_during_training': summ.get('best_val_bpb'),
        'load_missing_keys': len(missing), 'load_unexpected_keys': len(unexpected),
    }
    if s0 is not None:
        out['sweep_vanilla_anchor'] = {'run': S0, 'proxy_val_bpb': s0['proxy_val_bpb']}
        out['delta_vs_sweep_vanilla'] = bpb - s0['proxy_val_bpb']
    with open(os.path.join(d, 'corrected_score.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ('run', 'total_params', 'proxy_val_bpb', 'delta_vs_sweep_vanilla',
                       'load_missing_keys', 'load_unexpected_keys') if k in out}, indent=2))


if __name__ == '__main__':
    main()
