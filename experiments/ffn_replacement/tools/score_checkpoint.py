"""Final-score a trained checkpoint on the fixed eval protocol (bs48 x 100, skip 12).

Uses the SAME `evaluate_bpb_fixed` the trainer logs its curve with, so a run's final score
and its training-time eval are the identical measurement. Rebuilds the model from the run's
own config.json via the shared `model_build`.

    python score_checkpoint.py --run ../runs/exp_n_0151_long48k_untied_vanilla
    python score_checkpoint.py --run <dir> --checkpoint /path/to/checkpoint.pt   # ckpt elsewhere
    python score_checkpoint.py --run <dir> --skip-rows 0     # reproduce the plain bs48x100 number

Checkpoints are gitignored, so --checkpoint lets you point at one that lives outside the
run folder (e.g. on the training host).
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # tools/ on path for sibling imports
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes

from model_build import build_model
from fixed_eval import evaluate_bpb_fixed, eval_config, EVAL_BATCH_SIZE, EVAL_STEPS, EVAL_SKIP_ROWS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir holding config.json (+ checkpoint.pt)')
    ap.add_argument('--checkpoint', default=None, help='checkpoint path (default: <run>/checkpoint.pt)')
    ap.add_argument('--eval-batch-size', type=int, default=None)
    ap.add_argument('--eval-steps', type=int, default=None)
    ap.add_argument('--skip-rows', type=int, default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, 'config.json')))
    ck = a.checkpoint or os.path.join(a.run, 'checkpoint.pt')
    assert os.path.exists(ck), f'checkpoint not found: {ck}'

    ec = eval_config(cfg)
    if a.eval_batch_size is not None: ec['eval_batch_size'] = a.eval_batch_size
    if a.eval_steps is not None:      ec['eval_steps'] = a.eval_steps
    if a.skip_rows is not None:       ec['skip_rows'] = a.skip_rows

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()
    token_bytes = get_token_bytes(device=a.device)

    model = build_model(cfg, vocab, device=a.device)
    sd = torch.load(ck, map_location=a.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()

    bpb = evaluate_bpb_fixed(model, tok, token_bytes, cfg['seq_len'], a.device, **ec)
    out = {
        'run': cfg.get('exp_name', os.path.basename(a.run.rstrip('/'))),
        'ffn_type': cfg.get('ffn_type'),
        'device_batch_size_train': cfg.get('device_batch_size'),
        'eval_batch_size': ec['eval_batch_size'], 'eval_steps': ec['eval_steps'],
        'skip_rows': ec['skip_rows'],
        'val_tokens_scored': (ec['eval_batch_size'] * ec['eval_steps'] - ec['skip_rows']) * cfg['seq_len'],
        'bpb_fixed': bpb,
        'load_missing_keys': len(missing), 'load_unexpected_keys': len(unexpected),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
