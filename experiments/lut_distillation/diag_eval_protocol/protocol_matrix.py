"""Is the published eval protocol uniform across runs, and do rankings survive the slice?

Companion to eval_windows.py. Two things:

  1. PROTOCOL SCAN. train.py builds its val loader as
         tokenizing_..._bestfit(tokenizer, DEVICE_BS, SEQ_LEN, split='val', ...)
     i.e. the EVAL batch size is the TRAINING `device_batch_size`. So any run with a
     different device_batch_size evaluated on a different packing of the val stream --
     a different set of tokens, with its own first-slice offset. This scans every
     config.json and groups runs by (device_batch_size, seq_len, eval_steps).

  2. CROSS-PROTOCOL / RANKING MATRIX. Every locally available checkpoint is evaluated on
     several DISJOINT windows at BOTH bs12 and bs48, so we can see (a) whether each run's
     published number reproduces under its own protocol, (b) how big the protocol offset is
     for the same model, and (c) whether rankings between checkpoints survive a change of
     slice.

Batches are cloned before caching -- the loader yields views into one reused GPU buffer.

    python protocol_matrix.py
"""
import glob
import json
import math
import os
import statistics as st
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESEARCH = "/home/astarostin/projects/spiky-fmhl-next"     # holds the 16k grid checkpoints
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
for p in (os.path.join(REPO, 'experiments', 'ffn_replacement', 'benchmark'),
          os.path.join(REPO, 'src'), NANOCHAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M                                                    # noqa: E402
from nanochat.common import get_base_dir                             # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes     # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402
from nanochat.loss_eval import evaluate_bpb                          # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW = 10
N_WINDOWS = 4
BATCH_SIZES = (12, 48)

# (label, exp_dir, published bpb, the device_batch_size it was published under)
CKPTS = [
    ('0151 vanilla 48k', os.path.join(REPO, 'experiments/ffn_replacement/runs/'
                                            'exp_n_0151_long48k_untied_vanilla'),
     1.151444329946291, 48),
    ('0126 LUT 16k', os.path.join(RESEARCH, 'experiments/hyperplane_ffn/'
                                            'exp_n_0126_grid_H4d48_nap7_tph64'), 1.20694, 12),
    ('0127 LUT 16k', os.path.join(RESEARCH, 'experiments/hyperplane_ffn/'
                                            'exp_n_0127_grid_H4d48_nap7_tph128'), 1.19471, 12),
    ('0128 LUT 16k', os.path.join(RESEARCH, 'experiments/hyperplane_ffn/'
                                            'exp_n_0128_grid_H4d48_nap8_tph64'), 1.20228, 12),
]


class EvalAdapter(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def get_device(self):
        return self.gpt.tok_emb.weight.device

    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits = self.gpt(idx)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.reshape(-1), ignore_index=-1,
                               reduction=loss_reduction)
        return loss.view(targets.shape) if loss_reduction == 'none' else loss


def scan_protocols():
    rows = []
    for root in (os.path.join(REPO, 'experiments/ffn_replacement/runs'),
                 os.path.join(REPO, 'experiments/hyperplane_ffn')):
        for cfgp in sorted(glob.glob(os.path.join(root, '*', 'config.json'))):
            c = json.load(open(cfgp))
            rows.append(dict(run=os.path.basename(os.path.dirname(cfgp)),
                             device_batch_size=c.get('device_batch_size'),
                             seq_len=c.get('seq_len'), eval_steps=c.get('eval_steps'),
                             eval_every=c.get('eval_every'), ffn_type=c.get('ffn_type')))
    groups = {}
    for r in rows:
        groups.setdefault(f"bs{r['device_batch_size']}_seq{r['seq_len']}"
                          f"_steps{r['eval_steps']}", []).append(r['run'])
    return rows, groups


def windows_for(tok, token_bytes, adapter, bs, seq):
    """bpb on N_WINDOWS disjoint WINDOW-batch windows at this batch size."""
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, bs, seq, split='val', device=DEVICE)
    out = []
    for _ in range(N_WINDOWS):
        batches = [(x.clone(), y.clone()) for x, y in
                   (next(loader) for _ in range(WINDOW))]
        with torch.no_grad():
            out.append(float(evaluate_bpb(adapter, iter(batches), WINDOW, token_bytes)))
    return out


def main():
    rows, groups = scan_protocols()
    print(f"PROTOCOL SCAN: {len(rows)} runs")
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k}: {len(v)} runs")
    print()

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=DEVICE)

    results = {}
    for label, d, published, pub_bs in CKPTS:
        if not os.path.exists(os.path.join(d, 'checkpoint.pt')):
            print(f"  SKIP {label}: no checkpoint at {d}")
            continue
        cfg, m = M.build(d, load_checkpoint=True, dev=DEVICE)
        m.eval()
        ad = EvalAdapter(m).to(DEVICE)
        entry = dict(published=published, published_bs=pub_bs,
                     device_batch_size_in_config=cfg.get('device_batch_size'))
        for bs in BATCH_SIZES:
            entry[f'bs{bs}'] = windows_for(tok, token_bytes, ad, bs, cfg['seq_len'])
        entry['reproduces_published'] = abs(
            entry[f'bs{pub_bs}'][0] - published) < 5e-5
        results[label] = entry
        w = entry[f'bs{pub_bs}'][0]
        print(f"{label:<18} published {published:.5f} (bs{pub_bs}) | "
              f"reproduced w0 {w:.5f} -> {'MATCH' if entry['reproduces_published'] else 'MISMATCH'}")
        for bs in BATCH_SIZES:
            v = entry[f'bs{bs}']
            print(f"    bs{bs:<3} windows: " + "  ".join(f"{x:.5f}" for x in v) +
                  f"   mean {st.mean(v):.5f} sd {st.stdev(v):.5f}")
        del m, ad
        torch.cuda.empty_cache()

    with open(os.path.join(HERE, 'protocol_matrix.json'), 'w') as f:
        json.dump(dict(protocol_scan=rows, protocol_groups=groups,
                       window_batches=WINDOW, n_windows=N_WINDOWS,
                       results=results), f, indent=2)
    print(f"\nwrote {HERE}/protocol_matrix.json")


if __name__ == '__main__':
    main()
