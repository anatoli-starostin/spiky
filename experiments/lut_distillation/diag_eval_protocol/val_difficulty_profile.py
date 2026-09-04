"""Why is the START of validation easy while the first 246k tokens as a whole are hard?

Two parts.

PART 1 -- does batch size change the token stream?
The bestfit packer builds rows one at a time (`for row_idx in range(B)`), and the per-row
logic reads only `doc_buffer` and `row_capacity`; it never references B. The buffer persists
across batches. So the SEQUENCE OF ROWS should be identical for any B, with B only deciding
where batch boundaries fall -- making bs12's 120 rows a strict prefix of bs48's 480. That is a
claim about code; this verifies it on actual tensors, row by row.

PART 2 -- where does the difficulty actually sit?
If part 1 holds there is no paradox to explain, only a difficulty profile to measure: the
first 61k tokens are easy and something in 61k-246k is hard. This scores every block of 12
rows (one bs12 batch) across the first 480 rows and reports bpb alongside per-block features
(bytes/token, BOS density, and character statistics of the decoded text) so the hard region
can be located and characterised rather than narrated.

    python val_difficulty_profile.py
"""
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
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
for p in (os.path.join(REPO, 'experiments', 'ffn_replacement', 'benchmark'),
          os.path.join(REPO, 'src'), NANOCHAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M                                                    # noqa: E402
from nanochat.common import get_base_dir                             # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes     # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEACHER = os.path.join(REPO, 'experiments/ffn_replacement/runs/'
                             'exp_n_0151_long48k_untied_vanilla')
RESEARCH = '/home/astarostin/projects/spiky-fmhl-next'
LUT = os.path.join(RESEARCH, 'experiments/hyperplane_ffn/'
                             'exp_n_0127_grid_H4d48_nap7_tph128')
N_ROWS = 480          # = 40 bs12 batches = 10 bs48 batches = 245,760 tokens
BLOCK_ROWS = 12       # one bs12 batch per block


def rows_from(tok, bs, seq, n_rows):
    """First n_rows rows of the val stream at this batch size, as [n_rows, seq] tensors."""
    ld = tokenizing_distributed_data_loader_bos_bestfit(
        tok, bs, seq, split='val', device=DEVICE)
    xs, ys = [], []
    while sum(t.shape[0] for t in xs) < n_rows:
        x, y = next(ld)
        xs.append(x.clone())
        ys.append(y.clone())
    return torch.cat(xs)[:n_rows], torch.cat(ys)[:n_rows]


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=DEVICE)
    bos_id = tok.get_bos_token_id()
    seq = 512

    # ---------------- PART 1: is bs12 a strict prefix of bs48? ----------------
    x12, y12 = rows_from(tok, 12, seq, N_ROWS)
    x48, y48 = rows_from(tok, 48, seq, N_ROWS)
    same_120 = bool(torch.equal(x12[:120], x48[:120]) and torch.equal(y12[:120], y48[:120]))
    same_480 = bool(torch.equal(x12, x48) and torch.equal(y12, y48))
    first_diff = None
    if not same_480:
        for i in range(N_ROWS):
            if not torch.equal(x12[i], x48[i]):
                first_diff = i
                break
    print('PART 1 — batch size vs token stream')
    print(f'  bs12 first 120 rows identical to bs48 first 120 rows : {same_120}')
    print(f'  bs12 first 480 rows identical to bs48 first 480 rows : {same_480}')
    print(f'  first differing row index: {first_diff}')
    verdict = ('(a) bs12 prefix is TOKEN-IDENTICAL to the bs48 prefix'
               if same_120 and same_480 else
               '(b/c) streams diverge — nested-prefix framing is WRONG')
    print(f'  VERDICT: {verdict}\n')

    # ---------------- PART 2: difficulty profile ----------------
    models = {}
    for label, d in (('0151 vanilla', TEACHER), ('0127 LUT', LUT)):
        if os.path.exists(os.path.join(d, 'checkpoint.pt')):
            _, m = M.build(d, load_checkpoint=True, dev=DEVICE)
            m.eval()
            models[label] = m

    x, y = x12, y12                       # identical to x48 by part 1
    blocks = []
    for b in range(N_ROWS // BLOCK_ROWS):
        s = slice(b * BLOCK_ROWS, (b + 1) * BLOCK_ROWS)
        xb, yb = x[s], y[s]
        bts = token_bytes[yb]
        keep = bts > 0
        rec = dict(block=b, first_token=b * BLOCK_ROWS * seq,
                   bytes_per_token=float(bts.sum().item() / keep.sum().item()),
                   bos_per_1k=float(1000.0 * (yb == bos_id).sum().item() / yb.numel()))
        # decoded-text character statistics
        txt = tok.decode(yb.reshape(-1).tolist())
        n = max(1, len(txt))
        rec.update(nonascii_frac=sum(1 for c in txt if ord(c) > 127) / n,
                   digit_frac=sum(1 for c in txt if c.isdigit()) / n,
                   upper_frac=sum(1 for c in txt if c.isupper()) / n,
                   newline_per_1k=1000.0 * txt.count('\n') / n,
                   sample=txt[:160].replace('\n', ' '))
        for label, m in models.items():
            with torch.no_grad():
                l2 = F.cross_entropy(m(xb).view(-1, m.vocab_size).float(),
                                     yb.reshape(-1), ignore_index=-1,
                                     reduction='none').view(yb.shape)
            nats = (l2 * keep).sum().item()
            rec[f'bpb_{label.split()[0]}'] = nats / (math.log(2) * bts.sum().item())
            rec[f'nats_{label.split()[0]}'] = nats / keep.sum().item()
        blocks.append(rec)

    key = 'bpb_0151'
    print('PART 2 — difficulty profile, 12-row blocks (each = one bs12 batch)')
    print('  blk  tokens         bpb0151   nats   B/tok  BOS/1k  n-asc%  dig%  sample')
    for r in blocks:
        print(f"  {r['block']:>3}  {r['first_token']:>7,}  {r[key]:>8.5f} "
              f"{r['nats_0151']:>6.3f} {r['bytes_per_token']:>6.3f} "
              f"{r['bos_per_1k']:>6.2f} {100*r['nonascii_frac']:>6.2f} "
              f"{100*r['digit_frac']:>5.2f}  {r['sample'][:52]}")

    vals = [r[key] for r in blocks]
    order = sorted(blocks, key=lambda r: r[key])
    print(f"\n  mean {st.mean(vals):.5f}  sd {st.stdev(vals):.5f}  "
          f"range [{min(vals):.5f}, {max(vals):.5f}]")
    print(f"  EASIEST blocks: {[(r['block'], round(r[key], 4)) for r in order[:4]]}")
    print(f"  HARDEST blocks: {[(r['block'], round(r[key], 4)) for r in order[-4:]]}")
    q = [st.mean(vals[i:i + 10]) for i in range(0, 40, 10)]
    print(f"  by bs12 window (10 blocks each): " + "  ".join(f"w{i} {v:.5f}"
                                                              for i, v in enumerate(q)))

    # correlations of features with difficulty
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
        den = math.sqrt(sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))
        return num / den if den else 0.0

    print('\n  correlation of block features with bpb (0151):')
    feats = {}
    for f in ('bytes_per_token', 'bos_per_1k', 'nonascii_frac', 'digit_frac',
              'upper_frac', 'newline_per_1k', 'nats_0151'):
        feats[f] = corr([r[f] for r in blocks], vals)
        print(f'    {f:<18} r = {feats[f]:+.3f}')
    if 'bpb_0127' in blocks[0]:
        r = corr([b['bpb_0151'] for b in blocks], [b['bpb_0127'] for b in blocks])
        print(f'\n  0151 vs 0127 per-block bpb correlation: r = {r:+.4f} '
              f'(same hard/easy structure for both models)')

    with open(os.path.join(HERE, 'val_difficulty_profile.json'), 'w') as f:
        json.dump(dict(prefix_identical_120=same_120, prefix_identical_480=same_480,
                       first_differing_row=first_diff, verdict=verdict,
                       block_rows=BLOCK_ROWS, n_rows=N_ROWS,
                       blocks=blocks, feature_correlations=feats), f, indent=2)
    print(f'\nwrote {HERE}/val_difficulty_profile.json')


if __name__ == '__main__':
    main()
