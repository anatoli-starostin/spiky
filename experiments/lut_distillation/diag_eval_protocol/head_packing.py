"""What the best-fit packer actually emits at the head of the val stream (no model, fast).

head_anatomy.py showed block 0 is the easiest of 40 and that rows 0 and 8 are the two easiest
of all 480. This script explains the STRUCTURE those rows have, straight from the packer's
algorithm, and reconciles the BOS-density observations.

The packer (nanochat/dataloader.py) fills each row of capacity T+1 = 513 by repeatedly taking
the LARGEST buffered document that fits ENTIRELY; when nothing fits it CROPS THE SHORTEST
buffered document to the exact remaining space. Documents are BOS-prepended, and targets are
row[1:], so the row's leading BOS is dropped and only BOS tokens from the 2nd placement
onward are visible in y. That detail is what makes BOS-per-1k readable as packing structure:

    ~0 BOS/1k   -> rows are a single 513-token crop of a document longer than the row
    ~2 BOS/1k   -> one complete document (~500 tok) + a small cropped filler per row

    python head_packing.py
"""
import json
import os
import math
import statistics as st
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
for p in (os.path.join(REPO, 'src'), NANOCHAT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from nanochat.common import get_base_dir                             # noqa: E402
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes     # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402

SEQ = 512
N_ROWS = 2880          # 1.47M tokens -- well past the 245,760-token bs48 slice
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=DEVICE)
    bos = tok.get_bos_token_id()

    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, SEQ, split='val', device=DEVICE)
    ys = []
    while sum(t.shape[0] for t in ys) < N_ROWS:
        _, y = next(ld)
        ys.append(y.clone())              # reused GPU buffer -- clone
    y = torch.cat(ys)[:N_ROWS]
    is_bos = (y == bos)
    out = {}

    print('1. PACKING REGIME BY POSITION (BOS in targets = 2nd-and-later placements in a row)')
    print('   rows        tokens        BOS/row  BOS/1k  B/tok   regime')
    spans = [('0-11      (block 0)', 0, 12), ('12-119   (blocks 1-9)', 12, 120),
             ('120-479 (blocks 10-39)', 120, 480), ('480-1439', 480, 1440),
             ('1440-2879', 1440, 2880)]
    rows = []
    for name, a, b in spans:
        seg = is_bos[a:b]
        bts = token_bytes[y[a:b]].float()
        per_row = float(seg.sum()) / (b - a)
        regime = ('complete doc + filler' if per_row > 0.5 else
                  'single long-doc crop, no reset')
        bpt = float(bts.sum()) / float((bts > 0).sum())
        print(f'   {name:<24} {(b-a)*SEQ:>8,}  {per_row:>6.2f} '
              f'{1000*float(seg.sum())/seg.numel():>7.2f} {bpt:>6.3f}   {regime}')
        rows.append(dict(span=name, rows=b - a, bos_per_row=per_row, bytes_per_token=bpt))
    out['regimes'] = rows

    # where exactly does the regime change?
    per_row_bos = is_bos.sum(1).float().cpu().tolist()
    W = 48
    print('\n   BOS-per-row, 48-row moving blocks (the transition is sharp, not gradual):')
    for s in range(0, 960, W):
        v = st.mean(per_row_bos[s:s + W])
        print(f'   rows {s:>4}-{s+W-1:<4} tok {s*SEQ:>8,}  {v:>5.2f}  ' + '#' * int(v * 20))
    change = next((s for s in range(0, N_ROWS - W, W)
                   if st.mean(per_row_bos[s:s + W]) < 0.5), None)
    print(f'   first 48-row block averaging <0.5 BOS/row: rows {change}-{change+W-1} '
          f'(token {change*SEQ:,})')
    out['regime_change_row'] = change

    print('\n2. THE ROWS THEMSELVES — rows 0-11, and the two easiest rows of all 480')
    for r in list(range(12)):
        seg = y[r].tolist()
        b = [i for i, t in enumerate(seg) if t == bos]
        txt = ' '.join(tok.decode([t for t in seg if t != bos]).split())
        bts = token_bytes[y[r]].float()
        print(f'   row {r:>2}  BOS at {b}  B/tok {float(bts.sum())/float((bts>0).sum()):.3f}')
        print(f'          {txt[:150]}')
    out['head_rows'] = [dict(row=r, bos_positions=[i for i, t in enumerate(y[r].tolist())
                                                   if t == bos],
                             text=' '.join(tok.decode([t for t in y[r].tolist()
                                                       if t != bos]).split())[:600])
                        for r in range(12)]

    print('\n3. THE PACKER SORTS DOCUMENTS BY LENGTH AT STREAM START')
    print('   "largest doc that fits in 513" applied to a fresh buffer drains the longest-'
          'fitting\n   documents first, so placed length falls monotonically from row 0:')
    first = []
    for r in range(480):
        b = [i for i, t in enumerate(y[r].tolist()) if t == bos]
        first.append(b[0] if b else SEQ)      # length of the complete doc placed at row start
    print(f'   rows 0-23: {first[:24]}')
    prof = os.path.join(HERE, 'val_difficulty_profile.json')
    if os.path.exists(prof):
        bp = [b['bpb_0151'] for b in json.load(open(prof))['blocks']]
        L = [st.mean(first[i*12:(i+1)*12]) for i in range(40)]
        print(f'   block 0 mean placed-doc length {L[0]:.1f} tokens '
              f'({SEQ-L[0]:.1f} tokens of truncated filler per row);')
        print(f'   blocks 1-39 mean {st.mean(L[1:]):.1f} '
              f'({SEQ-st.mean(L[1:]):.1f} truncated). Block 0 is the least-fragmented block '
              f'of the 40.')

        def corr(a, b):
            ma, mb = st.mean(a), st.mean(b)
            n = sum((u-ma)*(v-mb) for u, v in zip(a, b))
            return n / math.sqrt(sum((u-ma)**2 for u in a) * sum((v-mb)**2 for v in b))
        print(f'   corr(block placed-doc length, block bpb) r = {corr(L, bp):+.3f} '
              f'-- right sign, weak.')
        xs, yy = L[1:], bp[1:]
        mx, my = st.mean(xs), st.mean(yy)
        sl = sum((a-mx)*(b-my) for a, b in zip(xs, yy)) / sum((a-mx)**2 for a in xs)
        ic = my - sl*mx
        pred0 = ic + sl*L[0]
        resid = st.stdev([b - (ic+sl*a) for a, b in zip(xs, yy)])
        print(f'   OLS on blocks 1-39 predicts block 0 at {pred0:.5f}; it actually reads '
              f'{bp[0]:.5f}')
        print(f'     structural (packing) part of the deficit: {pred0-st.mean(bp):+.5f}')
        print(f'     content part (residual):                  {bp[0]-pred0:+.5f}  '
              f'({(bp[0]-pred0)/resid:+.2f} residual sd)')
        out['doc_len_rows'] = first
        out['packing_vs_content'] = dict(pred_block0=pred0, actual_block0=bp[0],
                                         structural=pred0-st.mean(bp), content=bp[0]-pred0,
                                         corr_doclen_bpb=corr(L, bp))

    with open(os.path.join(HERE, 'head_packing.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {HERE}/head_packing.json')


if __name__ == '__main__':
    main()
