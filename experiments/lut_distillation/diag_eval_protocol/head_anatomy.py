"""What exactly is block 0, and why is the very head of the val stream so easy?

val_difficulty_profile.py resolved the val stream into 40 blocks of 12 rows and found block 0
(0.98681) the easiest of the forty. That is coarse: a block is 6,144 tokens. This resolves the
head to the ROW (512 tokens) and, inside row 0, to the TOKEN POSITION, then names the actual
documents the packer emitted at stream start.

Everything is scored on ONE cached copy of the first 480 rows, cloned off the loader (it
yields views into a single reused GPU buffer -- caching without .clone() silently re-scores
the last batch). Per-row nats and bytes are accumulated so that any grouping -- row, block,
window -- is a ratio of the same sums, exactly as `evaluate_bpb` computes it.

    python head_anatomy.py
"""
import json
import math
import os
import statistics as st
import sys

import torch
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
SEQ = 512
N_ROWS = 480          # 245,760 tokens = the bs48 eval slice = 40 blocks of 12 rows
BLOCK = 12            # rows per block (one bs12 batch)
CHUNK = 24            # rows per forward pass
LN2 = math.log(2)
BOS_SCAN_TOKENS = 1_474_560   # 2880 rows -- BOS-density curve well past the 246k slice


def load_rows(tok, n_rows, bs=48):
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, bs, SEQ, split='val',
                                                        device=DEVICE)
    xs, ys = [], []
    while sum(t.shape[0] for t in xs) < n_rows:
        x, y = next(ld)
        xs.append(x.clone())          # loader reuses one GPU buffer -- clone or lose it
        ys.append(y.clone())
    return torch.cat(xs)[:n_rows], torch.cat(ys)[:n_rows]


def score(model, x, y, token_bytes):
    """Per-row and per-(row, position) nats and bytes, on counted tokens only."""
    nats_rp, bytes_rp = [], []
    for i in range(0, x.shape[0], CHUNK):
        xb, yb = x[i:i + CHUNK], y[i:i + CHUNK]
        with torch.no_grad():
            l = F.cross_entropy(model(xb).view(-1, model.vocab_size).float(),
                                yb.reshape(-1), ignore_index=-1,
                                reduction='none').view(yb.shape)
        b = token_bytes[yb].float()
        nats_rp.append((l * (b > 0)).float().cpu())
        bytes_rp.append(b.cpu())
    return torch.cat(nats_rp), torch.cat(bytes_rp)


def bpb(nats, byts):
    return float(nats.sum()) / (LN2 * float(byts.sum()))


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    token_bytes = get_token_bytes(device=DEVICE)
    bos_id = tok.get_bos_token_id()
    out = {}

    x, y = load_rows(tok, N_ROWS)
    is_bos = (y == bos_id)

    models = {}
    for label, d in (('0151', TEACHER), ('0127', LUT)):
        if os.path.exists(os.path.join(d, 'checkpoint.pt')):
            _, m = M.build(d, load_checkpoint=True, dev=DEVICE)
            m.eval()
            models[label] = m
    nats, byts = {}, None
    for label, m in models.items():
        nats[label], byts = score(m, x, y, token_bytes)

    row_bpb = {k: [bpb(v[i], byts[i]) for i in range(N_ROWS)] for k, v in nats.items()}
    blk_bpb = {k: [bpb(v[b * BLOCK:(b + 1) * BLOCK], byts[b * BLOCK:(b + 1) * BLOCK])
                   for b in range(N_ROWS // BLOCK)] for k, v in nats.items()}
    r0 = row_bpb['0151']

    # ---------------- 1. row resolution at the head ----------------
    print('1. PER-ROW bpb AT THE HEAD (0151 vanilla; each row = 512 tokens)\n')
    order = sorted(range(N_ROWS), key=lambda i: r0[i])
    rank = {i: k for k, i in enumerate(order)}          # 0 = easiest of 480
    print('  row  blk    bpb     nats/t  B/tok  BOS  rank/480   '
          '| row  blk    bpb     nats/t  B/tok  BOS  rank/480')
    for i in range(0, 42, 2):
        cells = []
        for j in (i, i + 1):
            if j >= 42:
                cells.append('')
                continue
            cells.append(f"{j:>4} {j//BLOCK:>4}  {r0[j]:>7.5f}  "
                         f"{float(nats['0151'][j].sum())/float((byts[j]>0).sum()):>6.3f} "
                         f"{float(byts[j].sum())/float((byts[j]>0).sum()):>6.3f} "
                         f"{int(is_bos[j].sum()):>3}  {rank[j]:>6}")
        print('  ' + ' | '.join(cells))

    head_ranks = [rank[i] for i in range(BLOCK)]
    print(f"\n  ranks of rows 0-11 among all 480 (0 = easiest): {head_ranks}")
    print(f"  median rank of rows 0-11: {st.median(head_ranks):.1f}   "
          f"(a typical block would sit near 240)")
    print(f"  rows 0-11 in the easiest 48 (decile): "
          f"{sum(1 for r in head_ranks if r < 48)}/12")
    b0 = blk_bpb['0151'][0]
    print(f"\n  block 0 = {b0:.5f}   rows 0-11 individually: "
          f"{[round(r0[i], 4) for i in range(BLOCK)]}")
    print(f"  block 0 without row 0     : "
          f"{bpb(nats['0151'][1:BLOCK], byts[1:BLOCK]):.5f}")
    print(f"  block 0 without rows 0-1  : "
          f"{bpb(nats['0151'][2:BLOCK], byts[2:BLOCK]):.5f}")
    print(f"  40-block mean {st.mean(blk_bpb['0151']):.5f}; "
          f"blocks 1-39 pooled {bpb(nats['0151'][BLOCK:], byts[BLOCK:]):.5f}")
    out['row_bpb_0151'] = r0
    out['row_ranks_head'] = head_ranks
    out['block_bpb'] = blk_bpb

    # ---------------- 2. position-within-sequence ----------------
    print('\n2. bpb BY POSITION WITHIN THE 512-TOKEN SEQUENCE')
    n0, b0p = nats['0151'], byts
    BINS = 16
    w = SEQ // BINS
    pos_all, pos_row0, pos_rest = [], [], []
    for k in range(BINS):
        s = slice(k * w, (k + 1) * w)
        pos_all.append(bpb(n0[:, s], b0p[:, s]))
        pos_row0.append(bpb(n0[0:1, s], b0p[0:1, s]))
        pos_rest.append(bpb(n0[BLOCK:, s], b0p[BLOCK:, s]))
    print('   positions      row 0    all 480   rows 12-479   row0 - rows12+')
    for k in range(BINS):
        print(f'   {k*w:>3}-{(k+1)*w-1:<3}      {pos_row0[k]:>7.5f}  {pos_all[k]:>7.5f}   '
              f'{pos_rest[k]:>7.5f}      {pos_row0[k]-pos_rest[k]:>+8.5f}')
    print(f'   overall       {bpb(n0[0:1], b0p[0:1]):>7.5f}  {bpb(n0, b0p):>7.5f}   '
          f'{bpb(n0[BLOCK:], b0p[BLOCK:]):>7.5f}      '
          f'{bpb(n0[0:1], b0p[0:1]) - bpb(n0[BLOCK:], b0p[BLOCK:]):>+8.5f}')
    out['position_profile'] = dict(bins=BINS, row0=pos_row0, all=pos_all, rows12plus=pos_rest)

    # ---------------- 3. the actual documents ----------------
    print('\n3. DOCUMENTS IN ROWS 0-11')
    flat = y[:BLOCK].reshape(-1).tolist()
    starts = [i for i, t in enumerate(flat) if t == bos_id]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    docs = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(flat)
        txt = tok.decode([t for t in flat[s:e] if t != bos_id])
        docs.append(dict(doc=k, tok_start=s, n_tokens=e - s,
                         row_start=s // SEQ, row_end=(e - 1) // SEQ,
                         excerpt=' '.join(txt.split())[:110]))
    print(f'   {len(docs)} document segments across the 6,144 tokens of block 0')
    print('   #   tokens  rows      excerpt')
    for d in docs:
        span = (f"{d['row_start']}" if d['row_start'] == d['row_end']
                else f"{d['row_start']}-{d['row_end']}")
        print(f"   {d['doc']:>2}  {d['n_tokens']:>6}  {span:<8}  {d['excerpt'][:88]}")
    lens = [d['n_tokens'] for d in docs]
    print(f"   block-0 doc lengths: mean {st.mean(lens):.0f}, median {st.median(lens):.0f}, "
          f"max {max(lens)}, min {min(lens)}")
    out['block0_docs'] = docs

    print('\n   BOS density and mean document length by span (within the 246k slice):')
    print('   span            tokens    BOS  BOS/1k  mean doc len (tok)')
    for name, sl in (('block 0', slice(0, BLOCK)), ('blocks 1-9', slice(BLOCK, 10 * BLOCK)),
                     ('blocks 10-39', slice(10 * BLOCK, N_ROWS)),
                     ('all 40 blocks', slice(0, N_ROWS))):
        nb = int(is_bos[sl].sum())
        nt = is_bos[sl].numel()
        print(f'   {name:<14} {nt:>7,} {nb:>6} {1000*nb/nt:>7.2f}  {nt/max(1,nb):>10.0f}')
    out['bos_by_span'] = {name: dict(tokens=int(is_bos[sl].numel()), bos=int(is_bos[sl].sum()))
                          for name, sl in (('block0', slice(0, BLOCK)),
                                           ('blocks1_9', slice(BLOCK, 10 * BLOCK)),
                                           ('blocks10_39', slice(10 * BLOCK, N_ROWS)))}

    print('\n   BOS-density-vs-position curve, 61,440-token bins, first 1.47M val tokens')
    print('   (no model; loader only. RECONCILES the earlier "2.18 vs 0.70 per 1k" finding,')
    print('    which compared bs48 window 0 = tokens 0-245,760 against LATER windows.)')
    xb, yb = load_rows(tok, BOS_SCAN_TOKENS // SEQ)
    bos_flat = (yb.reshape(-1) == bos_id)
    BIN = 61440
    curve = []
    for s in range(0, bos_flat.numel(), BIN):
        seg = bos_flat[s:s + BIN]
        curve.append(dict(first_token=s, bos=int(seg.sum()),
                          bos_per_1k=1000 * float(seg.sum()) / seg.numel()))
    for c in curve:
        bar = '#' * int(round(c['bos_per_1k'] * 8))
        mark = '  <- bs12 slice' if c['first_token'] == 0 else (
               '  <- end of bs48 slice' if c['first_token'] == 245760 - BIN else '')
        print(f"   {c['first_token']:>9,}  {c['bos_per_1k']:>5.2f}  {bar}{mark}")
    first246 = 1000 * sum(c['bos'] for c in curve[:4]) / (4 * BIN)
    rest = 1000 * sum(c['bos'] for c in curve[4:]) / ((len(curve) - 4) * BIN)
    print(f"   tokens 0-245,760: {first246:.2f}/1k     tokens 245,760+: {rest:.2f}/1k "
          f"  ratio {first246/rest:.2f}x")
    out['bos_curve'] = curve
    del xb, yb, bos_flat

    # ---------------- 4. block 0 decomposition ----------------
    print('\n4. BLOCK 0 DECOMPOSITION: model confidence vs tokenizer fertility')
    mean40 = st.mean(blk_bpb['0151'])
    keep0 = (byts[:BLOCK] > 0)
    n_b0 = float(nats['0151'][:BLOCK].sum()) / float(keep0.sum())
    b_b0 = float(byts[:BLOCK].sum()) / float(keep0.sum())
    keepA = (byts > 0)
    n_all = float(nats['0151'].sum()) / float(keepA.sum())
    b_all = float(byts.sum()) / float(keepA.sum())
    deficit = blk_bpb['0151'][0] - mean40
    # counterfactuals: block-0 nats at global fertility, global nats at block-0 fertility
    cf_nats_only = n_b0 / (LN2 * b_all) - n_all / (LN2 * b_all)
    cf_byte_only = n_all / (LN2 * b_b0) - n_all / (LN2 * b_all)
    print(f'   block 0   nats/token {n_b0:.4f}   bytes/token {b_b0:.4f}   bpb {blk_bpb["0151"][0]:.5f}')
    print(f'   40-block  nats/token {n_all:.4f}   bytes/token {b_all:.4f}   bpb {bpb(nats["0151"], byts):.5f}')
    print(f'   deficit vs 40-block MEAN of block bpb: {deficit:+.5f}')
    print(f'     attributable to lower nats/token   : {cf_nats_only:+.5f}  '
          f'({100*cf_nats_only/(cf_nats_only+cf_byte_only):.0f}%)')
    print(f'     attributable to higher bytes/token : {cf_byte_only:+.5f}  '
          f'({100*cf_byte_only/(cf_nats_only+cf_byte_only):.0f}%)')
    print(f'   block 0 bpb if it had average fertility: '
          f'{n_b0/(LN2*b_all):.5f}  (vs {n_all/(LN2*b_all):.5f} average)')
    out['block0_decomposition'] = dict(nats_b0=n_b0, bytes_b0=b_b0, nats_all=n_all,
                                       bytes_all=b_all, deficit=deficit,
                                       from_nats=cf_nats_only, from_bytes=cf_byte_only)

    # ---------------- 5. cross-model check ----------------
    if '0127' in row_bpb:
        print('\n5. CROSS-MODEL CHECK (0127 LUT)')
        rL = row_bpb['0127']
        print('   row     0151      0127     diff')
        for i in range(12):
            print(f'   {i:>3}  {r0[i]:>8.5f}  {rL[i]:>8.5f}  {rL[i]-r0[i]:>+7.5f}')
        for lab in ('0151', '0127'):
            bb = blk_bpb[lab]
            print(f'   {lab}: block 0 {bb[0]:.5f}   40-block mean {st.mean(bb):.5f}   '
                  f'deficit {bb[0]-st.mean(bb):+.5f}   '
                  f'z {(bb[0]-st.mean(bb))/st.stdev(bb):+.2f}')
        def corr(a, b):
            ma, mb = st.mean(a), st.mean(b)
            num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
            den = math.sqrt(sum((u-ma)**2 for u in a) * sum((v-mb)**2 for v in b))
            return num / den if den else 0.0
        print(f"   per-ROW bpb correlation over 480 rows: r = {corr(r0, rL):+.4f}")
        print(f"   per-row correlation over the head (rows 0-11): "
              f"r = {corr(r0[:12], rL[:12]):+.4f}")
        agree = '' if (blk_bpb['0127'][0] - st.mean(blk_bpb['0127'])) < -0.10 else \
                '   *** MODELS DISAGREE AT THE HEAD -- INVESTIGATE ***'
        print(f'   verdict: block 0 is anomalously easy for BOTH models{agree}')
        out['row_bpb_0127'] = rL

    # ---------------- 6. practical consequence ----------------
    print('\n6. WHAT TO SKIP, AND HOW MUCH IT BUYS')
    full = bpb(nats['0151'], byts)                      # the bs48 (246k) value
    def win(skip_rows, n_rows_):
        s = slice(skip_rows, skip_rows + n_rows_)
        return bpb(nats['0151'][s], byts[s])
    b12 = win(0, 120)
    print(f'   reference: full 480-row (bs48) value            {full:.5f}')
    print(f'   bs12 window as published (rows 0-119)           {b12:.5f}  '
          f'bias {b12-full:+.5f}')
    for skip in (12, 24, 48):
        v = win(skip, 120)
        print(f'   same 120-row length, skipping first {skip:>2} rows   {v:.5f}  '
              f'bias {v-full:+.5f}')
    print('\n   bias of a leading window of L rows vs the full 480-row value:')
    print('   skip\\L      60      120      240      360')
    for skip in (0, 12, 24, 48, 96):
        cells = []
        for L in (60, 120, 240, 360):
            cells.append(f'{win(skip, L)-full:>+8.5f}' if skip + L <= N_ROWS else '       -')
        print(f'   {skip:>4}   ' + ' '.join(cells))
    mins = None
    for skip in range(0, 200):
        if abs(win(skip, 120) - full) < 0.005:
            mins = skip
            break
    print(f'\n   smallest leading skip making a 120-row window unbiased to <0.005 bpb: '
          f'{mins} rows = {None if mins is None else mins*SEQ:,} tokens')
    out['practical'] = dict(full=full, bs12=b12, bias_bs12=b12 - full,
                            skip_12=win(12, 120) - full, skip_24=win(24, 120) - full,
                            skip_48=win(48, 120) - full, min_skip_rows=mins)

    with open(os.path.join(HERE, 'head_anatomy.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {HERE}/head_anatomy.json')

    # ---------------- chart ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.5), height_ratios=[3, 2])
    xs = list(range(N_ROWS))
    ax[0].axvspan(0, BLOCK, color='#f6d9c8', alpha=.9, label='block 0')
    ax[0].axvspan(BLOCK, 120, color='#cfe8d4', alpha=.5, label='rest of bs12 window')
    for b in range(0, N_ROWS, BLOCK):
        ax[0].axvline(b, color='#dddddd', lw=.6, zorder=0)
    ax[0].plot(xs, r0, lw=1.0, color='#2f6f4f', alpha=.85)
    ax[0].plot(xs, r0, '.', ms=3.5, color='#2f6f4f')
    if '0127' in row_bpb:
        ax[0].plot(xs, row_bpb['0127'], lw=.9, color='#8a5a2b', alpha=.55, label='0127 LUT')
    ax[0].axhline(full, color='#444', ls='--', lw=1.2, label=f'480-row value {full:.4f}')
    ax[0].set_xlim(-2, N_ROWS); ax[0].set_ylabel('val bpb')
    ax[0].set_title('Per-row val bpb (512 tokens per row), 0151 vanilla — block boundaries every 12 rows',
                    fontsize=11)
    ax[0].legend(fontsize=8, ncol=4, loc='upper right'); ax[0].grid(alpha=.2)

    ax[1].axvspan(0, BLOCK, color='#f6d9c8', alpha=.9)
    for b in range(0, 60, BLOCK):
        ax[1].axvline(b, color='#bbbbbb', lw=.8, zorder=0)
    ax[1].plot(range(60), r0[:60], 'o-', ms=5, lw=1.6, color='#2f6f4f', label='0151 vanilla')
    if '0127' in row_bpb:
        ax[1].plot(range(60), row_bpb['0127'][:60], 's-', ms=4, lw=1.3,
                   color='#8a5a2b', alpha=.85, label='0127 LUT')
    ax[1].axhline(full, color='#444', ls='--', lw=1.2)
    for i in range(0, 60, 12):
        ax[1].annotate(f'blk {i//12}', (i + .3, ax[1].get_ylim()[1]), fontsize=8,
                       va='top', color='#666')
    ax[1].set_xlabel('row index (each row = 512 tokens)'); ax[1].set_ylabel('val bpb')
    ax[1].set_title('zoom: the first 60 rows', fontsize=10)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
    plt.tight_layout()
    png = os.path.join('/tmp', 'head_anatomy.png')
    plt.savefig(png, dpi=135)
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
