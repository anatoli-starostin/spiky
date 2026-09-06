"""Is Light's routing bad because the GRADIENT is weak, or because the HASH is bad?

The question this answers: LookupFFN might work because its LSH simply distributes
addresses well, which would make a weak routing gradient affordable. If our addresses are
already well spread, a bad-hash explanation is ruled out and the deficit stays on the
gradient side; if ours are badly spread, the hash family itself is a target.

WHAT IS MEASURED, per run / per layer / per head, on real val tokens:
  1. bucket occupancy   -- distinct cells addressed out of 2^NAP, top-1 and top-10 share,
                           Gini skew
  2. address entropy    -- bits, against NAP (the address width) so the ratio is readable
  3. collision rate     -- sum_c p_c^2 (probability two random tokens share a cell) and the
                           effective bucket count 1/sum p_c^2
  4. bit correlation    -- our address bits are sign(z[a_j] - z[b_j]) with FIXED random
                           pairs, so pairs sharing an endpoint coordinate induce correlated
                           bits and cost effective buckets. Reports how often pairs share an
                           endpoint, plus mean/max |corr| and the count of |corr| > 0.5
  5. contrast           -- the SAME statistic for LookupFFN-style coordinate-sign addressing
                           on the SAME z (post-hoc; no retraining, no new model)

CPU ONLY, and deliberately so: a training run may be in flight and this must not perturb
it. Torch threads are capped for the same reason.

BOTH implementations expose flat anchor buffers indexing a flat [N, n_heads*inner_in_dim]
code -- Light as native_anchor_a/b, Fast as soft_anchor_a_long/b_long -- so one code path
reads addresses from either. Note Fast (exp_n_0129) runs tph=256 where the Light runs use
tph=128: it has twice as many tables, which is stated with the results rather than
normalised away.

    python diag_hash_quality.py [--tokens 2048]
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(FR, 'tools'))
sys.path.insert(0, os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))

RUNS = [
    ('exp_g_0190 z_norm  (best Light)', 'exp_g_0190_B16k_light_bnorm_tph128_znorm_seed1'),
    ('exp_g_0189 nodecay (Light ctrl)', 'exp_g_0189_B16k_light_bnorm_tph128_nodecay_seed1'),
    ('exp_n_0185 Light pre-decay-fix ', 'exp_n_0185_B16k_light_bnorm_tph128_seed1'),
    ('exp_n_0129 FAST (the one that works)', 'exp_n_0129_grid_H4d48_nap8_tph256'),
]


def gini(p):
    """Gini of a probability vector (0 = uniform over the cells present, 1 = all mass in one)."""
    x, _ = torch.sort(p)
    n = x.numel()
    idx = torch.arange(1, n + 1, dtype=x.dtype)
    s = x.sum()
    return float(((2 * idx - n - 1) * x).sum() / (n * s)) if s > 0 else float('nan')


def bit_corr_stats(bits):
    """bits [N, T, B] in {0,1} -> (mean |corr|, max |corr|, frac of bit-pairs |corr|>0.5).

    Correlation is computed per table over the N tokens, then summarised over tables and
    over the B*(B-1)/2 distinct bit pairs. Constant bits (a bit that never flips) have
    undefined correlation; they are counted separately and excluded from the mean, because
    calling them 'uncorrelated' would flatter the hash.
    """
    N, T, B = bits.shape
    x = bits.to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)                      # [N,T,B]
    sd = x.pow(2).mean(dim=0).sqrt()                         # [T,B]
    cov = torch.einsum('nta,ntb->tab', x, x) / N             # [T,B,B]
    denom = sd.unsqueeze(2) * sd.unsqueeze(1)                # [T,B,B]
    ok = denom > 1e-12
    corr = torch.where(ok, cov / denom.clamp_min(1e-12), torch.zeros_like(cov)).abs()
    iu = torch.triu_indices(B, B, offset=1)
    c = corr[:, iu[0], iu[1]]                                # [T, B(B-1)/2]
    valid = ok[:, iu[0], iu[1]]
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float('nan'), float('nan'), float('nan'), 1.0
    dead = float((sd <= 1e-12).sum()) / (T * B)
    return (float(c[valid].mean()), float(c[valid].max()),
            float((c[valid] > 0.5).sum()) / n_valid, dead)


def endpoint_sharing(a, b):
    """Fraction of the B(B-1)/2 anchor-pair pairs, per table, that share an endpoint coord."""
    T, B = a.shape
    iu = torch.triu_indices(B, B, offset=1)
    a1, b1, a2, b2 = a[:, iu[0]], b[:, iu[0]], a[:, iu[1]], b[:, iu[1]]
    share = ((a1 == a2) | (a1 == b2) | (b1 == a2) | (b1 == b2))
    per_table = share.to(torch.float64).mean(dim=1)
    return float(per_table.mean()), float((per_table > 0).to(torch.float64).mean())


def analyse(bits, powers, nap, label, tag, out):
    """bits [N,T,B] -> occupancy / entropy / collision, aggregated over the T tables."""
    N, T, B = bits.shape
    idx = (bits.long() * powers).sum(-1)                     # [N, T]
    cells = 1 << nap
    acc = {k: [] for k in ('distinct', 'frac', 'top1', 'top10', 'gini', 'H', 'coll', 'eff')}
    for t in range(T):
        cnt = torch.bincount(idx[:, t], minlength=cells).to(torch.float64)
        p = cnt / cnt.sum()
        nz = p[p > 0]
        srt, _ = torch.sort(p, descending=True)
        H = float(-(nz * nz.log2()).sum())
        coll = float((p * p).sum())
        acc['distinct'].append(float((cnt > 0).sum()))
        acc['frac'].append(float((cnt > 0).sum()) / cells)
        acc['top1'].append(float(srt[0]))
        acc['top10'].append(float(srt[:10].sum()))
        acc['gini'].append(gini(p))
        acc['H'].append(H)
        acc['coll'].append(coll)
        acc['eff'].append(1.0 / coll if coll > 0 else float('nan'))
    mean = {k: sum(v) / len(v) for k, v in acc.items()}
    mc, xc, hi, dead = bit_corr_stats(bits)
    out.append(dict(label=label, tag=tag, nap=nap, cells=cells, n_tables=T, **mean,
                    corr_mean=mc, corr_max=xc, corr_hi=hi, dead_bits=dead))
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=2048)
    args = ap.parse_args()
    torch.set_num_threads(4)          # be a polite neighbour to any live training run

    from nanochat.common import get_base_dir
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from model_build import build_model

    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    vocab = tok.get_vocab_size()
    # Same val shard and the same leading-12-row skip as the scoring protocol.
    ld = tokenizing_distributed_data_loader_bos_bestfit(tok, 48, 512, split='val',
                                                        device='cpu')
    x_all, _ = next(iter(ld))
    n_rows = max(1, args.tokens // 512)
    x_ids = x_all[12:12 + n_rows].clone()
    print(f'val tokens: {x_ids.numel():,} ({n_rows} rows x 512, leading 12 rows skipped, '
          f'same shard as the scoring protocol)\n')

    rows, rows_coord = [], []
    for label, run in RUNS:
        d = os.path.join(HERE, run)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        torch.manual_seed(cfg['random_seed'])
        m = build_model(cfg, vocab, device='cpu')
        sd = torch.load(os.path.join(d, 'checkpoint.pt'), map_location='cpu',
                        weights_only=True)
        miss, unexp = m.load_state_dict(sd, strict=True), None
        m.eval()

        cap = {}
        hooks = []
        for li, blk in enumerate(m.blocks):
            def mk(li, ffn):
                def h(mod, inp, out):
                    z = ffn.compress(inp[0])
                    if getattr(ffn, 'z_norm', None) is not None:
                        z = ffn.z_norm(z.view(-1, ffn.n_heads, ffn.inner_in_dim))
                        z = z.reshape(inp[0].shape[0], -1)
                    cap[li] = z.detach()
                return h
            hooks.append(blk.ffn.register_forward_hook(mk(li, blk.ffn)))
        with torch.no_grad():
            m(x_ids)
        for h in hooks:
            h.remove()

        f0 = m.blocks[0].ffn
        lut0 = getattr(f0, 'lut_light', None) or f0.lut_batched
        nap = int(lut0.anchor_a.shape[-1]) if hasattr(lut0, 'anchor_a') \
            else int(lut0.soft_anchor_a_long.shape[-1])
        n_heads, din = f0.n_heads, f0.inner_in_dim
        print('=' * 104)
        print(f'{label}   [{run}]')
        print(f'   NAP (address width) = {nap} bits -> {1 << nap} cells/table; '
              f'{n_heads} heads x {getattr(lut0, "tables_per_head", "?")} tables/head; '
              f'inner_in_dim {din}')
        print('=' * 104)
        print(f'   {"layer":>5}{"head":>5}{"distinct/256":>14}{"frac":>8}{"top1":>8}'
              f'{"top10":>8}{"gini":>7}{"H bits":>8}{"H/NAP":>7}{"coll":>9}{"eff.buck":>10}'
              f'{"|corr|":>8}{"max|c|":>8}{">0.5":>7}')
        for li in range(len(m.blocks)):
            ffn = m.blocks[li].ffn
            lut = getattr(ffn, 'lut_light', None) or ffn.lut_batched
            fa = getattr(lut, 'native_anchor_a', None)
            fb = getattr(lut, 'native_anchor_b', None)
            if fa is None:
                fa, fb = lut.soft_anchor_a_long, lut.soft_anchor_b_long
            powers = (1 << torch.arange(nap))
            z = cap[li]                                   # [N, n_heads*din]
            tph = fa.shape[0] // n_heads
            for hh in range(n_heads):
                sl = slice(hh * tph, (hh + 1) * tph)
                a, b = fa[sl], fb[sl]
                bits = (z[:, a] - z[:, b] > 0)            # [N, tph, nap]
                mean = analyse(bits, powers, nap, label, (li, hh), rows)
                sh_mean, sh_any = endpoint_sharing(a, b)
                mc, xc, hi, _ = bit_corr_stats(bits)
                print(f'   {li:>5}{hh:>5}{mean["distinct"]:>14.1f}{mean["frac"]:>8.3f}'
                      f'{mean["top1"]:>8.4f}{mean["top10"]:>8.3f}{mean["gini"]:>7.3f}'
                      f'{mean["H"]:>8.3f}{mean["H"]/nap:>7.3f}{mean["coll"]:>9.5f}'
                      f'{mean["eff"]:>10.1f}{mc:>8.3f}{xc:>8.3f}{hi:>7.3f}')
                rows[-1]['share_mean'], rows[-1]['share_any'] = sh_mean, sh_any

                # CONTRAST: LookupFFN-style coordinate-sign addressing on the SAME z.
                g = torch.Generator().manual_seed(1234 + li * 16 + hh)
                base = hh * din
                coord = torch.stack([base + torch.randperm(din, generator=g)[:nap]
                                     for _ in range(tph)])          # [tph, nap]
                cbits = (z[:, coord] > 0)
                analyse(cbits, powers, nap, label, (li, hh), rows_coord)
        print()
        del m, cap
    return rows, rows_coord


def summarise(rows, rows_coord):
    import collections
    print('=' * 104)
    print('SUMMARY — averaged over all layers and heads of each run')
    print('=' * 104)
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r['label'], []).append(r)
    byc = collections.OrderedDict()
    for r in rows_coord:
        byc.setdefault(r['label'], []).append(r)
    def avg(v, k):
        """nan-aware: a run where every bit of some head is CONSTANT has undefined
        correlation there, and averaging it in as 0 would flatter that scheme."""
        x = [e[k] for e in v if not (isinstance(e[k], float) and math.isnan(e[k]))]
        return sum(x) / len(x) if x else float('nan')
    print('OURS — address = sign(z[a_j] - z[b_j]), fixed random anchor pairs')
    print(f'   {"run":<38}{"frac cells":>11}{"H/NAP":>8}{"eff.buck":>10}{"gini":>7}'
          f'{"|corr|":>9}{"dead bits":>11}{"share%":>8}')
    for lab, v in by.items():
        print(f'   {lab:<38}{avg(v,"frac"):>11.3f}{avg(v,"H")/v[0]["nap"]:>8.3f}'
              f'{avg(v,"eff"):>10.1f}{avg(v,"gini"):>7.3f}{avg(v,"corr_mean"):>9.4f}'
              f'{100*avg(v,"dead_bits"):>10.2f}%{100*avg(v,"share_mean"):>8.2f}')
    print('\nCONTRAST — LookupFFN-style address = sign(z_coord), SAME codes, post-hoc')
    print(f'   {"run":<38}{"frac cells":>11}{"H/NAP":>8}{"eff.buck":>10}{"gini":>7}'
          f'{"|corr|":>9}{"dead bits":>11}')
    for lab, c in byc.items():
        print(f'   {lab:<38}{avg(c,"frac"):>11.3f}{avg(c,"H")/c[0]["nap"]:>8.3f}'
              f'{avg(c,"eff"):>10.1f}{avg(c,"gini"):>7.3f}{avg(c,"corr_mean"):>9.4f}'
              f'{100*avg(c,"dead_bits"):>10.2f}%')
    print(f'\n   "share%" = fraction of anchor-pair PAIRS within a table sharing an endpoint'
          f' coordinate.\n   "dead bits" = address bits whose sign NEVER flips across the'
          f' scored tokens; such a bit\n   carries no information and halves the reachable'
          f' cells, so it is reported rather than\n   silently dropped from the correlation'
          f' mean.')


if __name__ == '__main__':
    summarise(*main())
