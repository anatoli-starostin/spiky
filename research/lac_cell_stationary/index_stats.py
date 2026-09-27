"""Index statistics of a real trained LUT layer at B=24,576, top-1.

Two things live here, both computed from the artifact extract_real.py saved (real
indices of blocks.0.ffn.lut_light in exp_n_0196, 4 heads x 256 tables x 256 rows):

STEP 4 of the kernel task -- the amortisation headroom for a future bucketed kernel:
    distinct rows touched per table per token tile, at tile sizes 128 / 256 / 512,
    plus the per-table participation ratio and its histogram.

THE DECIDING MEASUREMENT for the intra-thread multi-table arm:
    a thread owning row r across G tables emits one atomic group if ANY of its G
    tables selects row r for the token, and its dedup factor is therefore
        dedup(G) = (hits) / (emitting (token, row) pairs)
    which under independent uniform indices is the birthday bound
        (G/R) / (1 - (1 - 1/R)^G).
    The empirical value is computed from the observed joint index distribution, so it
    answers whether different tables are hot on the SAME row index (dedup above the
    bound) or effectively independent (dedup at the bound). Cross-table alignment is
    also reported directly: the row-usage histogram of each table, the mean pairwise
    cosine similarity between those histograms, and how much mass sits on rows 0 and
    255 (the signature of saturating comparison patterns).
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, 'artifacts')
TILES = (128, 256, 512)
GS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def birthday(G, R):
    return (G / R) / (1.0 - (1.0 - 1.0 / R) ** G)


def row_hist(j, R):
    """[B, NT] uint8 -> [NT, R] fp32 counts."""
    B, NT = j.shape
    h = torch.zeros(NT, R, device=j.device)
    jt = j.t().long().contiguous()
    h.scatter_add_(1, jt, torch.ones_like(jt, dtype=torch.float))
    return h


def main():
    a = torch.load(os.path.join(ART, 'real_layer.pt'), map_location='cuda')
    R, tph, nh, B = a['table_size'], a['tables_per_head'], a['n_heads'], a['B']
    print(f'{a["run"]} / {a["layer"]}')
    print(f'B={B} tokens, {nh} heads x {tph} tables x {R} rows, output {a["output_dim"]}\n')
    out = {'run': a['run'], 'layer': a['layer'], 'B': B, 'R': R,
           'tables_per_head': tph, 'n_heads': nh}

    j_all = a['j'].cuda()

    # ---------------- per-head utilisation ---------------------------------------
    print('PER-TABLE ROW UTILISATION (participation ratio 1/sum(p^2), of '
          f'{R} rows; entropy in nats, uniform max {math.log(R):.3f})')
    print(f'{"head":>5}{"PR mean":>10}{"PR min":>9}{"PR max":>9}{"entropy":>10}'
          f'{"dead rows":>11}{"top-1 share":>13}{"row0+row255":>13}')
    heads = []
    for h in range(nh):
        jh = j_all[:, h * tph:(h + 1) * tph]
        cnt = row_hist(jh, R)
        p = cnt / cnt.sum(1, keepdim=True)
        pr = 1.0 / (p * p).sum(1)
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(1)
        dead = (cnt == 0).float().mean()
        top1 = p.max(1).values.mean()
        edge = (p[:, 0] + p[:, R - 1]).mean()
        print(f'{h:>5}{pr.mean():>10.1f}{pr.min():>9.1f}{pr.max():>9.1f}'
              f'{ent.mean():>10.3f}{dead:>11.4f}{top1:>13.4f}{edge:>13.4f}')
        heads.append({'head': h, 'pr_mean': pr.mean().item(), 'pr_min': pr.min().item(),
                      'pr_max': pr.max().item(), 'entropy_mean': ent.mean().item(),
                      'dead_frac': dead.item(), 'top1_share': top1.mean().item(),
                      'edge_share': edge.item()})
    out['heads'] = heads

    # ---------------- head 0 in detail -------------------------------------------
    j = j_all[:, :tph]
    cnt = row_hist(j, R)
    p = cnt / cnt.sum(1, keepdim=True)

    print('\nCROSS-TABLE ALIGNMENT OF HOT ROWS (head 0)')
    am = cnt.argmax(1)
    amh = torch.bincount(am, minlength=R).float()
    print(f'  argmax row over the {tph} tables: {int((amh > 0).sum())} distinct values, '
          f'most common row {int(amh.argmax())} used by {int(amh.max())} tables '
          f'({100*amh.max()/tph:.1f}%)')
    print(f'  argmax row == 0: {int(amh[0])} tables;  == {R-1}: {int(amh[R-1])} tables')
    pn = p / p.norm(dim=1, keepdim=True)
    cos = pn @ pn.t()
    off = cos[~torch.eye(tph, dtype=torch.bool, device=cos.device)]
    uni = torch.full((R,), 1.0 / R, device=cos.device)
    uni = uni / uni.norm()
    print(f'  mean pairwise cosine between row-usage histograms: {off.mean():.4f}  '
          f'(max {off.max():.4f}, min {off.min():.4f})')
    print(f'  cosine of a single table against the uniform histogram: '
          f'{(pn @ uni).mean():.4f}')
    print('  reading: histograms that are all near-uniform are automatically near-cosine-1,')
    print('  so this number alone cannot show alignment -- the dedup table below is what does.')
    out['alignment'] = {'argmax_distinct': int((amh > 0).sum()),
                        'argmax_mode_row': int(amh.argmax()),
                        'argmax_mode_count': int(amh.max()),
                        'argmax_row0': int(amh[0]), 'argmax_rowlast': int(amh[R - 1]),
                        'mean_pairwise_cosine': off.mean().item(),
                        'cosine_vs_uniform': (pn @ uni).mean().item()}

    # ---------------- the deciding measurement: empirical dedup vs G --------------
    print('\nINTRA-THREAD DEDUP: a thread owning row r across G tables')
    print(f'{"G":>5}{"empirical":>12}{"birthday":>11}{"ratio":>8}{"atomics/token":>15}'
          f'{"vs G=1":>9}')
    ded = []
    for G in GS:
        if tph % G:
            continue
        tot_hits, tot_emit = 0, 0
        for g0 in range(0, tph, G):
            oh = torch.zeros(B, R, device=j.device)
            for t in range(g0, g0 + G):
                oh.scatter_add_(1, j[:, t].long().unsqueeze(1),
                                torch.ones(B, 1, device=j.device))
            tot_hits += B * G
            tot_emit += int((oh > 0).sum())
        emp = tot_hits / tot_emit
        bd = birthday(G, R)
        per_tok = tot_emit / B
        print(f'{G:>5}{emp:>12.4f}{bd:>11.4f}{emp/bd:>8.3f}{per_tok:>15.1f}'
              f'{tph/per_tok:>9.3f}')
        ded.append({'G': G, 'empirical_dedup': emp, 'birthday_dedup': bd,
                    'ratio': emp / bd, 'atomic_groups_per_token': per_tok,
                    'reduction_vs_G1': tph / per_tok})
    out['dedup'] = ded

    # ---------------- step 4: distinct rows per table per token tile --------------
    print('\nDISTINCT ROWS TOUCHED PER TABLE PER TOKEN TILE (head 0; the amortisation')
    print('headroom for a bucketed / accumulator-stationary kernel)')
    print(f'{"tile":>6}{"distinct mean":>15}{"of tile":>9}{"of R":>8}{"min":>7}{"max":>7}'
          f'{"reads vs TS":>13}')
    tl = []
    for S in TILES:
        ntile = B // S
        ds = []
        for i in range(ntile):
            blk = j[i * S:(i + 1) * S].long()                     # [S, tph]
            oh = torch.zeros(tph, R, device=j.device)
            oh.scatter_add_(1, blk.t().contiguous(),
                            torch.ones(tph, S, device=j.device))
            ds.append((oh > 0).float().sum(1))                    # [tph]
        d = torch.stack(ds)                                       # [ntile, tph]
        print(f'{S:>6}{d.mean():>15.2f}{d.mean()/S:>9.3f}{d.mean()/R:>8.3f}'
              f'{d.min():>7.0f}{d.max():>7.0f}{S/d.mean():>13.2f}')
        tl.append({'tile': S, 'distinct_mean': d.mean().item(),
                   'frac_of_tile': (d.mean() / S).item(),
                   'frac_of_R': (d.mean() / R).item(),
                   'min': d.min().item(), 'max': d.max().item(),
                   'read_reduction_vs_TS': (S / d.mean()).item()})
    out['tiles'] = tl
    print('\n"reads vs TS" is the factor by which an accumulator-stationary sweep over')
    print('the distinct rows of a tile reads less than one row per table per token.')
    print(f'The paper\'s bound for this is min(tile, R)/1 = {min(TILES[-1], R)} at tile '
          f'{TILES[-1]}, i.e. {TILES[-1]/min(TILES[-1], R):.2f}x; the measured value is above')
    print('it only if rows go unused.')

    p_ = os.path.join(ART, 'index_stats.json')
    json.dump(out, open(p_, 'w'), indent=1)
    torch.save({'row_hist_head0': cnt.cpu()}, os.path.join(ART, 'row_hist.pt'))
    print(f'\nwrote {p_} and row_hist.pt')


if __name__ == '__main__':
    main()
