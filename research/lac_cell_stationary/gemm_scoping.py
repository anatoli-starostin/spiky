"""U -- distinct cell values touched per (table, token-tile) -- and whether a compacted
tensor-core GEMM formulation can survive the arithmetic.

Measurement only: no kernel, no timings. Real cells from the exp_n_0196 forward over the
24,576 climbmix tokens, cross-checked against uniform-random cells.

U matters because a compacted GEMM's reduction dimension becomes U rather than K, so the
FLOP-waste multiplier against the useful work is U / 1.3272 (1.3272 = the measured cells
actually fetched per table at the real discard rate).
"""
import json
import math
import os
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_act3 import T, K, D, real_cells, uniform_cells, discard_stats  # noqa: E402

N = 24576
BNS = [32, 64, 128, 256]
FETCH = 1.3272
DISCARD = 15
# RTX 5090: 170 SMs x 128 unified INT32/FP32 lanes = 21,760; spec boost 2.41 GHz, observed
# under these workloads ~3.0 GHz (nvidia-smi clocks.max.sm reports 3,210 MHz).
SMS, LANES, CLK_SPEC, CLK_OBS = 170, 128, 2.41e9, 3.00e9
AS_MS = 1.4187
LANE_UPDATES = 8.550e9


def distinct_per_tile(c1, c2, live1, live2, Bn, include_discard):
    """[ntiles, T] int32 count of distinct cell values per (tile, table)."""
    ntiles = N // Bn
    outs = []
    step = max(1, 2 ** 22 // (T * K))          # chunk tiles to bound memory
    for t0 in range(0, ntiles, step):
        t1 = min(ntiles, t0 + step)
        sl = slice(t0 * Bn, t1 * Bn)
        nt = t1 - t0
        mask = torch.zeros(nt, T, K, dtype=torch.bool, device=c1.device)
        for c, live in ((c1, live1), (c2, live2)):
            v = c[sl].view(nt, Bn, T).permute(0, 2, 1).long()        # [nt, T, Bn]
            if include_discard:
                src = torch.ones_like(v, dtype=torch.bool)
            else:
                lv = live[sl].view(nt, Bn, T).permute(0, 2, 1)
                src = lv
                v = torch.where(lv, v, torch.zeros_like(v))
                # a dead entry is forced to cell 0 with src False, so scatter leaves it alone
            mask.scatter_(2, v, src, reduce='add') if False else mask.scatter_(
                2, v, src)
        outs.append(mask.sum(2).to(torch.int32))
    return torch.cat(outs, 0)


def describe(u, Bn, tag):
    f = u.float()
    q = torch.quantile(f.flatten().float(),
                       torch.tensor([0.5, 0.9, 0.99], device=u.device))
    return {'tag': tag, 'Bn': Bn, 'mean': f.mean().item(), 'median': q[0].item(),
            'p90': q[1].item(), 'p99': q[2].item(), 'max': f.max().item(),
            'min': f.min().item(), 'stdev': f.std().item()}


@torch.no_grad()
def main():
    print(f'CONCRETE SHAPE ON THE RECORD: K = {K} cells per table (2^nap, nap = 8), '
          f'T = {T} tables, D = {D} lanes, N = {N} tokens.')
    print(f'Useful work = N * T * {FETCH} * D = {N*T*FETCH*D:.3e} lane updates.\n')
    res = {'K': K, 'T': T, 'D': D, 'N': N, 'rows': []}

    for dist in ('real', 'uniform'):
        cells, _, _, _ = (real_cells(N) if dist == 'real' else uniform_cells(N))
        cells = cells.squeeze(1)                                  # [N, T, 3]
        st = discard_stats(cells.unsqueeze(1))
        c1, c2 = cells[..., 0], cells[..., 1]
        sh = cells[..., 2]
        live1, live2 = (sh & 15) != DISCARD, (sh >> 4) != DISCARD
        fetched = st['frac_c1_fetched'] + st['frac_c2_fetched']
        print(f'{"="*104}\n{dist} cells: {fetched:.4f} cells fetched per table '
              f'(tables fully discarded {100*st["frac_table_discarded"]:.2f}%, '
              f'second cell only {100*st["frac_second_cell_dropped"]:.2f}%)\n{"="*104}')
        print(f'{"Bn":>5}{"mean U":>9}{"median":>8}{"p90":>7}{"p99":>7}{"max":>6}'
              f'{"stdev":>8}{"U/K":>7}{"U/(Bn*1.33)":>13}{"waste mean":>12}{"waste p99":>11}')
        for Bn in BNS:
            u = distinct_per_tile(c1, c2, live1, live2, Bn, False)
            d = describe(u, Bn, dist)
            nore = Bn * FETCH
            d.update({'U_over_K': d['mean'] / K, 'U_over_noreuse': d['mean'] / nore,
                      'waste_mean': d['mean'] / FETCH, 'waste_p99': d['p99'] / FETCH})
            print(f'{Bn:>5}{d["mean"]:>9.2f}{d["median"]:>8.0f}{d["p90"]:>7.0f}'
                  f'{d["p99"]:>7.0f}{d["max"]:>6.0f}{d["stdev"]:>8.2f}'
                  f'{d["U_over_K"]:>7.3f}{d["U_over_noreuse"]:>13.3f}'
                  f'{d["waste_mean"]:>12.1f}{d["waste_p99"]:>11.1f}')
            res['rows'].append(d)
            if Bn == 128:
                per_table = u.float().mean(0)                     # [T]
                s, _ = per_table.sort()
                res[f'per_table_{dist}'] = {
                    'min5': s[:5].tolist(), 'max5': s[-5:].tolist(),
                    'mean': per_table.mean().item(), 'std': per_table.std().item(),
                    'ratio_max_min': (s[-1] / s[0]).item()}
            del u
            torch.cuda.empty_cache()

        # discards counted as if real, to separate "reuse" from "nothing to fetch"
        print(f'{"Bn":>5}{"U excl discard":>16}{"U incl discard":>16}{"ratio":>8}'
              f'{"  <- how much apparent reuse is just discards"}')
        for Bn in BNS:
            ue = distinct_per_tile(c1, c2, live1, live2, Bn, False).float().mean().item()
            ui = distinct_per_tile(c1, c2, live1, live2, Bn, True).float().mean().item()
            print(f'{Bn:>5}{ue:>16.2f}{ui:>16.2f}{ui/ue:>8.3f}')
            res['rows'].append({'tag': dist + '_discard', 'Bn': Bn,
                                'U_excl': ue, 'U_incl': ui})
            torch.cuda.empty_cache()

        pt = res.get(f'per_table_{dist}')
        print(f'\nper-table mean U at Bn=128: mean {pt["mean"]:.2f}, stdev {pt["std"]:.2f}, '
              f'max/min {pt["ratio_max_min"]:.3f}')
        print(f'  lowest 5 tables: {[round(x,1) for x in pt["min5"]]}')
        print(f'  highest 5 tables: {[round(x,1) for x in pt["max5"]]}')

        # whole-dataset concentration per table
        cnt = torch.zeros(T, K, device=cells.device)
        for c, live in ((c1, live1), (c2, live2)):
            v = torch.where(live, c.long(), torch.zeros_like(c, dtype=torch.long))
            cnt.scatter_add_(1, v.t().contiguous(),
                             live.t().to(torch.float).contiguous())
        p = cnt / cnt.sum(1, keepdim=True).clamp_min(1)
        pr = 1.0 / (p * p).sum(1)
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(1)
        top8 = p.topk(8, dim=1).values.sum(1)
        print(f'whole-dataset concentration per table: effective distinct values '
              f'(participation ratio) mean {pr.mean():.1f} of {K}, '
              f'entropy {ent.mean():.3f} nats of {math.log(K):.3f} max, '
              f'top-8 mass {100*top8.mean():.2f}%')
        res[f'concentration_{dist}'] = {'pr_mean': pr.mean().item(),
                                        'entropy_mean': ent.mean().item(),
                                        'top8_mass': top8.mean().item()}
        del cells, c1, c2, live1, live2, cnt
        torch.cuda.empty_cache()

    # ---- the scalar-ALU anchor ----
    cores = SMS * LANES
    print(f'\n{"="*104}\nSCALAR-ALU ANCHOR -- DERIVED FROM SPEC NUMBERS, NOT MEASURED')
    print(f'(ncu unavailable: RmProfilingAdminOnly=1)\n{"="*104}')
    print(f'{SMS} SMs x {LANES} unified INT32/FP32 lanes = {cores:,} lanes')
    for nm, clk in (('spec boost 2.41 GHz', CLK_SPEC), ('observed ~3.00 GHz', CLK_OBS)):
        ips = cores * clk
        print(f'\n  {nm}: {ips:.3e} integer ops/s')
        for ops, what in ((1, 'the ADD alone'), (4, 'extract + AND + shift + add'),
                          (5, 'the same with a separate sign-extend')):
            t = LANE_UPDATES * ops / ips * 1e3
            print(f'     {ops} op(s) per lane update ({what:<38}) -> floor {t:7.3f} ms, '
                  f'AS at {AS_MS} ms is {AS_MS/t:5.2f}x above it')
    res['anchor'] = {'cores': cores, 'as_ms': AS_MS, 'lane_updates': LANE_UPDATES}
    json.dump(res, open(os.path.join(HERE, 'artifacts', 'gemm_scoping.json'), 'w'), indent=1)
    print('\nwrote artifacts/gemm_scoping.json')


if __name__ == '__main__':
    main()
