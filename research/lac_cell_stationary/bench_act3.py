"""Act-3-only gather benchmark: the p2_int8 kernel's accumulation, in isolation.

VEHICLE  `pow2_int8.read_cells` -- the PROLOGUE=false instantiation of the same kernel
         template the fused read uses, so Acts 1 and 2 (Z staging, anchor margins,
         table_scalars) are compiled out and only the gather remains. Wide-row ladder:
         D = 1024 -> UPR = 64, block_n = 16, 1024 threads per block.
SHAPE    H=1, T=256 tables, K=256 rows/table (nap=8), D=1024 lanes.
         Tables int8 [65536, 1024] = 64.0 MiB. Output fp32 [N, 1, 1024].
CELLS    Built offline by cells_producer (gated bit-exact against the fused kernel's own
         CELLS_OUT, 0/2,097,152 differing). Two distributions:
           real     the margins of exp_n_0196's blocks.0.ffn.lut_light head 0 on 24,576
                    real tokens, through the p2 quantiser at the preset's scalars
           uniform  random z and random anchors at din=256 -- a uniform c1 distribution
ARMS     load16=True  (branchless: both cells fetched every table, DISCARD only masks)
         load16=False (branch: `continue` on DISCARD, so discarded cells are not fetched)
TIMING   median of >=50 iterations, warmup discarded, CUDA events with the sync outside
         the window. B <= 256 is timed through a captured CUDA graph so the 3-5 us launch
         overhead does not dominate. HOT = tensors left resident; COLD = a 512 MiB copy
         issued immediately before start.record(), outside the measured window.
TRAFFIC  Table bytes are derived from the MEASURED discard counts of each cells tensor,
         not assumed: load16=True always moves 2*T*D*N; load16=False moves
         (frac(sh1 != DISCARD) + frac(sh2 != DISCARD)) * T*D*N.
"""
import json
import os
import statistics
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spiky.lutorch import pow2_int8          # noqa: E402
import cells_producer                        # noqa: E402

H, T, NAP, K, D = 1, 256, 8, 256, 1024
LO, HI, Q = -3, 4, 3
TAU, G, BETA, GAMMA = 0.1, 0.0, 2.0, 1.0     # the p2_int8 preset's learned_margin init
BLOCK_N = 16
BATCHES = [1, 64, 256, 1024, 4096, 24576]
GRAPH_MAX_B = 256
FLUSH_MIB = 512
HBM_PEAK_GBS = 1792.0
L2_MIB = 96.0
FP32_BASELINE_MS = {24576: 18.8061}          # embedding_bag LightMHL, same shape, hot


def make_flusher():
    n = FLUSH_MIB * 2 ** 20 // 4
    a = torch.empty(n, device='cuda', dtype=torch.float32).fill_(1.0)
    b = torch.empty(n, device='cuda', dtype=torch.float32)
    return lambda: b.copy_(a)


def discard_stats(cells):
    sh1 = cells[..., 2] & 15
    sh2 = cells[..., 2] >> 4
    tot = sh1.numel()
    skip = float((sh1 == 15).sum()) / tot                 # whole table discarded
    drop = float(((sh2 == 15) & (sh1 != 15)).sum()) / tot  # second cell only
    return {'frac_table_discarded': skip, 'frac_second_cell_dropped': drop,
            'frac_c1_fetched': 1.0 - skip,
            'frac_c2_fetched': float((sh2 != 15).sum()) / tot}


def real_cells(N):
    a = torch.load(os.path.join(HERE, 'artifacts', 'real_margins.pt'), map_location='cuda')
    z, aa, ab = a['z'], a['anchor_a'], a['anchor_b']
    reps = (N + z.shape[0] - 1) // z.shape[0]
    z = z.repeat(reps, 1, 1)[:N].contiguous()
    return cells_producer.cells(z, aa, ab, TAU, G, BETA, GAMMA, LO, HI, Q), z, aa, ab


def uniform_cells(N, din=256, seed=11):
    g = torch.Generator(device='cuda').manual_seed(seed)
    z = torch.randn(N, H, din, device='cuda', generator=g)
    aa = torch.randint(0, din, (H, T, NAP), device='cuda', dtype=torch.int64, generator=g)
    ab = torch.randint(0, din, (H, T, NAP), device='cuda', dtype=torch.int64, generator=g)
    return cells_producer.cells(z, aa, ab, TAU, G, BETA, GAMMA, LO, HI, Q), z, aa, ab


def timeit(fn, iters=50, warmup=8, flush=None, graph=None):
    run = (lambda: graph.replay()) if graph is not None else fn
    for _ in range(warmup):
        if flush:
            flush()
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if flush:
            flush()                       # before start.record(): not measured
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def capture(fn):
    """Capture one kernel launch into a CUDA graph; None if capture is not possible."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g
    except Exception as ex:
        print(f'      (cuda graph capture failed: {type(ex).__name__}: '
              f'{str(ex).splitlines()[0][:90]})')
        return None


@torch.no_grad()
def gate(tab, dist):
    """Both arms against each other and against the fused reference, before any timing."""
    print(f'\n-- correctness gate, {dist} --')
    Ng = 512
    cells, z, aa, ab = (real_cells(Ng) if dist == 'real' else uniform_cells(Ng))
    scal = tuple(torch.as_tensor([v], device='cuda', dtype=torch.float32)
                 for v in (TAU, G, BETA, GAMMA))
    fused = pow2_int8.read_fused(z, aa.to(torch.int32).contiguous(),
                                 ab.to(torch.int32).contiguous(), tab, scal,
                                 NAP, D, LO, HI, Q, block_n=BLOCK_N)
    a_t = pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=True)
    a_f = pow2_int8.read_cells(tab, cells, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=False)
    e_tf = (a_t - a_f).abs().max().item()
    e_tr = (a_t - fused).abs().max().item()
    e_fr = (a_f - fused).abs().max().item()
    print(f'   load16 True vs False   max|diff| {e_tf:.1f}')
    print(f'   load16 True vs fused   max|diff| {e_tr:.1f}')
    print(f'   load16 False vs fused  max|diff| {e_fr:.1f}')
    ok = (e_tf == 0 and e_tr == 0 and e_fr == 0)
    print(f'   {"BIT-EXACT, proceeding" if ok else "MISMATCH -- stopping"}')
    return ok


@torch.no_grad()
def main():
    print(f'pow2_int8 extension: {pow2_int8.available()[1]}')
    g = torch.Generator(device='cuda').manual_seed(5)
    packed = torch.randint(-127, 128, (H * T * K, D), device='cuda', dtype=torch.int8,
                           generator=g)
    tab = pow2_int8.stride_tables(packed, D)
    tbl_bytes = tab.numel()
    print(f'tables {tuple(tab.shape)} int8 = {tbl_bytes/2**20:.1f} MiB; '
          f'L2 = {L2_MIB:.0f} MB -> {"FITS" if tbl_bytes/2**20 <= L2_MIB else "does not fit"}')
    print(f'block_n={BLOCK_N}, UPR={D//16}, threads/block={BLOCK_N*D//16}')

    for dist in ('real', 'uniform'):
        if not gate(tab, dist):
            return 1

    flush = make_flusher()
    out = {'shape': dict(H=H, T=T, K=K, D=D, block_n=BLOCK_N),
           'table_bytes': tbl_bytes, 'rows': []}

    for dist in ('real', 'uniform'):
        cells_full, _, _, _ = (real_cells(max(BATCHES)) if dist == 'real'
                               else uniform_cells(max(BATCHES)))
        st = discard_stats(cells_full)
        print(f'\n{"="*104}\ncells distribution: {dist}')
        print(f'  tables fully discarded      {100*st["frac_table_discarded"]:.3f}%')
        print(f'  second cell dropped only    {100*st["frac_second_cell_dropped"]:.3f}%')
        print(f'  cells actually fetched by load16=False: c1 {100*st["frac_c1_fetched"]:.3f}%, '
              f'c2 {100*st["frac_c2_fetched"]:.3f}%  -> '
              f'{st["frac_c1_fetched"]+st["frac_c2_fetched"]:.4f} of 2 per table')
        print(f'{"="*104}')
        hdr = (f'{"B":>7}{"arm":>8}{"graph":>7}'
               f'{"hot ms":>10}{"tok/s":>11}{"GB/s":>8}{"%peak":>7}'
               f'{"cold ms":>10}{"tok/s":>11}{"GB/s":>8}{"%peak":>7}{"h/c":>7}')
        print(hdr)
        for B in BATCHES:
            cells = cells_full[:B].contiguous()
            for load16 in (True, False):
                frac = 2.0 if load16 else (st['frac_c1_fetched'] + st['frac_c2_fetched'])
                tbytes = frac * T * D * B
                fn = (lambda l=load16, c=cells: pow2_int8.read_cells(
                    tab, c, NAP, D, LO, HI, Q, block_n=BLOCK_N, load16=l))
                gr = capture(fn) if B <= GRAPH_MAX_B else None
                hot = timeit(fn, graph=gr)
                cold = timeit(fn, flush=flush, graph=gr)
                r = {'dist': dist, 'B': B, 'load16': load16, 'graph': gr is not None,
                     'hot_ms': hot, 'cold_ms': cold, 'table_bytes': tbytes,
                     'cells_bytes': B * T * 3, 'out_bytes': B * D * 4,
                     'frac_cells_fetched': frac, **st}
                out['rows'].append(r)
                hg = tbytes / (hot * 1e-3) / 1e9
                cg = tbytes / (cold * 1e-3) / 1e9
                print(f'{B:>7}{("T" if load16 else "F"):>8}{("y" if gr else "-"):>7}'
                      f'{hot:>10.4f}{B/(hot*1e-3):>11.3e}{hg:>8.0f}{100*hg/HBM_PEAK_GBS:>7.0f}'
                      f'{cold:>10.4f}{B/(cold*1e-3):>11.3e}{cg:>8.0f}'
                      f'{100*cg/HBM_PEAK_GBS:>7.0f}{cold/hot:>7.2f}')
                del gr
            del cells
            torch.cuda.empty_cache()
        del cells_full
        torch.cuda.empty_cache()

    p = os.path.join(HERE, 'artifacts', 'act3.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(f'\ncompulsory table traffic for ANY batch: {tbl_bytes/2**20:.1f} MiB '
          f'(the whole table set, once) -- any GB/s above '
          f'{HBM_PEAK_GBS:.0f} is L2 supply, not HBM')
    if 24576 in FP32_BASELINE_MS:
        print(f'reference: fp32 embedding_bag LightMHL, same shape, B=24576 hot = '
              f'{FP32_BASELINE_MS[24576]:.4f} ms')
    print('wrote', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
