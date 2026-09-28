"""The shipped int8 (p2_int8) LightMHL read, same methodology as bench_shipped_light.py.

THE REQUESTED CONFIG CANNOT RUN, for two independent reasons, both in shipped code:

 1. Module constraints (light_multi_head_lut.py:172-190): quant_mode is refused unless
    confidence_form='learned_margin', learned_margin_freeze_g=True, read_top_n=2,
    forward_mode='scored', cell_mode='constant', multi_head_input=True, anchor_mode='pair',
    output_heads=1. The fp32 run used margin / top-1 / multi_head_input=False: three
    violations.
 2. Kernel constraints (csrc/pow2_int8_read.cu:54, 253, 264): UPR_MAX=8 caps the cell
    width at D <= 128, and the anchor column index is staged as one byte so din <= 256.
    The paper's 1024-wide LUT Core violates both.

So this measures the two things the shipped code CAN do, both labelled:

 ARM A -- the fused CUDA int8 kernel at a geometry it supports: din = D = 48, T = 256,
    nap = 8, one head. That is the repo's real FFN-slot width, and it is the only way the
    fused kernel runs at all. One launch does the per-table integers and the int8
    shift-add together (pow2_int8.read_fused).
 ARM B -- the torch integer read (pow2_read.int_blend_read, what
    LightMultiHeadLUT.forward_int calls and what QuantisedLightFFN falls back to without
    the extension) at the REQUESTED width, din = D = 1024. It has no D limit. It is not
    the fast path and is not what would run in deployment at a supported width.

Both arms are two-cell reads, which is inherent to the power-of-two form: int8 is 4x
smaller per row but two rows are read per table, so the traffic saving over an fp32 top-1
read is 2x, not 4x.

Methodology identical to the fp32 run: median of 50 iterations (20 at B=24576), CUDA
events with the sync outside the window, warmup 5, no CUDA graphs, every buffer allocated
and filled outside the timed region, and a 512 MiB copy issued immediately before
start.record() for the cold rows.
"""
import json
import os
import statistics
import sys

import torch

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
from spiky.lutorch import pow2_int8, pow2_read                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCHES = [1, 8, 64, 256, 1024, 4096, 24576]
BASE = dict(n_tables=256, n_anchor_pairs=8, n_heads=1, multi_head_input=True,
            read_top_n=2, confidence_form='learned_margin',
            learned_margin_freeze_g=True, forward_mode='scored',
            quant_mode='p2_int8', initial_weights_noise=0.001, random_seed=1234)
FLUSH_MIB, HBM_PEAK_GBS, L2_MIB = 512, 1792.0, 96.0


def make_flusher():
    n = FLUSH_MIB * 2 ** 20 // 4
    a = torch.empty(n, device='cuda', dtype=torch.float32).fill_(1.0)
    b = torch.empty(n, device='cuda', dtype=torch.float32)
    return lambda: b.copy_(a)


def timeit(fn, iters, warmup=5, flush=None):
    for _ in range(warmup):
        if flush:
            flush()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if flush:
            flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def build(dim):
    lut = LightMultiHeadLUT(input_dim=dim, output_dim=dim, **BASE,
                            device=torch.device('cuda')).cuda().eval()
    return lut


@torch.no_grad()
def prepared(lut, B):
    """Margins, cell addresses and per-table integers, all outside any timed region."""
    H, T, NAP = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
    z = torch.randn(B, H, lut.input_dim, device='cuda')
    ia = lut.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    ib = lut.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    d = (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(B, H, T, NAP)
    index = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, 1, -1)).sum(-1)
    tau, g, beta, gamma = lut._quant_scalars()
    idx, q, k, skip, drop = pow2_int8.table_integers(
        d, index, lut.powers, tau, g, beta, gamma, lut._quant)
    return z, d, index, idx, q, k, skip, drop


@torch.no_grad()
def arm_a(flush):
    lut = build(48)
    cfg = lut._quant
    D, H, T, NAP = lut.output_dim, lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
    _, packed = lut.quantised_tables()
    tab = pow2_int8.stride_tables(packed, D)
    a32 = lut.anchor_a.to(torch.int32).contiguous()
    b32 = lut.anchor_b.to(torch.int32).contiguous()
    tau, g, beta, gamma = lut._quant_scalars()
    scal = tuple(torch.as_tensor([float(v)], device='cuda', dtype=torch.float32)
                 for v in (tau, g, beta, gamma))
    lo, hi, Q = cfg['lo'], cfg['hi'], cfg['Q']
    per_tok = 2 * T * D
    print(f'\nARM A -- fused CUDA int8 kernel, din=D={D}, T={T}, nap={NAP}, heads={H}')
    print(f'  packed int8 tables {tuple(tab.shape)} = {tab.numel()/2**20:.2f} MiB '
          f'(float master {lut.tables.numel()*4/2**20:.2f} MiB)')
    print(f'  per-token read 2 x {T} x {D} B = {per_tok/1024:.0f} KiB')
    print(f'{"B":>7}{"total hot":>11}{"gather hot":>12}{"tok/s":>11}{"GB/s":>7}'
          f'{"total cold":>12}{"gather cold":>13}{"tok/s":>11}{"GB/s":>7}')
    rows = []
    for B in BATCHES:
        z, d, index, idx, q, k, skip, drop = prepared(lut, B)
        cells = pow2_int8.pack_cells(idx, q, k, skip, drop)
        f_tot = lambda: pow2_int8.read_fused(z, a32, b32, tab, scal, NAP, D, lo, hi, Q)
        f_gat = lambda: pow2_int8.read_cells(tab, cells, NAP, D, lo, hi, Q)
        it = 50 if B <= 4096 else 20
        r = {'B': B, 'read_bytes': B * per_tok}
        line = f'{B:>7}'
        for tag, fl in (('hot', None), ('cold', flush)):
            r[f'total_{tag}'] = timeit(f_tot, it, flush=fl)
            r[f'gather_{tag}'] = timeit(f_gat, it, flush=fl)
            line += (f'{r[f"total_{tag}"]:>11.4f}{r[f"gather_{tag}"]:>12.4f}'
                     f'{B/(r[f"total_{tag}"]*1e-3):>11.3e}'
                     f'{r["read_bytes"]/(r[f"gather_{tag}"]*1e-3)/1e9:>7.0f}')
        print(line)
        rows.append(r)
        del z, d, index, idx, q, k, skip, drop, cells
        torch.cuda.empty_cache()
    # numerical check at this geometry, where both reads exist
    B = 1024
    z = torch.randn(B, H, lut.input_dim, device='cuda')
    yf = lut(z).float()
    yi = lut.forward_int(z).float()
    da, sc = (yi - yf).abs(), yf.abs()
    num = {'max_abs': da.max().item(), 'max_rel_vs_ymax': da.max().item() / sc.max().item(),
           'y_absmax': sc.max().item(), 'y_absmean': sc.mean().item()}
    print(f'  int8 forward_int vs the float quant forward, same weights, B={B}: '
          f'max abs {num["max_abs"]:.3e}, max rel vs |y|max {num["max_rel_vs_ymax"]:.2e} '
          f'(|y|max {num["y_absmax"]:.4f})')
    return rows, num, tab.numel(), per_tok


@torch.no_grad()
def arm_b(flush, batches):
    lut = build(1024)
    cfg = lut._quant
    D, H, T = lut.output_dim, lut.n_heads, lut.tables_per_head
    _, packed = lut.quantised_tables()
    per_tok = 2 * T * D
    print(f'\nARM B -- torch integer read (pow2_read.int_blend_read), din=D={D}, T={T}')
    print(f'  packed int8 tables {tuple(packed.shape)} = {packed.numel()/2**20:.1f} MiB '
          f'(float master {lut.tables.numel()*4/2**20:.1f} MiB) -> '
          f'{"FITS" if packed.numel()/2**20 <= L2_MIB else "does not fit"} the '
          f'{L2_MIB:.0f} MB L2; the float master does NOT')
    print(f'  per-token read 2 x {T} x {D} B = {per_tok/1024:.0f} KiB '
          f'(the fp32 top-1 run read {T*D*4/1024:.0f} KiB -> {T*D*4/per_tok:.1f}x less)')
    print(f'{"B":>7}{"gather hot":>12}{"tok/s":>11}{"GB/s":>7}{"gather cold":>13}'
          f'{"tok/s":>11}{"GB/s":>7}')
    rows = []
    for B in batches:
        z, d, index, idx, q, k, skip, drop = prepared(lut, B)
        flat_idx = idx + lut.table_offset.view(1, H, T, 1)
        f = lambda: pow2_read.int_blend_read(packed, cfg['bits'], D, flat_idx,
                                             q, k, skip, drop)
        it = 50 if B <= 1024 else 20
        r = {'B': B, 'read_bytes': B * per_tok}
        line = f'{B:>7}'
        for tag, fl in (('hot', None), ('cold', flush)):
            r[f'gather_{tag}'] = timeit(f, it, flush=fl)
            line += (f'{r[f"gather_{tag}"]:>12.4f}{B/(r[f"gather_{tag}"]*1e-3):>11.3e}'
                     f'{r["read_bytes"]/(r[f"gather_{tag}"]*1e-3)/1e9:>7.0f}')
        print(line)
        rows.append(r)
        del z, d, index, idx, q, k, skip, drop, flat_idx
        torch.cuda.empty_cache()
    return rows, packed.numel(), per_tok


def main():
    print(f'pow2_int8 extension: {pow2_int8.ensure_registered()} / {pow2_int8.available()}')
    flush = make_flusher()
    a_rows, num, a_bytes, a_pertok = arm_a(flush)
    # B=4096 OOMs: int8_accumulate materialises the gathered rows as int32, and
    # int8_blend_read's chunking reads flat_idx.shape[1] as the cells-per-bag, which is H
    # (=1) for the 4-D [N,H,T,2] tensor forward_int passes -- so nothing is chunked and it
    # asks for an 8 GiB allocation. A shipped limitation of the torch read, reported as one.
    b_batches = [b for b in BATCHES if b <= 1024]
    b_rows, b_bytes, b_pertok = arm_b(flush, b_batches)
    p = os.path.join(HERE, 'artifacts', 'shipped_int8.json')
    json.dump({'base_config': BASE, 'arm_a': {'dim': 48, 'rows': a_rows,
                                              'table_bytes': a_bytes,
                                              'per_token_bytes': a_pertok, 'numeric': num},
               'arm_b': {'dim': 1024, 'rows': b_rows, 'table_bytes': b_bytes,
                         'per_token_bytes': b_pertok}},
              open(p, 'w'), indent=1)
    print('\nwrote', p)


if __name__ == '__main__':
    main()
