"""Latency of the SHIPPED LightMultiHeadLUT inference path, at the paper's reference
LUT-Core geometry, hot and cold cache.

This measures the repo's own code as-is -- `spiky.lutorch.light_multi_head_lut`, the
module the `lut_impl=light` experiments instantiate -- not the generic kernel written in
this directory. One module, random init, no checkpoint, no real data.

STAGE SPLIT. The eval path is `LightMultiHeadLUT._fused_eval` (light_multi_head_lut.py,
the `torch.is_grad_enabled() == False and read_top_n == 1 and forward_mode == "scored"`
branch of `_forward_impl`). It has exactly two parts:

  (i)  index + score: one call into the native bit-pack kernel `_native_msb_scored`,
       which reads the anchor pairs out of the 1024-wide input, packs the sign bits into
       an integer address per table, and accumulates the margin score in the same loop.
  (ii) gather + accumulate: build the flat address (index + per-table offset) and call
       `F.embedding_bag(..., mode="sum", per_sample_weights=score)`.

The two are timed SEPARATELY IN THE SAME PROCESS, each with its own inputs already
materialised, and the sum is checked against the timed whole-forward. That is the method;
nothing is subtracted.

CACHE STATE. Every iteration is timed twice over:
  hot  -- tight repeat loop, whatever the previous iteration left in L2 stays there.
  cold -- a 512 MiB scratch buffer is written and read immediately BEFORE the start
          event of each iteration, which evicts the 96 MB L2. The flush is outside the
          measured window (it precedes `start.record()`), and so is every buffer
          allocation and zeroing.

No CUDA graphs; plain stream launches, so ~3-5 us of launch overhead sits inside every
number and dominates the small-B rows.
"""
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

SPIKY = os.path.expanduser('~/projects/spiky')
sys.path.insert(0, os.path.join(SPIKY, 'src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCHES = [1, 8, 64, 256, 1024, 4096, 24576]
CFG = dict(input_dim=1024, n_tables=256, output_dim=1024, n_anchor_pairs=8,
           n_heads=1, multi_head_input=False, read_top_n=1,
           confidence_form='margin', forward_mode='scored',
           initial_weights_noise=0.001, random_seed=1234)
FLUSH_MIB = 512
HBM_PEAK_GBS = 1792.0
L2_MIB = 96.0


def make_flusher():
    n = FLUSH_MIB * 2 ** 20 // 4
    a = torch.empty(n, device='cuda', dtype=torch.float32)
    b = torch.empty(n, device='cuda', dtype=torch.float32)
    a.fill_(1.0)

    def flush():
        b.copy_(a)          # 512 MiB read + 512 MiB write: evicts a 96 MB L2
    return flush


def timeit(fn, iters, warmup=5, flush=None):
    for _ in range(warmup):
        if flush:
            flush()
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if flush:
            flush()                       # before the start event: not measured
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    lut = LightMultiHeadLUT(**CFG, device=torch.device('cuda')).cuda().eval()
    tb = lut.tables
    print(f'module   {type(lut).__module__}.{type(lut).__name__}')
    print(f'config   {CFG}')
    print(f'tables   shape {tuple(tb.shape)}  dtype {tb.dtype}  '
          f'{tb.numel():,} values = {tb.numel()*tb.element_size()/2**20:.1f} MiB')
    print(f'         (the same value count as the white paper\'s 64 MiB reference LUT '
          f'Core, which stores w=8; this path stores {8*tb.element_size()}-bit)')
    print(f'anchors  a {tuple(lut.native_anchor_a.shape)} b '
          f'{tuple(lut.native_anchor_b.shape)} dtype {lut.native_anchor_a.dtype}')
    tot = sum(p.numel() for p in lut.parameters())
    print(f'params   {tot:,} ({tot*4/2**20:.1f} MiB fp32)')
    print(f'per-token gather read: n_t * n * {tb.element_size()} B = '
          f'{256*1024*tb.element_size()/1024:.0f} KiB')
    print(f'L2 is {L2_MIB:.0f} MB, the table set is '
          f'{tb.numel()*tb.element_size()/2**20:.0f} MiB -> '
          f'{"fits" if tb.numel()*tb.element_size()/2**20 <= L2_MIB else "DOES NOT FIT"}\n')

    flush = make_flusher()
    rows = []
    elem = tb.element_size()

    for B in BATCHES:
        x = torch.randn(B, CFG['input_dim'], device='cuda')
        # everything below is materialised OUTSIDE any timed region
        out = lut(x)
        assert out.shape == (B, CFG['output_dim']), out.shape
        fused = lut._fused_eval(x.contiguous())
        assert fused is not None, 'the shipped fused eval path did not engage'
        index, score = lut._native_msb_scored(
            x.contiguous(), lut.native_anchor_a, lut.native_anchor_b, 0.0,
            lut._score_form_id, lut.confidence_gain, 256)
        flat = tb.reshape(lut.n_tables * lut.table_size, lut.output_dim)
        n_bags = B * lut.n_heads

        def f_total():
            lut(x)

        def f_index():
            lut._native_msb_scored(x.contiguous(), lut.native_anchor_a,
                                   lut.native_anchor_b, 0.0, lut._score_form_id,
                                   lut.confidence_gain, 256)

        def f_gather():
            fi = (index + lut.table_offset.view(1, -1)).reshape(-1)
            lut._bagged_sum(flat, fi, score, n_bags, lut.tables_per_head)

        it = 50 if B <= 4096 else 20
        r = {'B': B}
        for tag, fl in (('hot', None), ('cold', flush)):
            r[f'total_{tag}'] = timeit(f_total, it, flush=fl)
            r[f'index_{tag}'] = timeit(f_index, it, flush=fl)
            r[f'gather_{tag}'] = timeit(f_gather, it, flush=fl)
        r['read_bytes'] = B * lut.n_tables * lut.output_dim * elem
        rows.append(r)
        del x, out, fused
        torch.cuda.empty_cache()

    print('SHIPPED LightMHL, n_heads=1 tph=256 nap=8 d_in=d_out=1024, top-1, '
          'margin/scored, fp32 tables')
    print(f'{"":>7}{"---------------- HOT L2 ----------------":>44}'
          f'{"--------------- COLD L2 ----------------":>44}')
    print(f'{"B":>7}{"total":>9}{"index":>9}{"gather":>9}{"tok/s":>10}{"GB/s":>8}'
          f'{"total":>9}{"index":>9}{"gather":>9}{"tok/s":>10}{"GB/s":>8}')
    for r in rows:
        line = f'{r["B"]:>7}'
        for tag in ('hot', 'cold'):
            tps = r['B'] / (r[f'total_{tag}'] * 1e-3)
            gbs = r['read_bytes'] / (r[f'gather_{tag}'] * 1e-3) / 1e9
            line += (f'{r[f"total_{tag}"]:>9.4f}{r[f"index_{tag}"]:>9.4f}'
                     f'{r[f"gather_{tag}"]:>9.4f}{tps:>10.3e}{gbs:>8.0f}')
        print(line)
    print('\nall times in ms (median); GB/s is the gather stage\'s requested table bytes '
          f'/ gather time; HBM peak {HBM_PEAK_GBS:.0f} GB/s')
    print(f'{"B":>7}{"i+g vs total, hot":>20}{"i+g vs total, cold":>21}'
          f'{"gather share, hot":>19}{"cold":>8}')
    for r in rows:
        for tag, w in (('hot', 20), ('cold', 21)):
            pass
        s_h = r['index_hot'] + r['gather_hot']
        s_c = r['index_cold'] + r['gather_cold']
        print(f'{r["B"]:>7}{s_h/r["total_hot"]:>20.3f}{s_c/r["total_cold"]:>21.3f}'
              f'{r["gather_hot"]/s_h:>19.3f}{r["gather_cold"]/s_c:>8.3f}')

    p = os.path.join(HERE, 'artifacts', 'shipped_light.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'config': CFG, 'flush_mib': FLUSH_MIB,
               'table_bytes': tb.numel() * elem, 'dtype': str(tb.dtype),
               'rows': rows}, open(p, 'w'), indent=1)
    print('\nwrote', p)


if __name__ == '__main__':
    main()
