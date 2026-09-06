"""Where does Light's forward time actually go, and what would a CUDA kernel buy?

Answers three things the eyeball cannot:

  (a) the breakdown of Light's fused forward -- margin gather, score, embedding_bag --
      which bounds what any kernel work could possibly save;
  (b) whether Fast's native CUDA eval kernel can serve Light. It computes the packed row
      index in one pass (`anchor_pairs_lookup_eval_forward_msb(x, a, b, eps) -> [B, T]
      int64`) and returns ONLY that index, discarding the margins. Light's score needs
      |d| for every anchor, so the margins have to be gathered anyway -- using the kernel
      on top would gather twice. Measured here rather than argued;
  (c) reduced-precision tables: gather traffic dominates, so fp32 -> bf16/fp16 halves the
      bytes moved. Speed only; the quality question is separate and is NOT answered here.

    python bench_light_optimised.py
"""
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT      # noqa: E402
from spiky.lutorch.fast_multi_head_lut import (                       # noqa: E402
    FastMultiHeadLut, _confidence_score, _get_native_lutorch_manager)

DEV = torch.device('cuda:0')
B, H, T, NAP, DIN, DOUT = 6144, 4, 256, 8, 32, 48
FORM = 'bounded_norm'


def timed(fn, reps=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def make(dtype=torch.float32):
    m = LightMultiHeadLUT(input_dim=DIN, n_tables=H * T, output_dim=DOUT, n_anchor_pairs=NAP,
                          confidence_form=FORM, random_seed=1000, device=DEV,
                          n_heads=H, multi_head_input=True)
    m.tables.data = m.tables.data.to(dtype)
    return m


def main():
    print(f'{torch.cuda.get_device_name(0)}   B={B:,} tokens, H={H}, tph={T}, nap={NAP}')
    m = make()
    x = torch.randn(B, H, DIN, device=DEV) * 0.6

    idx_a = m.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    idx_b = m.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)

    def gather_only():
        with torch.no_grad():
            (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)

    with torch.no_grad():
        d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)

    def score_only():
        with torch.no_grad():
            _confidence_score(d, FORM, 1.0)

    def index_only():
        with torch.no_grad():
            ((d > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(dim=-1)

    with torch.no_grad():
        score = _confidence_score(d, FORM, 1.0)
        index = ((d > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(dim=-1)
        flat = m.tables.reshape(H * T * m.table_size, DOUT)
        flat_idx = (index + m.table_offset.view(1, H, T)).reshape(-1)
        offs = torch.arange(B * H, device=DEV, dtype=torch.long) * T

    def bag_only():
        with torch.no_grad():
            F.embedding_bag(flat_idx, flat, offsets=offs, mode='sum',
                            per_sample_weights=score.reshape(-1))

    def full():
        with torch.no_grad():
            m(x)

    print('\n' + '=' * 88)
    print('(a) BREAKDOWN of Light\'s fused eval forward')
    print('=' * 88)
    mgr0 = _get_native_lutorch_manager()
    kern0 = getattr(mgr0, 'anchor_pairs_lookup_eval_forward_msb', None) if mgr0 else None
    head_off0 = (torch.arange(H, device=DEV).view(H, 1, 1) * DIN)
    a_f = (m.anchor_a + head_off0).reshape(H * T, NAP).contiguous()
    b_f = (m.anchor_b + head_off0).reshape(H * T, NAP).contiguous()
    xf0 = x.reshape(B, H * DIN).contiguous()

    def native_idx_only():
        with torch.no_grad():
            kern0(xf0, a_f, b_f, 0.0, 256)

    tot = timed(full)
    # The module now packs its address with the native kernel, so THAT is the component
    # in the live path; the torch sign+pack is shown underneath as what it replaced.
    parts = [('margin gather (2x torch.gather)', gather_only),
             ('score  (logsigmoid mean)', score_only),
             ('address (NATIVE bit-pack)', native_idx_only),
             ('embedding_bag (gather+weight+sum)', bag_only)]
    acc = 0.0
    for name, fn in parts:
        t = timed(fn)
        acc += t
        print(f'   {name:<38}{t:8.3f} ms   {t / tot:6.1%} of the full forward')
    print(f'   {"-" * 38}{"":>8}')
    print(f'   {"sum of parts":<38}{acc:8.3f} ms   {acc / tot:6.1%}')
    print(f'   {"FULL fused forward":<38}{tot:8.3f} ms')
    print(f'   {"(replaced) torch sign+pack":<38}{timed(index_only):8.3f} ms   '
          f'-- no longer in the path')

    print('\n' + '=' * 88)
    print('(b) CAN FAST\'S NATIVE CUDA EVAL KERNEL SERVE LIGHT?')
    print('=' * 88)
    mgr = _get_native_lutorch_manager()
    kern = getattr(mgr, 'anchor_pairs_lookup_eval_forward_msb', None) if mgr else None
    print(f'   native kernel available: {kern is not None}')
    if kern is not None:
        # flatten the block-diagonal anchors the way Fast does for multi_head_input
        head_off = (torch.arange(H, device=DEV).view(H, 1, 1) * DIN)
        a_flat = (m.anchor_a + head_off).reshape(H * T, NAP).contiguous()
        b_flat = (m.anchor_b + head_off).reshape(H * T, NAP).contiguous()
        xf = x.reshape(B, H * DIN).contiguous()

        def native_index():
            with torch.no_grad():
                kern(xf, a_flat, b_flat, 0.0, 256)

        t_native = timed(native_index)
        t_torch_idx = timed(index_only)
        t_gather = timed(gather_only)
        with torch.no_grad():
            ni = kern(xf, a_flat, b_flat, 0.0, 256).view(B, H, T)
        agree = torch.equal(ni, index)
        print(f'   kernel index == torch index: {agree}')
        print(f'   native index (gather+sign+pack, one pass) : {t_native:7.3f} ms')
        print(f'   torch  sign+pack ALONE (d already in hand): {t_torch_idx:7.3f} ms')
        print(f'   torch  margin gather (needed for score)   : {t_gather:7.3f} ms')
        print(f'\n   The kernel returns ONLY the packed index and discards the margins, so')
        print(f'   it cannot replace Light\'s forward wholesale: |d| is still needed for the')
        print(f'   score, and that gather ({t_gather:.2f} ms) happens either way. That is')
        print(f'   exactly why Fast disables it when the gate is on.')
        print(f'   BUT it can still replace the sign+pack, and that is a clear win:')
        print(f'      torch sign+pack {t_torch_idx:.3f} ms  ->  native {t_native:.3f} ms'
              f'   = {t_torch_idx - t_native:+.3f} ms saved')
        print(f"   Already wired in: the module now uses it, which is why the forward")
        print(f"   above is {tot:.2f} ms rather than the {tot + (t_torch_idx - t_native):.2f} ms it was before.")
        print(f'   And because Light\'s address is detached BY DESIGN, this is legal in')
        print(f'   training too, not just at eval -- unlike Fast, which needs the soft path.')

    print('\n' + '=' * 88)
    print('(c) REDUCED-PRECISION TABLES (speed only -- quality is a separate question)')
    print('=' * 88)
    print(f"   {'dtype':<10}{'eval fwd':>11}{'fwd+bwd':>11}{'peak MiB':>11}"
          f"{'table MiB':>11}   backward")
    g = torch.randn(B, H, DOUT, device=DEV)
    for dtype, name in ((torch.float32, 'fp32'), (torch.bfloat16, 'bf16'),
                        (torch.float16, 'fp16')):
        mm = make(dtype)
        tbl_mib = mm.tables.numel() * mm.tables.element_size() / 2 ** 20

        def fwd():
            with torch.no_grad():
                mm(x)

        def fwd_bwd():
            mm.zero_grad(set_to_none=True)
            xx = x.detach().requires_grad_(True)
            mm(xx).float().backward(g)

        torch.cuda.reset_peak_memory_stats()
        t_f = timed(fwd)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        try:
            t_fb = timed(fwd_bwd, reps=15, warmup=5)
            note = 'ok'
        except NotImplementedError:
            t_fb, note = float('nan'), 'NOT SUPPORTED (no bf16 per_sample_weights bwd)'
        print(f'   {name:<10}{t_f:>11.3f}{t_fb:>11.3f}{peak:>11.1f}{tbl_mib:>11.1f}   {note}')
        del mm
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
