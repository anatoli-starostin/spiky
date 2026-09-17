#!/usr/bin/env python3
"""Paper exp_n_0127 + vanilla, and the new int8 / config-B rows, under BOTH float32_matmul_precision settings (task bb461ce7).

    python paper_timings/p2int8_paper0127.py --precision high --tag r1
    python paper_timings/p2int8_paper0127.py --precision highest --tag r1

One process = one precision (set once, first thing, asserted). Methodology as p2int8_B_timing.py: real val-text block-0
FFN inputs captured through each row's OWN trained model (0127: its paper checkpoint; int8/B/T16: abl_48; vanilla: abl_10,
a trained dense model of the paper vanilla's exact shape -- the paper's exp_n_0135 has no checkpoint and dense timing is
weight-independent), inputs pre-cast to each row's dtype outside timing, correctness gate and call counters before timing,
300-call global warm-up + 60 per row, 11 interleaved rounds with P1 (harness timeit) and P2 (CUDA events), and every
stage of every bar timed on its real intermediates for the stacked-bar figure.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--precision', required=True, choices=('high', 'highest'))
ap.add_argument('--tag', default='r1')
ARGS = ap.parse_args()
torch.set_float32_matmul_precision(ARGS.precision)          # set ONCE, before anything else runs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2int8_verify as V                                    # noqa: E402
import bench                                                 # noqa: E402
import gather_fused                                          # noqa: E402
import hybrid                                                # noqa: E402
import model as M                                            # noqa: E402
from spiky.lutorch import pow2_int8                          # noqa: E402
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut  # noqa: E402

BF = torch.bfloat16
N, C, B_, SEQ = V.N, V.C, V.B, V.SEQ
PAPER_ROOT = os.path.expanduser('~/projects/spiky-fmhl-next/experiments/hyperplane_ffn')
E127 = 'exp_n_0127_grid_H4d48_nap7_tph128'


def main():
    res = {'tag': ARGS.tag, 'precision_requested': ARGS.precision, 'pid': os.getpid(), 'gpu': bench.gpu_name(),
           'torch': torch.__version__}
    res['other_compute_apps_start'] = V.compute_apps()
    res['idle_samples_before'] = [V.smi('utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.active,temperature.gpu')
                                  for _ in range(5)]
    ok, msg = pow2_int8.available()
    reg = pow2_int8.ensure_registered()
    okf, msgf = gather_fused.available()
    res['kernels'] = dict(p2=msg, p2_registered=reg, fused=msgf, mapped_p2_before=V.mapped_p2_ext())
    V.log(f'[{ARGS.precision}/{ARGS.tag}] apps {res["other_compute_apps_start"]} | idle {res["idle_samples_before"]} | {msg} | {msgf}')
    if not (ok and okf and res['kernels']['mapped_p2_before']):
        V.log('a kernel is not serving; abort'); return 1

    toks = V.val_tokens()
    # ---- vanilla (abl_10 trained dense, same shape as the paper's exp_n_0135)
    _, van32 = V.load(V.VAN)
    xv = V.capture_ffn_inputs(van32, toks, [0])[0][0]
    _, van16 = V.load(V.VAN)
    van16 = van16.to(BF)
    mlp32, mlp16 = van32.blocks[0].mlp, van16.blocks[0].mlp
    xv32, xv16 = xv.contiguous(), xv.to(BF).contiguous()

    # ---- paper 0127 (its own checkpoint), real inputs through its own fp32 model
    d127 = os.path.join(PAPER_ROOT, E127)
    _, ref127 = M.build(d127, load_checkpoint=True)
    got = {}
    h = ref127.blocks[0].ffn.register_forward_pre_hook(lambda m, a: got.__setitem__('x', a[0].detach().float().clone()))
    with torch.no_grad():
        ref127(toks)
    h.remove()
    x127_32 = got['x'].reshape(B_, SEQ, C).contiguous()
    x127_16 = x127_32.to(BF).contiguous()
    p32 = M.build(d127, load_checkpoint=True)[1]
    assert gather_fused.patch(p32, table_dtype='fp32') == len(M.lut_modules(p32))
    p16 = hybrid.apply(M.build(d127, load_checkpoint=True)[1])
    assert gather_fused.patch(p16, table_dtype='bf16') == len(M.lut_modules(p16))
    l127 = M.lut_modules(p32)[0]
    c32, c16 = p32.blocks[0].ffn, p16.blocks[0].ffn

    # ---- abl_48 int8 rows
    _, m48 = V.load(V.QNT)
    xq = V.capture_ffn_inputs(m48, toks, [0])[0][0]
    q = m48.blocks[0].ffn.export_quantised().cuda()
    q_today = m48.blocks[0].ffn.export_quantised().cuda()

    def forbid(x):
        raise RuntimeError('torch fallback served')
    q._forward_torch = forbid
    q_today._forward_torch = forbid
    mt = q.meta
    H, Din, D, NAP, T = mt['n_heads'], mt['input_dim'], mt['output_dim'], mt['n_anchor_pairs'], mt['tables_per_head']
    kc = q._kernel_cache(torch.device('cuda'))
    lo, hi, Q = q.cfg['lo'], q.cfg['hi'], q.cfg['Q']
    cw32, cb32, dw32, db32 = q.compress_weight, q.compress_bias, q.decompress_weight, q.decompress_bias
    cw16, cb16, dw16, db16 = (t.to(BF).contiguous() for t in (cw32, cb32, dw32, db32))
    xq32 = xq.reshape(N, C).contiguous()
    xq16 = xq32.to(BF).contiguous()

    def read(z):
        return pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, lo, hi, Q)

    def b_slot():
        z = F.linear(xq16, cw16, cb16).view(N, H, Din).float()
        return F.linear(read(z).reshape(N, H * D).to(BF), dw16, db16).reshape(B_, SEQ, C)

    # ---- real intermediates for stages
    with torch.no_grad():
        up32 = F.gelu(mlp32[0](xv32)); up16 = F.gelu(mlp16[0](xv16))
        z127_32 = c32.compress(x127_32.reshape(N, C)).view(N, 4, 48).contiguous()
        y127_32 = c32.lut_batched(z127_32).to(z127_32.dtype).reshape(N, -1).contiguous()
        z127_16 = c16.compress(x127_16.reshape(N, C)).view(N, 4, 48).contiguous()
        y127_16 = c16.lut_batched(z127_16).to(z127_16.dtype).reshape(N, -1).contiguous()
        zq32 = F.linear(xq32, cw32, cb32).view(N, H, Din).contiguous()
        accq32 = read(zq32).reshape(N, H * D).contiguous()
        zb16 = F.linear(xq16, cw16, cb16).view(N, H, Din).contiguous()
        zbf = zb16.float().contiguous()
        accb32 = read(zbf).reshape(N, H * D).contiguous()
        accb16 = accb32.to(BF).contiguous()
        xt = xq16.float().contiguous()
        yt32 = q_today(xt).contiguous()

    rows = {
        'V16': lambda: mlp16(xv16),
        'V32': lambda: mlp32(xv32),
        'P127_16': lambda: p16.blocks[0].ffn_slot(x127_16),
        'P127_32': lambda: p32.blocks[0].ffn_slot(x127_32),
        'Q32': lambda: q(xq32).reshape(B_, SEQ, C),
        'B': b_slot,
        'T16': lambda: q_today(xq16.float()).reshape(B_, SEQ, C).to(BF),
    }
    stages = {
        'V16': {'Linear 384->1536 + GELU': lambda: F.gelu(mlp16[0](xv16)), 'Linear 1536->384': lambda: mlp16[2](up16)},
        'V32': {'Linear 384->1536 + GELU': lambda: F.gelu(mlp32[0](xv32)), 'Linear 1536->384': lambda: mlp32[2](up32)},
        'P127_16': {'compress 384->192': lambda: c16.compress(x127_16.reshape(N, C)),
                    'routing + gather (fused)': lambda: c16.lut_batched(z127_16),
                    'decompress 192->384': lambda: c16.decompress(y127_16)},
        'P127_32': {'compress 384->192': lambda: c32.compress(x127_32.reshape(N, C)),
                    'routing + gather (fused)': lambda: c32.lut_batched(z127_32),
                    'decompress 192->384': lambda: c32.decompress(y127_32)},
        'Q32': {'compress 384->192': lambda: F.linear(xq32, cw32, cb32), 'int8 read (kernel)': lambda: read(zq32),
                'decompress 192->384': lambda: F.linear(accq32, dw32, db32)},
        'B': {'compress 384->192': lambda: F.linear(xq16, cw16, cb16), 'dtype casts': lambda: zb16.float(),
              'int8 read (kernel)': lambda: read(zbf), 'dtype casts 2': lambda: accb32.to(BF),
              'decompress 192->384': lambda: F.linear(accb16, dw16, db16)},
        'T16': {'dtype casts': lambda: xq16.float(), 'compress 384->192': lambda: F.linear(xt, cw32, cb32),
                'int8 read (kernel)': lambda: read(zq32), 'decompress 192->384': lambda: F.linear(accq32, dw32, db32),
                'dtype casts 2': lambda: yt32.to(BF)},
    }

    # ---- correctness gate + counters, before timing
    chk = {}
    with torch.no_grad():
        yref = ref127.blocks[0].ffn_slot(x127_32)
        chk['P127_32_vs_unpatched_fp32_max_abs'] = (rows['P127_32']() - yref).abs().max().item()
        chk['P127_16_vs_unpatched_fp32_rel'] = ((rows['P127_16']().float() - yref).abs().max() / yref.abs().max()).item()
        ytr = m48.blocks[0].ffn(xq32)
        chk['Q32_vs_training_read_max_abs'] = (q(xq32) - ytr).abs().max().item()
        cells_k = torch.empty(N, H, T, 3, dtype=torch.uint8, device='cuda')
        pow2_int8.read_fused(zbf, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, lo, hi, Q, cells_out=cells_k)
        ia = q.anchor_a.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        ib = q.anchor_b.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        dm = (torch.gather(zbf, 2, ia) - torch.gather(zbf, 2, ib)).view(N, H, T, NAP).contiguous()
        _p, oc, okq = torch.ops.spiky_lutorch.p2_scalars(dm, q.tau.float(), q.g.float(), q.beta.float(), q.gamma.float(), lo, hi, Q)
        chk['B_integers_eq_training_op'] = bool(torch.equal(cells_k, pow2_int8.pack_cells(oc[..., :2].to(torch.int64), *pow2_int8._integers_from(oc, okq, Q))))
    # P127_16 is approximate by construction (bf16 dense + bf16 table): gate it with the PAPER's own criterion
    # (run_bench.py: bf16-table gather output rel <= 1e-2, gather_fused.check_table_precision), not on the slot output,
    # which also carries the bf16 compress/decompress rounding (reported above, not gated; a first 1e-2 slot gate refused).
    worst = 0.0
    for l in M.lut_modules(ref127):
        rel, _dif, _sc = gather_fused.check_table_precision(l, n_tokens=N)
        worst = max(worst, rel)
    chk['P127_16_bf16_table_gather_rel_worst'] = worst
    gate = (chk['P127_32_vs_unpatched_fp32_max_abs'] == 0.0 and chk['Q32_vs_training_read_max_abs'] == 0.0
            and chk['B_integers_eq_training_op'] and worst <= 1e-2)
    chk['gate_pass'] = gate
    cnt = {'fast_127_32': 0, 'fast_127_16': 0, 'read_fused': 0, 'q_fused': 0, 'today_fused': 0}
    saved = []
    for key, mdl in (('fast_127_32', p32), ('fast_127_16', p16)):
        for mm in mdl.modules():
            if isinstance(mm, FastMultiHeadLut):
                o = mm._hard_eval_native
                saved.append((mm, o))
                mm._hard_eval_native = (lambda x, w, _o=o, _k=key: (cnt.__setitem__(_k, cnt[_k] + 1), _o(x, w))[1])
    orf = pow2_int8.read_fused
    pow2_int8.read_fused = lambda *a, **k: (cnt.__setitem__('read_fused', cnt['read_fused'] + 1), orf(*a, **k))[1]
    oq, ot = q._forward_fused, q_today._forward_fused
    q._forward_fused = lambda x: (cnt.__setitem__('q_fused', cnt['q_fused'] + 1), oq(x))[1]
    q_today._forward_fused = lambda x: (cnt.__setitem__('today_fused', cnt['today_fused'] + 1), ot(x))[1]
    per = {}
    with torch.no_grad():
        for k in ('P127_32', 'P127_16', 'Q32', 'B', 'T16'):
            before = dict(cnt)
            for _ in range(10):
                rows[k]()
            per[k] = {kk: cnt[kk] - before[kk] for kk in cnt}
    for mm, o in saved:
        mm._hard_eval_native = o
    pow2_int8.read_fused = orf
    q._forward_fused, q_today._forward_fused = oq, ot
    chk['calls_per_10'] = per
    res['checks'] = chk
    V.log(f'[{ARGS.precision}/{ARGS.tag}] checks {json.dumps(chk)}')
    if not gate:
        V.log('correctness gate FAILED: refusing to time')
        json.dump(res, open(os.path.join(V.OUT_DIR, f'fig0127_{ARGS.precision}_{ARGS.tag}.json'), 'w'), indent=2, default=str)
        return 2

    all_fns = dict(rows)
    for r, st in stages.items():
        for s, f in st.items():
            all_fns[f'{r}|{s}'] = f
    with torch.no_grad():
        for _ in range(300):
            rows['V16']()
        for f in all_fns.values():
            for _ in range(60):
                f()
    torch.cuda.synchronize()
    tele = V.Telemetry()
    t0 = time.time()
    acc = {k: {'P1': [], 'P2': []} for k in all_fns}
    with torch.no_grad():
        for _ in range(11):
            for k, f in all_fns.items():
                acc[k]['P1'].append(bench.timeit(f, iters=30, warmup=2))
                acc[k]['P2'].append(V.timeit_events(f, 30))
    res['timing_seconds'] = time.time() - t0
    res['telemetry'] = tele.stop()
    res['other_compute_apps_end'] = V.compute_apps()
    res['kernels']['mapped_p2_after'] = V.mapped_p2_ext()
    res['precision_asserted'] = torch.get_float32_matmul_precision()
    assert res['precision_asserted'] == ARGS.precision
    res['median_ms'] = {k: {p: dict(median=statistics.median(v), min=min(v), max=max(v)) for p, v in d.items()}
                        for k, d in acc.items()}
    for k in all_fns:
        m = res['median_ms'][k]
        V.log(f'[{ARGS.precision}/{ARGS.tag}] {k:<40} P1 {m["P1"]["median"]:.4f} [{m["P1"]["min"]:.4f}-{m["P1"]["max"]:.4f}]  P2 {m["P2"]["median"]:.4f}')
    V.log(f'[{ARGS.precision}/{ARGS.tag}] telemetry {json.dumps(res["telemetry"])} | apps end {res["other_compute_apps_end"]}')
    json.dump(res, open(os.path.join(V.OUT_DIR, f'fig0127_{ARGS.precision}_{ARGS.tag}.json'), 'w'), indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
