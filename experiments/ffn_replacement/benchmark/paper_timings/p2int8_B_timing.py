#!/usr/bin/env python3
"""Configuration B timing (task 2891e788): bf16 compress -> z fp32 -> kernel (fp32 decisions) -> acc bf16 -> bf16 decompress.

    python paper_timings/p2int8_B_timing.py --tag r1      # one fresh-process launch; run 3 times

Methodology of p2int8_verify.py --mode timing (task 0341bbbc): real val-text block-0 FFN inputs captured through each
row's own trained model at the trainer's precision (float32_matmul_precision left at the default 'highest' -- NOT set
anywhere in this process), inputs pre-cast to each row's own dtype outside timing, 300-call global warm-up + 60 per row,
protocol A (60-call burn-in, median of 7 x timeit(30)) and 11 interleaved rounds with two timing paths (P1 harness
timeit, P2 CUDA events). B is timed BOTH inline (plain ops in a lambda) and through an nn.Module with the artefact's
forward structure, and every stage and stage pair of B is timed on its real intermediates.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2int8_verify as V                                # noqa: E402
import bench                                             # noqa: E402
from spiky.lutorch import pow2_int8                      # noqa: E402

BF = torch.bfloat16
N, C, B_, SEQ = V.N, V.C, V.B, V.SEQ


class BSlot(nn.Module):
    """Configuration B as a module, mirroring QuantisedLightFFN.forward/_forward_fused's structure (checks, no_grad,
    kernel cache lookup), with bf16 compress/decompress copies of the artefact's buffers."""
    def __init__(self, q):
        super().__init__()
        self.q = q
        mt = q.meta
        self.H, self.Din, self.D, self.NAP = mt['n_heads'], mt['input_dim'], mt['output_dim'], mt['n_anchor_pairs']
        self.cw16, self.cb16, self.dw16, self.db16 = (t.to(BF).contiguous() for t in (
            q.compress_weight, q.compress_bias, q.decompress_weight, q.decompress_bias))
        self.calls = 0

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.q.meta['model_dim']:
            raise ValueError('bad shape')
        with torch.no_grad():
            n = x.shape[0]
            kc = self.q._kernel_cache(x.device)
            z = F.linear(x, self.cw16, self.cb16).view(n, self.H, self.Din).float()
            acc = pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], self.NAP, self.D,
                                       self.q.cfg['lo'], self.q.cfg['hi'], self.q.cfg['Q'])
            return F.linear(acc.reshape(n, self.H * self.D).to(BF), self.dw16, self.db16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='r1')
    args = ap.parse_args()
    res = {'tag': args.tag, 'pid': os.getpid(), 'gpu': bench.gpu_name(), 'torch': torch.__version__}
    res['other_compute_apps_start'] = V.compute_apps()
    res['idle_samples_before'] = [V.smi('utilization.gpu,clocks.sm,power.draw,clocks_event_reasons.active') for _ in range(5)]
    V.log(f'[{args.tag}] other compute apps {res["other_compute_apps_start"]} | idle {res["idle_samples_before"]}')
    ok, msg = pow2_int8.available()                      # returns only after the build/load finished
    reg = pow2_int8.ensure_registered()
    res['kernel'] = dict(available=ok, message=msg, registered=reg, mapped_before=V.mapped_p2_ext())
    if not (ok and res['kernel']['mapped_before']):
        V.log('kernel not serving; abort'); return 1
    res['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    V.log(f'[{args.tag}] kernel {msg} | mapped {res["kernel"]["mapped_before"]} | matmul precision {res["float32_matmul_precision"]}')
    assert res['float32_matmul_precision'] == 'highest'

    toks = V.val_tokens()
    _, van32 = V.load(V.VAN)
    xv = V.capture_ffn_inputs(van32, toks, [0])[0][0]
    _, van16 = V.load(V.VAN)
    van16 = van16.to(BF)
    _, m48 = V.load(V.QNT)
    xq = V.capture_ffn_inputs(m48, toks, [0])[0][0]
    q = m48.blocks[0].ffn.export_quantised().cuda()          # Q32 row (module, kernel)
    q_today = m48.blocks[0].ffn.export_quantised().cuda()    # today's bf16 path (module)
    bmod = BSlot(m48.blocks[0].ffn.export_quantised().cuda())

    def forbid(x):
        raise RuntimeError('torch fallback served')
    q._forward_torch = forbid
    q_today._forward_torch = forbid

    mlp32, mlp16 = van32.blocks[0].mlp, van16.blocks[0].mlp
    xv32, xv16 = xv.contiguous(), xv.to(BF).contiguous()
    xq32 = xq.reshape(N, C).contiguous()
    xq16 = xq.reshape(N, C).to(BF).contiguous()
    xq32_btc, xq16_btc = xq32.view(B_, SEQ, C), xq16.view(B_, SEQ, C)
    qb = bmod.q
    mt = qb.meta
    H, Din, D, NAP = mt['n_heads'], mt['input_dim'], mt['output_dim'], mt['n_anchor_pairs']
    kc = qb._kernel_cache(xq32.device)
    lo, hi, Q = qb.cfg['lo'], qb.cfg['hi'], qb.cfg['Q']
    cw16, cb16, dw16, db16 = bmod.cw16, bmod.cb16, bmod.dw16, bmod.db16
    cw32, cb32, dw32, db32 = qb.compress_weight, qb.compress_bias, qb.decompress_weight, qb.decompress_bias

    def read(z):
        return pow2_int8.read_fused(z, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, lo, hi, Q)

    def b_inline():
        z = F.linear(xq16_btc.reshape(N, C), cw16, cb16).view(N, H, Din).float()
        return F.linear(read(z).reshape(N, H * D).to(BF), dw16, db16).reshape(B_, SEQ, C)

    def q32_inline():
        z = F.linear(xq32_btc.reshape(N, C), cw32, cb32).view(N, H, Din)
        return F.linear(read(z).reshape(N, H * D), dw32, db32).reshape(B_, SEQ, C)

    with torch.no_grad():                                    # real B intermediates for the stage breakdown
        z16 = F.linear(xq16, cw16, cb16).view(N, H, Din).contiguous()
        zf = z16.float().contiguous()
        acc32 = read(zf).reshape(N, H * D).contiguous()
        acc16 = acc32.to(BF).contiguous()

    rows = {
        'V16 dense vanilla bf16': lambda: mlp16(xv16),
        'V32 dense vanilla fp32': lambda: mlp32(xv32),
        'Q32 int8 kernel fp32 (module)': lambda: q(xq32_btc.reshape(N, C)).reshape(B_, SEQ, C),
        'Q32i int8 kernel fp32 (inline)': q32_inline,
        'B   config B (module)': lambda: bmod(xq16_btc.reshape(N, C)).reshape(B_, SEQ, C),
        'Bi  config B (inline)': b_inline,
        'T16 today bf16: x.float() -> fp32 module -> .to(bf16)': lambda: q_today(xq16_btc.reshape(N, C).float()).reshape(B_, SEQ, C).to(BF),
        's1 B compress bf16': lambda: F.linear(xq16, cw16, cb16),
        's2 B cast z bf16->fp32': lambda: z16.float(),
        's3 B kernel read (fp32 z)': lambda: read(zf),
        's4 B cast acc fp32->bf16': lambda: acc32.to(BF),
        's5 B decompress bf16': lambda: F.linear(acc16, dw16, db16),
        'p12 compress bf16 + cast z': lambda: F.linear(xq16, cw16, cb16).view(N, H, Din).float(),
        'p123 compress bf16 + cast z + read': lambda: read(F.linear(xq16, cw16, cb16).view(N, H, Din).float()),
        'p345 read + cast acc + decompress bf16': lambda: F.linear(read(zf).reshape(N, H * D).to(BF), dw16, db16),
    }

    # ---- correctness / path, before timing
    chk = {}
    with torch.no_grad():
        yb_mod, yb_inl = rows['B   config B (module)'](), rows['Bi  config B (inline)']()
        chk['B_module_eq_inline'] = bool(torch.equal(yb_mod, yb_inl))
        yq = rows['Q32 int8 kernel fp32 (module)']()
        chk['B_vs_Q32_rel'] = ((yb_mod.float() - yq).abs().max() / yq.abs().max()).item()
        chk['Q32_inline_eq_module'] = bool(torch.equal(rows['Q32i int8 kernel fp32 (inline)'](), yq))
        # B's integers are the kernel's on B's fp32 z: compare with the p2_scalars op on the same z
        cells_k = torch.empty(N, H, 128, 3, dtype=torch.uint8, device='cuda')
        pow2_int8.read_fused(zf, kc['anchor_a'], kc['anchor_b'], kc['tables'], kc['scalars'], NAP, D, lo, hi, Q, cells_out=cells_k)
        T = mt['tables_per_head']
        ia = qb.anchor_a.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        ib = qb.anchor_b.reshape(1, H, T * NAP).expand(N, H, T * NAP)
        dm = (torch.gather(zf, 2, ia) - torch.gather(zf, 2, ib)).view(N, H, T, NAP).contiguous()
        _psw, ocells, okq = torch.ops.spiky_lutorch.p2_scalars(dm, qb.tau.float(), qb.g.float(), qb.beta.float(),
                                                               qb.gamma.float(), lo, hi, Q)
        op_cells = pow2_int8.pack_cells(ocells[..., :2].to(torch.int64), *pow2_int8._integers_from(ocells, okq, Q))
        chk['B_kernel_integers_eq_training_op_on_same_z'] = bool(torch.equal(cells_k, op_cells))
    # call counters (one pass, then removed)
    cnt = {'read_fused': 0, 'fused': 0, 'torch': 0}
    orig_rf = pow2_int8.read_fused
    pow2_int8.read_fused = lambda *a, **k: (cnt.__setitem__('read_fused', cnt['read_fused'] + 1), orig_rf(*a, **k))[1]
    of = q._forward_fused
    q._forward_fused = lambda x: (cnt.__setitem__('fused', cnt['fused'] + 1), of(x))[1]
    with torch.no_grad():
        for _ in range(10):
            rows['B   config B (module)']()
    chk['B_module_10_calls'] = dict(read_fused=cnt['read_fused'])
    cnt['read_fused'] = 0
    with torch.no_grad():
        for _ in range(10):
            rows['Q32 int8 kernel fp32 (module)']()
    chk['Q32_module_10_calls'] = dict(fused=cnt['fused'], read_fused=cnt['read_fused'])
    pow2_int8.read_fused = orig_rf
    q._forward_fused = of
    res['checks'] = chk
    V.log(f'[{args.tag}] checks {json.dumps(chk)}')

    # ---- warm-up, then timing
    with torch.no_grad():
        for _ in range(300):
            rows['V16 dense vanilla bf16']()
        for f in rows.values():
            for _ in range(60):
                f()
    torch.cuda.synchronize()
    tele = V.Telemetry()
    t0 = time.time()
    A = {}
    with torch.no_grad():
        for k, f in rows.items():
            for _ in range(60):
                f()
            torch.cuda.synchronize()
            A[k] = statistics.median([bench.timeit(f, iters=30) for _ in range(7)])
    acc = {k: {'P1': [], 'P2': []} for k in rows}
    with torch.no_grad():
        for _ in range(11):
            for k, f in rows.items():
                acc[k]['P1'].append(bench.timeit(f, iters=30, warmup=2))
                acc[k]['P2'].append(V.timeit_events(f, 30))
    res['timing_seconds'] = time.time() - t0
    res['telemetry'] = tele.stop()
    res['other_compute_apps_end'] = V.compute_apps()
    res['kernel']['mapped_after'] = V.mapped_p2_ext()
    res['kernel']['fallback_compiled'] = [q._compiled is not None, q_today._compiled is not None, qb._compiled is not None]
    res['protocol_A'] = A
    res['interleaved'] = {k: {p: dict(median=statistics.median(v), min=min(v), max=max(v)) for p, v in d.items()}
                          for k, d in acc.items()}
    for k in rows:
        i = res['interleaved'][k]
        V.log(f'[{args.tag}] {k:<58} A {A[k]:.4f}  P1 {i["P1"]["median"]:.4f} [{i["P1"]["min"]:.4f}-{i["P1"]["max"]:.4f}]  P2 {i["P2"]["median"]:.4f}')
    V.log(f'[{args.tag}] telemetry {json.dumps(res["telemetry"])} | apps end {res["other_compute_apps_end"]} | '
          f'mapped {res["kernel"]["mapped_after"]} | fallback compiled {res["kernel"]["fallback_compiled"]}')
    os.makedirs(V.OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(V.OUT_DIR, f'B_timing_{args.tag}.json'), 'w'), indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
