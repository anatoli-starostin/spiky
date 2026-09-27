"""Head-to-head benchmark of the four kernel arms, at the paper's reference
configuration and at the repo's real shape.

ARMS
  gather   output-stationary gather+sum. The baseline: the generalisation of the repo's
           experiments/ffn_replacement/benchmark/gather_cuda.cu (hardcoded to D=48
           fp32/bf16) to arbitrary N and int8 tables. Tuned over its table-split factor.
  cs_v0    naive cell-stationary: one thread owns (table, row, K lanes), register
           accumulators, one atomicAdd per (token, lane) touched.
  cs_v1    the strengthened design as specified: wide lane groups, block-level partial-sum
           reduction in shared memory, vector atomics, optional cluster reduction.
  cs_v2    cs_v1 with the shared-memory ATOMICS removed. Exactly one row matches per
           (table, token), so the writer of each reduction slot is unique and a plain
           store suffices. Optional transposed table layout [R, NT, N].
  cs_v3    G tables per thread at the same row (the intra-thread dedup arm), direct
           vector atomics, no shared memory. Optional transposed layout.

SHAPES
  B "paper reference"  NT=256 R=256 N=1024  -> 64 MiB int8 tables; synthetic uniform
                       indices. The shape the owner's cost model is stated for.
  A "repo real"        NT=256 R=256 N=48    -> 3 MiB; ONE HEAD of blocks.0.ffn.lut_light
                       in exp_n_0196, with the real top-1 indices and real margin
                       coefficients extracted by extract_real.py.

Measurement follows the repo's FFN benchmark rules: burn-in before any timing (the 5090
idles at 1627 MHz and boosts to 3210; timing the ramp has flipped a headline sign here
before), CUDA-event timing, median of >=50 iterations where the time budget allows, and
correctness asserted against the torch reference before any timing is recorded.
Clock locking needs root and is not used; the burn-in is the substitute.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lac  # noqa: E402

BATCHES = [1, 8, 64, 256, 1024, 4096, 24576]
TSPLITS = [1, 2, 4, 8, 16, 32, 64]

V0 = [(1, 192), (1, 128), (1, 64), (2, 96), (2, 64), (4, 48), (4, 64), (4, 32),
      (4, 96), (4, 128), (8, 32), (8, 48), (8, 96), (2, 128), (16, 48), (16, 96)]
V1 = [(8, 4, 4, 1), (16, 2, 4, 1), (32, 1, 4, 1), (32, 1, 4, 8),
      (8, 32, 4, 1), (16, 32, 4, 1), (32, 32, 4, 1), (32, 16, 4, 1),
      (64, 32, 2, 1), (32, 32, 1, 1), (16, 32, 2, 1),
      (32, 16, 4, 4), (32, 16, 4, 8), (16, 32, 4, 8)]
V2 = [(16, 32, 4), (32, 32, 4), (64, 32, 4), (32, 32, 2), (64, 32, 2), (128, 32, 2),
      (64, 16, 4), (128, 16, 4), (128, 32, 1), (64, 32, 1), (32, 16, 4), (16, 32, 2)]
V3 = [(32, 32, 1), (32, 32, 4), (32, 32, 8), (64, 32, 8), (16, 32, 8), (16, 32, 32),
      (8, 32, 32), (8, 32, 64), (4, 32, 128), (4, 32, 64), (8, 32, 16), (16, 16, 32)]


# ------------------------------------------------------------------ analytic counters

def counters(kind, B, NT, R, N, cfg):
    """Exact request volumes and operation counts from the launch geometry.

    ncu is unavailable on this machine (the driver has RmProfilingAdminOnly=1, so GPU
    performance counters need root), so these are derived, not measured. They are exact
    counts of what each kernel ASKS the memory system for. What the 96 MB L2 absorbs is
    not captured here; bench_l2.py bounds that separately from achieved bandwidth.
    """
    tbl = NT * R * N
    o = {'table_set_bytes': tbl}
    if kind == 'gather':
        S = cfg['tsplit']
        o['table_requested'] = B * NT * N
        o['table_compulsory'] = min(B * NT * N, tbl)
        o['l2_amortisation_needed'] = (B * NT * N) / max(1, min(B * NT * N, tbl))
        o['index_requested'] = B * NT * S
        o['coef_requested'] = B * NT * 4 * S
        o['atomic_scalars'] = 0 if S == 1 else B * N * S
        o['blocks'] = ((B + cfg.get('tpb', 1) - 1) // cfg.get('tpb', 1)) * S
        o['compares'] = 0
        o['useful_adds'] = B * NT * N
        return o
    K, M = cfg['K'], cfg['M']
    if kind == 'v0':
        blocks, red, thr_per_blk, per_blk_tables = NT * (N // K), 1, 256, 1
    elif kind == 'v1':
        TB, CLU = cfg['TB'], cfg['CLU']
        blocks = (NT // (TB * CLU)) * (N // K) * CLU
        red, thr_per_blk, per_blk_tables = TB * CLU, 256 * TB, TB
    elif kind == 'v2':
        TB = cfg['TB']
        blocks, red, thr_per_blk, per_blk_tables = (NT // TB) * (N // K), TB, 256 * TB, TB
    else:  # v3: reduction is the measured dedup, not a fixed factor
        G = cfg['G']
        blocks, red, thr_per_blk, per_blk_tables = (NT // G) * (N // K), 1.0, 256, G
    threads = NT * R * (N // K)
    tiles = (B + M - 1) // M
    o['table_requested'] = tbl
    o['table_compulsory'] = tbl
    o['l2_amortisation_needed'] = 1.0
    o['index_requested'] = blocks * tiles * M * per_blk_tables
    o['coef_requested'] = o['index_requested'] * 4
    o['atomic_scalars'] = int(B * N * NT / red)
    o['shared_atomic_scalars'] = B * N * NT if kind == 'v1' else 0
    o['blocks'] = blocks
    o['threads'] = threads
    o['compares'] = threads * B
    o['useful_adds'] = B * NT * N
    o['compares_per_useful_add'] = threads * B / max(1, B * NT * N)
    return o


# ------------------------------------------------------------------ problems

def make_shape_B(B, dev='cuda', seed=0):
    NT, R, N = 256, 256, 1024
    g = torch.Generator(device=dev).manual_seed(seed)
    T = torch.randint(-127, 128, (NT, R, N), device=dev, dtype=torch.int8, generator=g)
    J = torch.randint(0, R, (B, NT), device=dev, dtype=torch.uint8, generator=g)
    C = torch.randn(B, NT, device=dev, generator=g) * 0.1 + 1.0
    return T, J, C


_REAL = None


def _real(head=0):
    global _REAL
    if _REAL is None:
        p = os.path.join(HERE, 'artifacts', 'real_layer.pt')
        if not os.path.exists(p):
            raise SystemExit('run extract_real.py first')
        _REAL = torch.load(p, map_location='cpu')
    return _REAL


def real_indices(B, dev='cuda', head=0):
    """The real trained top-1 indices and margin coefficients of one head.

    j is [B, n_t] and does not depend on the row width n at all -- n lives only in the
    tables -- so the real index distribution can be used at ANY output width. For
    B <= 24,576 this is a prefix of the real rows and nothing is tiled.
    """
    a = _real(head)
    tph = a['tables_per_head']
    sl = slice(head * tph, (head + 1) * tph)
    reps = (B + a['j'].shape[0] - 1) // a['j'].shape[0]
    J = a['j'][:, sl].repeat(reps, 1)[:B].to(dev).contiguous()
    C = a['c'][:, sl].repeat(reps, 1)[:B].to(dev).contiguous()
    return J, C


def make_shape_A(B, dev='cuda', head=0):
    a = _real(head)
    tph = a['tables_per_head']
    T = a['T_int8'][head * tph:(head + 1) * tph].to(dev).contiguous()
    J, C = real_indices(B, dev, head)
    return T, J, C


def make_shape_R(B, dev='cuda', seed=0, head=0):
    """Paper geometry, real index distribution.

    The white paper's reference LUT Core is 256 tables x 256 rows x 1024 lanes, which no
    trained model in this repo has; the trained layer is 256 x 256 x 48. But the index
    tensor is [B, n_t] and is independent of the row width, so the REAL trained indices
    and REAL margin coefficients can drive the paper-width tables directly. That
    separates the two things the synthetic shape conflates: the geometry (which sets the
    footprint, hence the L2 story) and the index distribution (which is what the dedup
    and skew results are about). Table VALUES stay synthetic -- no kernel here branches on
    a table value, so they cannot affect timing.
    """
    NT, R, N = 256, 256, 1024
    g = torch.Generator(device=dev).manual_seed(seed)
    T = torch.randint(-127, 128, (NT, R, N), device=dev, dtype=torch.int8, generator=g)
    J, C = real_indices(B, dev, head)
    assert J.shape[1] == NT, (J.shape, NT)
    return T, J, C


# ------------------------------------------------------------------ arms

def arms(T, Tt, J, C, y, use_coef, N):
    """(family, cfg, label, callable) for every runnable configuration."""
    out = []
    tpb = max(1, min(32, 1024 // max(1, N // 4)))
    NT = T.shape[0]
    for s in TSPLITS:
        if NT % s:
            continue
        out.append(('gather', {'tsplit': s, 'tpb': tpb}, f'gather tsplit={s}',
                    lambda s=s: lac.run_gather(T, J, C, tsplit=s, use_coef=use_coef,
                                               y=y, tok_per_blk=tpb)))
    for K, M in V0:
        if N % K:
            continue
        out.append(('v0', {'K': K, 'M': M}, f'cs_v0 K={K} M={M}',
                    lambda K=K, M=M: lac.run_v0(T, J, C, K=K, M=M, use_coef=use_coef, y=y)))
    for K, M, TB, CLU in V1:
        if N % K or NT % (TB * CLU):
            continue
        out.append(('v1', {'K': K, 'M': M, 'TB': TB, 'CLU': CLU},
                    f'cs_v1 K={K} M={M} TB={TB} CLU={CLU}',
                    lambda K=K, M=M, TB=TB, CLU=CLU: lac.run_v1(
                        T, J, C, K=K, M=M, TB=TB, CLU=CLU, use_coef=use_coef, y=y)))
    for K, M, TB in V2:
        if N % K or NT % TB:
            continue
        for tr in (0, 1):
            out.append(('v2', {'K': K, 'M': M, 'TB': TB, 'trans': tr},
                        f'cs_v2 K={K} M={M} TB={TB} trans={tr}',
                        lambda K=K, M=M, TB=TB, tr=tr: lac.run_v2(
                            Tt if tr else T, J, C, K=K, M=M, TB=TB, trans=tr,
                            use_coef=use_coef, y=y)))
    for K, M, G in V3:
        if N % K or NT % G:
            continue
        for tr in (0, 1):
            out.append(('v3', {'K': K, 'M': M, 'G': G, 'trans': tr},
                        f'cs_v3 K={K} M={M} G={G} trans={tr}',
                        lambda K=K, M=M, G=G, tr=tr: lac.run_v3(
                            Tt if tr else T, J, C, K=K, M=M, G=G, trans=tr,
                            use_coef=use_coef, y=y)))
    return out


# ------------------------------------------------------------------ timing

def time_fn(fn, budget_s=45.0, min_iters=50, burn=True, cap_ms=None):
    """Time `fn`. `cap_ms` short-circuits: a configuration whose single launch already
    exceeds it is reported from that one launch rather than timed min_iters times, so one
    pathological config (a spilling cs_v3, a cluster variant) cannot eat the whole run."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter() - t0
    if cap_ms is not None and t1 * 1e3 > cap_ms:
        ms = t1 * 1e3
        return {'median_ms': ms, 'min_ms': ms, 'max_ms': ms, 'iters': 1, 'capped': True}
    iters = min_iters if t1 * min_iters <= budget_s else max(5, int(budget_s / t1))
    if burn:
        lac.burn_in(fn, seconds=min(2.0, max(0.25, 20 * t1)))
    r = lac.timeit(fn, iters=iters, warmup=3)
    r['capped'] = False
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', default='B,A')
    ap.add_argument('--sweep-at', default='1024,24576')
    ap.add_argument('--out', default=os.path.join(HERE, 'artifacts', 'bench.json'))
    args = ap.parse_args()
    lac.mod()
    print('clocks before:', lac.clocks())
    res = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'batches': BATCHES, 'sweep': [], 'ladder': []}

    for shape in args.shapes.split(','):
        maker = make_shape_B if shape == 'B' else make_shape_A
        for use_coef in (True, False):
            tag = f'shape {shape}  c={"real" if use_coef else "1"}'
            print(f'\n{"="*104}\n{tag}\n{"="*104}')
            winners = {}
            for Bs in [int(x) for x in args.sweep_at.split(',')]:
                T, J, C = maker(Bs)
                Tt = lac.transpose_tables(T)
                NT, R, N = T.shape
                y = torch.zeros(Bs, N, device='cuda')
                ref = lac.reference(T, J, C, use_coef=use_coef, chunk=64 if N > 512 else 512)
                sc = max(ref.abs().max().item(), 1e-9)
                cap = 300.0 if Bs >= 4096 else 60.0
                print(f'\n-- sweep at B={Bs} (NT={NT} R={R} N={N}); a config over '
                      f'{cap:.0f} ms/launch is reported from one launch --', flush=True)
                rows = []
                for fam, cfg, label, fn in arms(T, Tt, J, C, y, use_coef, N):
                    fn()
                    torch.cuda.synchronize()
                    e = (y - ref).abs().max().item() / sc
                    if e >= 1e-5:
                        print(f'   {label:<34} WRONG rel={e:.2e}', flush=True)
                        continue
                    r = time_fn(fn, budget_s=12.0, min_iters=10, cap_ms=cap)
                    print(f'   {label:<34}{r["median_ms"]:>10.4f} ms   rel={e:.1e}'
                          f'{"   (1 launch)" if r["capped"] else ""}', flush=True)
                    rows.append((fam, cfg, label, r['median_ms'], e))
                print('   --- sorted ---', flush=True)
                for fam, cfg, label, ms, e in sorted(rows, key=lambda x: x[3]):
                    print(f'   {label:<34}{ms:>10.4f} ms', flush=True)
                for fam, cfg, label, ms, e in rows:
                    if fam not in winners.setdefault(Bs, {}) or ms < winners[Bs][fam][2]:
                        winners[Bs][fam] = (cfg, label, ms)
                res['sweep'].append({'shape': shape, 'use_coef': use_coef, 'B': Bs,
                                     'rows': [{'family': f, 'cfg': c, 'label': l,
                                               'median_ms': m, 'rel_err': e}
                                              for f, c, l, m, e in rows]})
                del T, J, C, Tt, y, ref
                torch.cuda.empty_cache()

            bs = sorted(winners)
            small, big = winners[bs[0]], winners[bs[-1]]
            print(f'\n-- ladder (per-family winner from the B={bs[-1]} sweep for B>=256, '
                  f'from B={bs[0]} below) --')
            print(f'{"B":>7} {"kernel":<34}{"median ms":>11}{"min":>9}{"max":>9}{"it":>5}'
                  f'{"tokens/s":>12}{"atomic scalars":>16}{"cmp/add":>9}{"x gather":>10}')
            for B in BATCHES:
                T, J, C = maker(B)
                Tt = lac.transpose_tables(T)
                NT, R, N = T.shape
                y = torch.zeros(B, N, device='cuda')
                ref = lac.reference(T, J, C, use_coef=use_coef, chunk=64 if N > 512 else 512)
                sc = max(ref.abs().max().item(), 1e-9)
                pick = big if B >= 256 else small
                want = {lbl for (_, lbl, _) in pick.values()}
                base = None
                for fam, cfg, label, fn in arms(T, Tt, J, C, y, use_coef, N):
                    if label not in want:
                        continue
                    fn()
                    torch.cuda.synchronize()
                    e = (y - ref).abs().max().item() / sc
                    assert e < 1e-5, f'{label} B={B} rel={e:.2e}'
                    r = time_fn(fn, cap_ms=2000.0)
                    ct = counters(fam, B, NT, R, N, cfg)
                    if fam == 'gather':
                        base = r['median_ms']
                    print(f'{B:>7} {label:<34}{r["median_ms"]:>11.4f}{r["min_ms"]:>9.4f}'
                          f'{r["max_ms"]:>9.4f}{r["iters"]:>5}'
                          f'{B/(r["median_ms"]*1e-3):>12.3e}{ct["atomic_scalars"]:>16.3e}'
                          f'{ct.get("compares_per_useful_add", 0):>9.2f}'
                          f'{(base or r["median_ms"])/r["median_ms"]:>10.4f}', flush=True)
                    res['ladder'].append({'shape': shape, 'use_coef': use_coef, 'B': B,
                                          'family': fam, 'cfg': cfg, 'label': label,
                                          'rel_err': e,
                                          'tokens_per_s': B / (r['median_ms'] * 1e-3),
                                          'vs_gather': (base or r['median_ms']) / r['median_ms'],
                                          **r, **ct})
                del T, J, C, Tt, y, ref
                torch.cuda.empty_cache()

    res['clocks_after'] = lac.clocks()
    print('\nclocks after:', res['clocks_after'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
