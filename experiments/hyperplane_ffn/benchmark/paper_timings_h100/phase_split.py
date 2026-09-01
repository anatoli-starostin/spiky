#!/usr/bin/env python3
"""Paper-citable H100 FFN-slot phase timings — H100 counterpart of ../paper_timings/.
Run on an IDLE GPU from anywhere (paths derive from this file). Builds the three
CompressionMHL models on their trained checkpoints, times each phase of the fused_v2
best path + the vanilla dense split + eager-vs-torch.compile, and writes results.json
(read by make_figure.py) + phase_split.txt. Needs the JIT kernels one dir up
(gather_fused_v2_h100.cu / route_v2_h100.cu / route_shared_h100.cu; -std=c++20)."""
import json, os, statistics, sys
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_cache_gap')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
os.environ.setdefault('HOME', '/tmp')
import torch
torch.set_grad_enabled(False); torch.set_float32_matmul_precision('high')
# Portable: derive paths from this file's location (benchmark/paper_timings_h100/).
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)                       # experiments/hyperplane_ffn/benchmark
R = os.path.dirname(BENCH)                          # experiments/hyperplane_ffn
REPO = os.path.dirname(os.path.dirname(R))          # repo root
for p in (BENCH, os.path.join(REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
import bench, gather, model as M  # noqa
MODELS = [('0126', 'exp_n_0126_grid_H4d48_nap7_tph64'),
          ('0127', 'exp_n_0127_grid_H4d48_nap7_tph128'),
          ('0128', 'exp_n_0128_grid_H4d48_nap8_tph64')]
VAN = 'exp_n_0135_untied_vanilla_baseline_16k'
B, T, Ch = 48, 512, 384
N = B * T; DEV = 'cuda'; OUT = []
ITERS, ROUNDS, WARMUP = 50, 15, 20
RES = {}


def logp(s=''):
    print(s, flush=True); OUT.append(s)


def ct(fn):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(ROUNDS):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS):
                fn()
            e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / ITERS)
    return statistics.median(ts)


def clocks():
    try:
        import subprocess
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=clocks.sm,clocks.max.sm,memory.used,utilization.gpu',
             '--format=csv,noheader']).decode().strip()
    except Exception as ex:
        return f'(clock query failed: {ex})'


def main():
    from torch.utils.cpp_extension import load
    ef = load(name='hyperplane_fused_v2', sources=[os.path.join(BENCH, 'gather_fused_v2_h100.cu')],
              extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
              extra_cflags=['-O3', '-std=c++20'], verbose=False)
    er = load(name='hyperplane_route_v2', sources=[os.path.join(BENCH, 'route_v2_h100.cu')],
              extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
              extra_cflags=['-O3', '-std=c++20'], verbose=False)
    er1 = load(name='hyperplane_route', sources=[os.path.join(BENCH, 'route_shared_h100.cu')],
               extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
               extra_cflags=['-O3', '-std=c++20'], verbose=False)

    cond = dict(gpu=bench.gpu_name(), torch=torch.__version__, cuda=torch.version.cuda,
                batch=B, seq=T, tokens=N, iters=ITERS, rounds=ROUNDS, warmup=WARMUP,
                clocks=clocks(),
                dtype='routed: bf16 compress/decompress + fp32 fused routing+gather (bit-exact); vanilla: bf16',
                timing='CUDA events, median, steady-state (warmed, clock boosted/unlocked), no_grad')
    RES['conditions'] = cond
    logp('=== H100 FFN-slot timing (paper table) ===')
    logp(f'GPU: {cond["gpu"]} | torch {cond["torch"]}, CUDA {cond["cuda"]}')
    logp(f'Conditions: batch {B}x{T}={N:,} tok/slot | {ITERS}x{ROUNDS} (median), {WARMUP} warmup | '
         f'CUDA events | steady-state | clocks(sm,max,mem,util)= {cond["clocks"]}')
    logp(f'dtype: {cond["dtype"]}')

    # ---- vanilla dense (bf16) ----
    _, van = M.build(os.path.join(R, VAN))
    mlp = van.blocks[0].mlp.to(torch.bfloat16)
    up, gelu, down = mlp[0], mlp[1], mlp[2]
    xv = torch.randn(N, Ch, device=DEV, dtype=torch.bfloat16)
    xv3 = torch.randn(B, T, Ch, device=DEV, dtype=torch.bfloat16)   # pre-allocated slot input
    h1 = up(xv); g = gelu(h1)
    v = dict(up=ct(lambda: up(xv)), gelu=ct(lambda: gelu(h1)), down=ct(lambda: down(g)))
    v['slot_eager'] = ct(lambda: van.blocks[0].ffn_slot(xv3))
    cvan = torch.compile(lambda x: van.blocks[0].ffn_slot(x))
    for _ in range(6):
        cvan(xv3)
    torch.cuda.synchronize()
    v['slot_compile'] = ct(lambda: cvan(xv3))
    RES['vanilla'] = v
    logp('\n--- VANILLA dense FFN (bf16), ms/call ---')
    logp(f'  up 384->1536 {v["up"]:.4f} | GELU {v["gelu"]:.4f} | down 1536->384 {v["down"]:.4f} | '
         f'sum {v["up"]+v["gelu"]+v["down"]:.4f} | slot eager {v["slot_eager"]:.4f} | compile {v["slot_compile"]:.4f}')

    RES['routed'] = {}
    logp('\n--- ROUTED best path (fused_v2), per-phase ms/call ---')
    logp(f'{"model":<6}{"P1 compress":>12}{"P2+3 fused":>12}{"P4 decompress":>14}{"slot":>9}{"vs_van":>8}'
         f'   {"[P2 route]":>11}{"[P3 gather]":>12}')
    for short, exp in MODELS:
        cfg, ref = M.build(os.path.join(R, exp), load_checkpoint=True)
        ffn = ref.blocks[0].ffn
        lut = M.lut_modules(ref)[0]
        H, tph, nap = lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
        a, b = lut.soft_anchor_a_long, lut.soft_anchor_b_long
        W = lut.weights.data.contiguous()
        cbf = ffn.compress.to(torch.bfloat16)
        dbf = ffn.decompress.to(torch.bfloat16)
        x = torch.randn(N, Ch, device=DEV, dtype=torch.bfloat16)
        z = cbf(x).float().contiguous()
        y = ef.fused_v2(z, a, b, W, H, tph, nap, 64)
        yb = y.to(torch.bfloat16).reshape(N, H * lut.n_outputs)
        p1 = ct(lambda: cbf(x))
        p23 = ct(lambda: ef.fused_v2(z, a, b, W, H, tph, nap, 64))
        p4 = ct(lambda: dbf(yb))
        idx = lut._native_eval_msb(z, a, b, 0.0, 256)
        tb = gather.tune(W, idx, H, tph)
        gather.BLOCK_N, gather.NUM_WARPS, gather.NUM_STAGES = tb[1], tb[2], tb[3]
        rt = ct(lambda: er.route_v2(z, a, b, H, tph, nap, 32, 256)) if short == '0128' \
            else ct(lambda: er1.route_shared(z, a, b, H, tph, nap, 128, 512))
        gt = ct(lambda: gather.gather_sum(W, idx, H, tph))
        slot = p1 + p23 + p4
        vsum = v['up'] + v['gelu'] + v['down']
        RES['routed'][short] = dict(nap=nap, tph=tph, compress=p1, fused_route_gather=p23,
                                    decompress=p4, slot=slot, vs_vanilla=slot / vsum,
                                    route_standalone=rt, gather_standalone=gt)
        logp(f'{short:<6}{p1:>12.4f}{p23:>12.4f}{p4:>14.4f}{slot:>9.4f}{slot/vsum:>7.2f}x'
             f'   {rt:>11.4f}{gt:>12.4f}')

    logp('\n--- eager vs torch.compile, routed full slot (fp32 reference path) ---')
    for short, exp in MODELS:
        cfg, ref = M.build(os.path.join(R, exp), load_checkpoint=True)
        blk = ref.blocks[0]
        h = torch.randn(B, T, Ch, device=DEV)
        eslot = ct(lambda: blk.ffn_slot(h))
        import torch._dynamo as dyn
        dyn.reset()
        try:
            gb = getattr(dyn.explain(lambda x: blk.ffn_slot(x))(h), 'graph_break_count', -1)
        except Exception:
            gb = -1
        cc = torch.compile(lambda x: blk.ffn_slot(x))
        for _ in range(6):
            cc(h)
        torch.cuda.synchronize()
        cslot = ct(lambda: cc(h))
        RES['routed'][short]['slot_eager_fp32'] = eslot
        RES['routed'][short]['slot_compile_fp32'] = cslot
        RES['routed'][short]['graph_breaks'] = gb
        logp(f'{short:<6} eager_fp32_slot {eslot:.4f} | compile {cslot:.4f} | graph_breaks {gb}')

    open(os.path.join(HERE, 'phase_split.txt'), 'w').write('\n'.join(OUT) + '\n')
    json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
    print('PAPER_OK', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        import traceback; traceback.print_exc()
        open(os.path.join(HERE, 'phase_split.txt'), 'w').write('\n'.join(OUT) + f'\nFAILED: {e}\n')
        print('PAPER_FAIL', flush=True); sys.exit(1)
