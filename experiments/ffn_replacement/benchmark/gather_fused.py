"""Optional FUSED routing+gather fast path (see gather_fused.cu).

Like gather_cuda.py this is OPT-IN, JIT-built at first use, and never raises: `available()`
reports why it could not build and the driver falls back to the portable Triton path.

What it fuses: the anchor-compare routing and the gather+sum become ONE kernel, with the
index held in shared memory. The 48-96 MB int64 index is therefore never written to HBM
and never read back -- on the 5090 that write costs 0.04-0.09 ms (measured; see the
`spill` control in the .cu), and it is the single largest remaining cost for 0127, whose
index is 96 MB = exactly this GPU's L2.

Routing regime is dispatched from the model, because the winner flips and neither choice
is safe everywhere (standalone routing, vs the native kernel):

    0126 nap7 : v1 1.30x   v2 0.70x  <- v2 REGRESSES
    0127 nap7 : v1 1.08x   v2 0.74x
    0128 nap8 : v1 1.51x   v2 1.78x  <- v2 wins

so v1 (row-major z, table-inner) for nap <= 7 and v2 (column-major z, token-inner,
bank-conflict-free) for nap >= 8. Inside the fused kernel the difference is much smaller
than standalone (0126: 0.0894 vs 0.0903) because once the index write is gone, routing is
no longer what the stage is bound on.

Measured on an RTX 5090, batch 48 x seq 512, trained checkpoints, routing+gather stage:

    model  native+Triton  native+bf16   fused bf16   vs native+bf16
    0126      0.1697        0.1213        0.0894        1.357x
    0127      0.3686        0.3239        0.1718        1.885x
    0128      0.2878        0.1958        0.0959        2.042x

FFN slot vs vanilla: 0.73 -> 0.53x, 1.23 -> 0.77x, 0.94 -> 0.55x. End-to-end vs vanilla:
0.93 -> 0.88x, 1.05 -> 0.93x, 0.98 -> 0.88x -- all three models beat vanilla end-to-end,
and 0127 crosses over for the first time.

NUMERICS: the fused fp32 table path is bit-exact against native routing + the Triton
gather. The fused bf16 path is bit-IDENTICAL (0.000e+00) to the committed non-fused
cuda-bf16 gather, so its accuracy cost is exactly the bf16 cost already measured on real
val data: +0.00014 / +0.00011 / +0.00007 bpb.
"""
import os

import torch

BLOCK_NS = (32, 64, 128)          # 128 is bf16-only (fp32 would need 1536 threads)
DEFAULT_BLOCK_N = 64              # best across every fused run on the 5090
PAD_TO = 64
MAX_NAP = 8                       # index is packed into a byte in shared
MAX_TABLE_DIM = 256

_ext = None
_error = None
_tried = False


def load():
    """Build/load the extension once. Returns the module, or None if it can't build."""
    global _ext, _error, _tried
    if _tried:
        return _ext
    _tried = True
    if not torch.cuda.is_available():
        _error = 'no CUDA device'
        return None
    try:
        from torch.utils.cpp_extension import load as _load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gather_fused.cu')
        # -std=c++20 on BOTH host and CUDA flags: at the default standard this torch
        # build fails compiling its own ATen/core/List_inl.h. Nothing here needs C++20.
        _ext = _load(name='hyperplane_gather_fused', sources=[src],
                     extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
                     extra_cflags=['-O3', '-std=c++20'], verbose=False)
    except Exception as e:
        _ext = None
        _error = f'{type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}' \
            if str(e).strip() else type(e).__name__
    return _ext


def available():
    """(bool, message) -- safe to call anywhere, never raises."""
    ext = load()
    return (ext is not None,
            'fused routing+gather extension ready' if ext is not None
            else f'fused routing+gather unavailable ({_error})')


def nap_of(lut):
    """Anchor pairs per table, derived from the anchor tensor."""
    return int(lut.soft_anchor_a_long.numel() // (lut.n_heads * lut.tables_per_head))


def use_v2(lut):
    """Routing regime: v2 (conflict-free, L1-bound case) at nap >= 8, else v1."""
    return nap_of(lut) >= 8


def supported(lut):
    """(bool, reason) -- the uint8-index-in-shared packing has hard limits."""
    nap = nap_of(lut)
    if nap > MAX_NAP:
        return False, f'nap={nap} > {MAX_NAP} (index would not fit a byte in shared)'
    td = int(lut.weights.shape[1])
    if td > MAX_TABLE_DIM:
        return False, f'table_dim={td} > {MAX_TABLE_DIM} (uint8 shared index)'
    if int(lut.weights.shape[2]) != 48:
        return False, f'row width {int(lut.weights.shape[2])} != 48'
    return True, ''


def prepare_table(weights, table_dtype='bf16'):
    """[T, R, 48] fp32 -> the padded [T, R, 64] table the kernel wants. Once, at patch."""
    if table_dtype not in ('fp32', 'bf16'):
        raise ValueError(f'table_dtype must be fp32 or bf16, got {table_dtype!r}')
    w = weights.detach()
    if table_dtype == 'bf16':
        w = w.to(torch.bfloat16)
    T, R, Dw = w.shape
    out = torch.zeros(T, R, PAD_TO, device=w.device, dtype=w.dtype)
    out[:, :, :Dw] = w
    return out.contiguous()


def fused(z, lut, table, block_n=None, v2=None):
    """One call: routing + gather. `z` is the compressed input [N, n_heads*48]."""
    ext = load()
    if ext is None:
        raise RuntimeError(f'fused gather unavailable: {_error}')
    return ext.fused(z.contiguous(), lut.soft_anchor_a_long.contiguous(),
                     lut.soft_anchor_b_long.contiguous(), table,
                     lut.n_heads, lut.tables_per_head, nap_of(lut),
                     block_n or DEFAULT_BLOCK_N,
                     use_v2(lut) if v2 is None else v2, False)


def patch(model, table_dtype='bf16', block_n=None, cast_out_to_input_dtype=True):
    """Route every FastMultiHeadLut through the fused kernel.

    This replaces routing AND gather, so unlike gather.patch()/gather_cuda.patch() the
    native bit-pack kernel is no longer called at all -- the routing happens inside this
    kernel, and is bit-exact against it. Returns the number of modules patched.

    The table is prepared once here and captured; patch last (after hybrid.apply).
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    ext = load()
    if ext is None:
        raise RuntimeError(f'fused gather unavailable: {_error}')
    n = 0
    for mod in model.modules():
        if not isinstance(mod, FastMultiHeadLut):
            continue
        ok, why = supported(mod)
        if not ok:
            raise RuntimeError(f'fused gather does not support this model: {why}')
        prepared = prepare_table(mod.weights.data, table_dtype)

        def fast(x, weights_compute, _m=mod, _w=prepared, _b=block_n):
            y = fused(x, _m, _w, _b)
            return y.to(x.dtype) if cast_out_to_input_dtype else y

        mod._hard_eval_native = fast
        n += 1
    return n


def _lut_input(lut, n_tokens):
    width = getattr(lut, '_fwd_input_dim', lut.input_dim)
    return torch.randn(n_tokens, width, device=lut.weights.device)


def check_table_precision(lut, n_tokens=48 * 512, block_n=None):
    """bf16 vs fp32 table, through the SAME fused kernel, on the gather output.

    As for cuda-bf16, the tolerance belongs here and not on logits: rounding the table
    moves logits ~1.5e-1 relative on random tokens while costing +0.0001 bpb on real data.

    Returns (rel_error, max_abs_diff, ref_scale).
    """
    z = _lut_input(lut, n_tokens)
    with torch.no_grad():
        a = fused(z, lut, prepare_table(lut.weights.data, 'fp32'), block_n)
        b = fused(z, lut, prepare_table(lut.weights.data, 'bf16'), block_n)
    scale = a.abs().max().item()
    dif = (b - a).abs().max().item()
    return (dif / scale if scale > 0 else float('inf')), dif, scale


def tune(lut, table_dtype='bf16', n_tokens=48 * 512, iters=30, reps=3):
    """Sweep BLOCK_N (and both routing regimes, to check the dispatch on new hardware).

    Returns (ms, block_n, v2).
    """
    import statistics
    import time

    z = _lut_input(lut, n_tokens)
    table = prepare_table(lut.weights.data, table_dtype)
    fp32 = (table_dtype == 'fp32')

    def run(bn, v2):
        with torch.no_grad():
            for _ in range(5):
                fused(z, lut, table, bn, v2)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fused(z, lut, table, bn, v2)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    best = None
    for bn in BLOCK_NS:
        if fp32 and bn == 128:
            continue                      # would need 1536 threads
        for v2 in (False, True):
            try:
                ms = statistics.median([run(bn, v2) for _ in range(reps)])
            except Exception:
                continue
            if best is None or ms < best[0]:
                best = (ms, bn, v2)
    return best
