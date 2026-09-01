"""Optional hand-written CUDA gather+sum fast path (see gather_cuda.cu).

This is OPT-IN and NOT the default. It needs a CUDA toolchain (nvcc) to build a torch
extension at first use; the Triton kernel in gather.py stays the portable fallback and
remains the default, so a machine without a compiler still runs the whole harness.
Call `available()` to find out which path you are on, and never assume: `load()`
returns None and records why rather than raising.

Two table precisions:

  'fp32'  bit-exact against the Triton kernel; the win is software pipelining alone.
  'bf16'  APPROXIMATE -- table values rounded to bf16, sum still accumulated in fp32.
          Faster (half the 32-byte sectors per row) but it must never be run through a
          bit-exact assertion. Use check_table_precision() to gate it on a tolerance.

Measured on an RTX 5090, batch 48 x seq 512, trained checkpoints, gather stage only:

    model    Triton     cuda/fp32   cuda/bf16
    0126     0.1247     0.1064      0.0781     (1.36x over cuda/fp32)
    0127     0.3024     0.2621      0.2538     (1.03x)
    0128     0.1777     0.1300      0.0826     (1.57x)

and at FFN-slot level, bf16 vs cuda/fp32: 1.05x / 1.04x / 1.13x, which takes 0126 to
0.71x and 0128 to 0.92x of the vanilla slot. Cost of the bf16 rounding, measured as
real val_bpb on the nanochat val set rather than inferred: +0.00014 / +0.00011 /
+0.00007 bpb.

0127 gains least because its bottleneck is not the table: at tph=128 its int64 index
is 24576 x 512 x 8 B = 96 MB, exactly this GPU's L2, so it is index-bound and halving
table bytes cannot help it.
"""
import os

import torch

# (BLOCK_N, threads) the kernel is compiled for. The optimum moved when pipelining was
# added, so re-sweep with tune() on new hardware rather than inheriting these.
CFGS = ((64, 256), (128, 256), (128, 512), (256, 512))
DEFAULT_CFG = (256, 512)          # best on the 5090 for 0126/0128; 0127 prefers (64,256)

PAD_TO = 64                       # pad the 48-wide row; free, and keeps rows 32 B aligned

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
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gather_cuda.cu')
        # -std=c++20 on BOTH: with this torch build, compiling its headers at the
        # default standard fails inside ATen/core/List_inl.h ("need 'typename' before
        # ... dependent scope"). Nothing here needs C++20 itself.
        _ext = _load(name='hyperplane_gather_cuda', sources=[src],
                     extra_cuda_cflags=['-O3', '-std=c++20', '--use_fast_math'],
                     extra_cflags=['-O3', '-std=c++20'], verbose=False)
    except Exception as e:                                   # nvcc missing, arch, perms
        _ext = None
        _error = f'{type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}' \
            if str(e).strip() else type(e).__name__
    return _ext


def available():
    """(bool, message) -- safe to call anywhere, never raises."""
    ext = load()
    return (ext is not None,
            'CUDA gather extension ready' if ext is not None
            else f'CUDA gather unavailable ({_error})')


def prepare_table(weights, table_dtype='bf16'):
    """[T, R, 48] fp32 -> the padded [T, R, 64] table the kernel wants.

    Done ONCE at patch time, not per forward: casting and padding every call would
    cost more than the kernel saves.
    """
    if table_dtype not in ('fp32', 'bf16'):
        raise ValueError(f'table_dtype must be fp32 or bf16, got {table_dtype!r}')
    w = weights.detach()
    if table_dtype == 'bf16':
        w = w.to(torch.bfloat16)
    T, R, Dw = w.shape
    if Dw != 48:
        raise ValueError(f'expected a 48-wide row, got {Dw}')
    out = torch.zeros(T, R, PAD_TO, device=w.device, dtype=w.dtype)
    out[:, :, :Dw] = w
    return out.contiguous()


def gather_sum(table, index, n_heads, tph, cfg=None):
    """Dispatch on the prepared table's dtype. table from prepare_table()."""
    ext = load()
    if ext is None:
        raise RuntimeError(f'CUDA gather unavailable: {_error}')
    bn, nt = cfg or DEFAULT_CFG
    fn = ext.gather_sum_bf16 if table.dtype == torch.bfloat16 else ext.gather_sum_fp32
    return fn(table, index, n_heads, tph, bn, nt)


def patch(model, table_dtype='bf16', cfg=None, cast_out_to_input_dtype=True):
    """Route every FastMultiHeadLut's hard-eval gather through the CUDA kernel.

    Like gather.patch(), only the gather is replaced -- the native CUDA bit-pack
    kernel is kept for addressing. Returns the number of modules patched.

    The table is prepared once here and captured, so weight edits AFTER patching are
    not picked up; patch last (in particular, after hybrid.apply).
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    ext = load()
    if ext is None:
        raise RuntimeError(f'CUDA gather unavailable: {_error}')
    n = 0
    for mod in model.modules():
        if not isinstance(mod, FastMultiHeadLut):
            continue
        if mod._native_eval_msb is None:
            raise RuntimeError(
                'native bit-pack kernel unavailable (lutorch_cuda not built?); '
                'the CUDA gather needs its int64 index')
        prepared = prepare_table(mod.weights.data, table_dtype)

        def fast(x, weights_compute, _m=mod, _w=prepared, _c=cfg):
            index = _m._native_eval_msb(x, _m.soft_anchor_a_long,
                                        _m.soft_anchor_b_long, 0.0, 256)
            y = gather_sum(_w, index, _m.n_heads, _m.tables_per_head, _c)
            return y.to(x.dtype) if cast_out_to_input_dtype else y

        mod._hard_eval_native = fast
        n += 1
    return n


def _lut_input(lut, n_tokens, device):
    width = getattr(lut, '_fwd_input_dim', lut.input_dim)
    return torch.randn(n_tokens, width, device=device)


def check_table_precision(lut, n_tokens=48 * 512, cfg=None):
    """Relative error of the bf16 table against the fp32 one, on the GATHER output.

    This is the right place for the tolerance, and the logit level is NOT: rounding the
    table shifts logits by ~1.5e-1 relative on random tokens, which would fail any
    sane logit tolerance, while the real cost on val data is +0.0001 bpb. Measuring
    where the approximation is introduced keeps the gate meaningful.

    Returns (rel_error, max_abs_diff, ref_scale).
    """
    dev = lut.weights.device
    x = _lut_input(lut, n_tokens, dev)
    with torch.no_grad():
        idx = lut._native_eval_msb(x, lut.soft_anchor_a_long, lut.soft_anchor_b_long,
                                   0.0, 256)
        a = gather_sum(prepare_table(lut.weights.data, 'fp32'), idx,
                       lut.n_heads, lut.tables_per_head, cfg)
        b = gather_sum(prepare_table(lut.weights.data, 'bf16'), idx,
                       lut.n_heads, lut.tables_per_head, cfg)
    scale = a.abs().max().item()
    dif = (b - a).abs().max().item()
    return (dif / scale if scale > 0 else float('inf')), dif, scale


def tune(lut, table_dtype='bf16', n_tokens=48 * 512, iters=30, reps=3):
    """Sweep (BLOCK_N, threads) on this GPU. Returns (ms, block_n, threads).

    Tune on the REAL index: an earlier version tuned against a zeros index, so every
    gather hit row 0, cache behaviour was unrepresentative and it picked a config 23%
    slower than the true best.
    """
    import statistics
    import time

    dev = lut.weights.device
    x = _lut_input(lut, n_tokens, dev)
    with torch.no_grad():
        idx = lut._native_eval_msb(x, lut.soft_anchor_a_long, lut.soft_anchor_b_long,
                                   0.0, 256)
    table = prepare_table(lut.weights.data, table_dtype)

    def run(cfg):
        with torch.no_grad():
            for _ in range(5):
                gather_sum(table, idx, lut.n_heads, lut.tables_per_head, cfg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                gather_sum(table, idx, lut.n_heads, lut.tables_per_head, cfg)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    best = None
    for cfg in CFGS:
        try:
            ms = statistics.median([run(cfg) for _ in range(reps)])
        except Exception:
            continue
        if best is None or ms < best[0]:
            best = (ms, cfg[0], cfg[1])
    return best
