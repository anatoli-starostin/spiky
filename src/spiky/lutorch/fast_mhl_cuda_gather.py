"""Fused CUDA routing+gather for FastMultiHeadLut -- hardware-gated, opt-in, never raises.

Ported from the RTX 5090 benchmark work in `experiments/ffn_replacement/` (see
`feature/ffn_replacement`, the paper "Replacing Transformer Feed-Forward Layers with
Lookup Tables"). Fuses the anchor-compare routing and the gather+sum into ONE kernel with
the index kept in shared memory, so the (48-96 MB) int64 routing index is never written to
or read back from HBM. Measured on an RTX 5090 (batch 48 x seq 512, trained checkpoints):
routing+gather stage 1.36-2.04x faster than native routing + Triton gather; FFN slot ends
up 1.3-1.9x faster than the vanilla dense FFN. **The identical kernel measured SLOWER than
the vanilla dense FFN on an H100** (routing is compute-bound there, not memory-bound, and
H100's dense matmuls are already near tensor-core peak) -- this is a hardware-dependent
result, not a universal one, which is the reason for the dispatch below rather than a
plain default-on patch.

NUMERICS: bit-exact against native routing + the (portable) Triton gather when the table is
kept fp32. The bf16-table path used here (`table_dtype="bf16"`, the default and the only
one measured as a net win) is bit-IDENTICAL to the already-shipped `cuda-bf16` path's
output and costs the same measured +0.00007..+0.00014 val bpb on trained checkpoints --
not a new source of error, the existing one.

## Usage

    from spiky.lutorch.fast_mhl_cuda_gather import patch, is_5090_class_gpu

    patch(model)                 # mode="auto" (default): fused kernel iff is_5090_class_gpu(),
                                  # else a no-op -- the model's existing, exact gather is untouched.
    patch(model, mode="force")   # always attempt; raises if unsupported/unavailable.
    patch(model, mode="off")     # explicit no-op (documents intent at the call site).

**Default is opt-in, not automatic.** `FastMultiHeadLut`/`CompressionMultiHeadLUT` do NOT
call this on your behalf -- every existing model, on every machine, keeps its current
exact behavior unless a caller explicitly imports and calls `patch()`. `mode="auto"` is
the hardware dispatch: on 5090-class GPUs it patches in the faster/approximate kernel: on
everything else (H100, older/other cards, CPU, or any future/unrecognized device) it does
nothing, leaving the model on its existing default gather path, which is unconditionally
correct and needs no dispatch table of its own.

## The H100 prototype kernels are shipped but PASSIVE

`csrc/h100_prototypes/` carries the three H100 kernels from the same optimization sweep
(`gather_fused_v2_h100.cu`, `route_v2_h100.cu`, `route_shared_h100.cu`). They are
source-only: **never compiled, never imported, never reachable from the dispatch.** They
are here so the H100 work is preserved in-tree and available for later experimentation on
the nebius H100 box -- not because any device switches to them. `H100_PROTOTYPE_KERNELS`
below maps their names to paths for inspection/manual builds; reading it compiles nothing.

On H100 the measured result is that this kernel family is *slower* than the vanilla dense
FFN, so there is no device for which enabling them is currently the right default, and they
were never wired into a callable Python API on the benchmark side either. Activating one
would be real integration work, not a dispatch-table entry. See `csrc/README.md`.
"""
import os

import torch

BLOCK_NS = (32, 64, 128)          # 128 is bf16-only (fp32 would need 1536 threads)
DEFAULT_BLOCK_N = 64              # best across every fused run on the 5090
PAD_TO = 64
MAX_NAP = 8                       # index is packed into a byte in shared
MAX_TABLE_DIM = 256

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc')

# PASSIVE. Paths only -- nothing here is compiled, loaded, or dispatched to. Present so the
# H100 sweep's kernels are inspectable/buildable by hand for future nebius work; see the
# module docstring and csrc/README.md for why they are inert rather than wired up.
H100_PROTOTYPE_KERNELS = {
    'gather_fused_v2_h100': os.path.join(_CSRC, 'h100_prototypes',
                                         'gather_fused_v2_h100.cu'),
    'route_v2_h100': os.path.join(_CSRC, 'h100_prototypes', 'route_v2_h100.cu'),
    'route_shared_h100': os.path.join(_CSRC, 'h100_prototypes', 'route_shared_h100.cu'),
}

_ext = None
_error = None
_tried = False


def is_5090_class_gpu(device=None) -> bool:
    """True iff `device` (default: current CUDA device) reports compute capability 12.x --
    Blackwell, sm_120, what the RTX 5090 identifies as. This is the ONLY class of hardware
    this kernel was benchmarked as a genuine win on. H100 is sm_90 (major 9) and measured
    SLOWER with this identical kernel; any other/future/unrecognized device also returns
    False here by construction, so `patch(..., mode="auto")` only ever touches hardware
    this was actually validated on.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major == 12


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
        # ONLY gather_fused.cu is ever built. csrc/h100_prototypes/ is deliberately not
        # referenced here -- see H100_PROTOTYPE_KERNELS and the module docstring.
        src = os.path.join(_CSRC, 'gather_fused.cu')
        # -std=c++20 on BOTH host and CUDA flags: at the default standard this torch
        # build fails compiling its own ATen/core/List_inl.h. Nothing here needs C++20.
        _ext = _load(name='spiky_lutorch_gather_fused', sources=[src],
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


def patch(model, mode='auto', table_dtype='bf16', block_n=None,
          cast_out_to_input_dtype=True):
    """Route every `FastMultiHeadLut` submodule through the fused CUDA kernel -- gated by
    `mode`:

      "auto"  (default) -- patch only if `is_5090_class_gpu()`; on any other hardware this
              is a documented no-op, so calling it unconditionally in a training/inference
              script is always safe. Returns 0 when it does nothing.
      "force" -- always attempt, regardless of detected hardware. Raises if the kernel
                 can't build or a module isn't `supported()` -- for benchmarking/testing on
                 hardware this wasn't validated on, not for casual production use.
      "off"   -- explicit no-op; documents at the call site that patching was considered
                 and deliberately skipped. Returns 0.

    This replaces routing AND gather, so unlike the portable Triton path the native
    bit-pack kernel is no longer called at all for a patched module -- the routing happens
    inside this kernel, and is bit-exact against it. Returns the number of modules patched.

    The table is prepared once here and captured in a closure; call this last, after any
    dtype/storage conversion (e.g. hybrid-precision setup) has already run.
    """
    if mode not in ('auto', 'force', 'off'):
        raise ValueError(f"mode must be 'auto', 'force', or 'off', got {mode!r}")
    if mode == 'off':
        return 0
    if mode == 'auto' and not is_5090_class_gpu():
        return 0

    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    ext = load()
    if ext is None:
        if mode == 'force':
            raise RuntimeError(f'fused gather unavailable: {_error}')
        return 0

    n = 0
    for mod in model.modules():
        if not isinstance(mod, FastMultiHeadLut):
            continue
        ok, why = supported(mod)
        if not ok:
            if mode == 'force':
                raise RuntimeError(f'fused gather does not support this model: {why}')
            continue
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

    Rounding the table moves logits ~1.5e-1 relative on random tokens while costing only
    +0.0001 bpb on real data -- the tolerance belongs on this gather-output comparison, not
    on logits from an out-of-distribution probe. Returns (rel_error, max_abs_diff, ref_scale).
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
