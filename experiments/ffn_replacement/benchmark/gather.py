"""Fused Triton gather+sum for the FastMultiHeadLut hard-eval path.

Replaces the reference path

    index = native_bit_pack(z)                    # [N, n_tables] int64
    flat  = (index + table_offset).reshape(-1)    # materialised, ~50 MB at batch 48
    out   = F.embedding_bag(flat, W.view(-1, D), offsets, mode='sum')

with one kernel computing

    out[n, h, :] = sum over the tph tables of head h of W[table, index[n, table], :]

The table offset is folded into the addressing so flat_indices is never built, and
the per-head sum accumulates in registers across the table loop.

Measured on an RTX 5090 at batch 48 x seq 512: 0.703 ms -> 0.135 ms, 5.2x, and
BIT-EXACT against embedding_bag (this is a different summation order only in the
sense that there isn't one -- the same rows are added in the same order).

The index is read as int64 straight from the native bit-pack kernel: converting it
to int32 first costs a full read+write pass, which is more than the wider loads cost
inside the kernel. Narrowing the index was measured and is a dead end -- re-confirmed
later in the hand-written CUDA kernel, where the real reason became clear: the index
has just been written by the anchor kernel and is still L2-resident, so its width is
not on the critical path. See the README.

A faster, optional CUDA version of this kernel lives in gather_cuda.py (software
pipelining, and a bf16 table worth 1.36-1.57x). It needs nvcc; THIS kernel stays the
default and the portable fallback.

D = n_outputs is typically 48, not a power of two, so lanes run at the next power of
two with a mask; that idle fraction is a known cost of the shape.
"""
import torch
import triton
import triton.language as tl

# Tuned on an RTX 5090 over BLOCK_N x num_warps x num_stages. Retune on new hardware
# with tune(), below; the defaults are a reasonable starting point elsewhere.
BLOCK_N, NUM_WARPS, NUM_STAGES = 128, 8, 1


@triton.jit
def _gather_sum_kernel(W, IDX, OUT, N,
                       TPH: tl.constexpr, TABLE_DIM: tl.constexpr, D: tl.constexpr,
                       H: tl.constexpr, NTAB: tl.constexpr,
                       BLOCK_N: tl.constexpr, BD: tl.constexpr):
    pid_n = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_d = tl.arange(0, BD)
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_N, BD), dtype=tl.float32)
    for t in range(TPH):
        table = h * TPH + t
        i = tl.load(IDX + offs_n * NTAB + table, mask=mask_n, other=0).to(tl.int32)
        p = W + table * (TABLE_DIM * D) + i[:, None] * D + offs_d[None, :]
        acc += tl.load(p, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    o = OUT + offs_n[:, None] * (H * D) + h * D + offs_d[None, :]
    tl.store(o, acc, mask=mask_n[:, None] & mask_d[None, :])


def gather_sum(weights, index, n_heads, tph,
               block_n=None, num_warps=None, num_stages=None):
    """weights [NTAB, TABLE_DIM, D], index [N, NTAB] -> [N, n_heads, D]."""
    NTAB, TABLE_DIM, D = weights.shape
    N = index.shape[0]
    out = torch.empty((N, n_heads, D), device=weights.device, dtype=weights.dtype)
    bn = block_n or BLOCK_N
    _gather_sum_kernel[(triton.cdiv(N, bn), n_heads)](
        weights, index, out, N,
        TPH=tph, TABLE_DIM=TABLE_DIM, D=D, H=n_heads, NTAB=NTAB,
        BLOCK_N=bn, BD=triton.next_power_of_2(D),
        num_warps=num_warps or NUM_WARPS, num_stages=num_stages or NUM_STAGES)
    return out


def patch(model, cast_out_to_input_dtype=True):
    """Route every FastMultiHeadLut's hard-eval gather through the Triton kernel.

    The native CUDA bit-pack kernel is KEPT for addressing -- it is a small part of
    the slot and a fused Triton addressing kernel measured much slower. Only the
    gather is replaced. Returns the number of modules patched.
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    n = 0
    for mod in model.modules():
        if not isinstance(mod, FastMultiHeadLut):
            continue
        if mod._native_eval_msb is None:
            raise RuntimeError(
                'native bit-pack kernel unavailable (lutorch_cuda not built?); '
                'the Triton gather needs its int64 index')

        def fast(x, weights_compute, _m=mod):
            index = _m._native_eval_msb(x, _m.soft_anchor_a_long,
                                        _m.soft_anchor_b_long, 0.0, 256)
            y = gather_sum(weights_compute.contiguous(), index,
                           _m.n_heads, _m.tables_per_head)
            # hybrid-v2 keeps tables fp32 while the surrounding model is bf16, so
            # hand the result back in x's dtype (a no-op in an all-fp32 model)
            return y.to(x.dtype) if cast_out_to_input_dtype else y

        mod._hard_eval_native = fast
        n += 1
    return n


def tune(weights, index, n_heads, tph, iters=30, reps=3):
    """Sweep BLOCK_N x warps x stages on this GPU. Returns (ms, block_n, warps, stages)."""
    import itertools
    import statistics
    import time

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    best = None
    for bn, nw, ns in itertools.product((32, 64, 128, 256), (2, 4, 8), (1, 2, 4)):
        try:
            ms = statistics.median(
                [bench(lambda: gather_sum(weights, index, n_heads, tph, bn, nw, ns))
                 for _ in range(reps)])
        except Exception:
            continue
        if best is None or ms < best[0]:
            best = (ms, bn, nw, ns)
    return best
