"""Fused Triton kernel for LightMultiHeadLUT's top-2 blended read-out (read_top_n==2).

DEV MODULE — not yet wired into the training path. Replaces the multi-kernel eager
`_blend_bag` (2-row embedding_bag + the sigmoid/score/blend elementwise) with a single pass
that gathers the two candidate rows, applies the softmax(≡sigmoid) routing weight and the
confidence score, and reduces over a head's tables — never materialising the
[., n_tables, O] rows.

Stage 1 = FORWARD only. The integer address work (argmin over the NAP margins, the
Hamming-1 neighbour via XOR) stays in torch (cheap, integer); the kernel fuses the
per-(bag,table) sigmoid weight + score multiply + the 2-row gather + the reduce over tables.

Math (matches light_multi_head_lut.py::_blend_bag at n==2):
    mv     = min_j |d_j|                       (the least-confident anchor margin)
    w1     = sigmoid(-2*mv/tau),  w0 = 1 - w1   (== softmax([0, -2*mv/tau]))
    y[bag] = sum_t score[bag,t] * ( w0*T[idx0[bag,t]] + w1*T[idx1[bag,t]] )
where idx0 is the argmax cell and idx1 its argmin-|d| single-bit flip.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:
    @triton.jit
    def _blend_fwd_kernel(idx0_ptr, idx1_ptr, mv_ptr, score_ptr, flat_ptr, y_ptr,
                          T: tl.constexpr, O: tl.constexpr, tau,
                          BLOCK_O: tl.constexpr):
        n = tl.program_id(0)                       # bag index in [0, N)
        o = tl.arange(0, BLOCK_O)
        omask = o < O
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        base = n * T
        for t in range(T):
            p = base + t
            i0 = tl.load(idx0_ptr + p)
            i1 = tl.load(idx1_ptr + p)
            mv = tl.load(mv_ptr + p)
            s = tl.load(score_ptr + p)
            w1 = 1.0 / (1.0 + tl.exp(2.0 * mv / tau))      # sigmoid(-2 mv / tau)
            w0 = 1.0 - w1
            r0 = tl.load(flat_ptr + i0 * O + o, mask=omask, other=0.0)
            r1 = tl.load(flat_ptr + i1 * O + o, mask=omask, other=0.0)
            acc += s * (w0 * r0 + w1 * r1)
        tl.store(y_ptr + n * O + o, acc, mask=omask)


def blend_fwd_triton(d, index, offset, flat, score, read_tau, n_bags, bag):
    """Drop-in FORWARD replacement for LightMultiHeadLUT._blend_bag (n==2). Returns [n_bags, O].
    Falls back to None if Triton is unavailable (caller should use the eager path)."""
    if not _HAVE_TRITON or not d.is_cuda:
        return None
    O = flat.shape[1]
    NAP = d.shape[-1]
    powers = (2 ** torch.arange(NAP - 1, -1, -1, device=d.device, dtype=torch.int64))
    m = d.abs()
    mv, mj = m.min(dim=-1, keepdim=True)
    bits = (d.detach() > 0).to(torch.int64)
    pw = powers[mj]
    bsel = torch.gather(bits, -1, mj)
    idx1_rel = (index.unsqueeze(-1) + pw * (1 - 2 * bsel)).squeeze(-1)
    # absolute rows: add the per-(head,table) offset (broadcasts [1,H,T] over [B,H,T])
    idx0_abs = (index + offset).reshape(n_bags, bag).to(torch.int32).contiguous()
    idx1_abs = (idx1_rel + offset).reshape(n_bags, bag).to(torch.int32).contiguous()
    mvf = mv.squeeze(-1).reshape(n_bags, bag).to(torch.float32).contiguous()
    scoref = score.reshape(n_bags, bag).to(torch.float32).contiguous()
    flat_c = flat.contiguous()
    y = torch.empty(n_bags, O, device=d.device, dtype=torch.float32)
    BLOCK_O = 1 << (O - 1).bit_length()            # next pow2 >= O
    tau = float(read_tau) if not torch.is_tensor(read_tau) else float(read_tau.item())
    _blend_fwd_kernel[(n_bags,)](idx0_abs, idx1_abs, mvf, scoref, flat_c, y,
                                 T=bag, O=O, tau=tau, BLOCK_O=BLOCK_O)
    return y.to(flat.dtype)
