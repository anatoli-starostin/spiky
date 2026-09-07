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

    @triton.jit
    def _blend_bwd_gp_kernel(idx0_ptr, idx1_ptr, g_ptr, flat_ptr, gp0_ptr, gp1_ptr,
                             T: tl.constexpr, O: tl.constexpr, BLOCK_O: tl.constexpr):
        # grad_psw_k[n,t] = <g[n,:], flat[idx_k[n,t],:]>  (gather 2 rows + dot; no materialise)
        n = tl.program_id(0)
        o = tl.arange(0, BLOCK_O); omask = o < O
        g = tl.load(g_ptr + n * O + o, mask=omask, other=0.0)
        base = n * T
        for t in range(T):
            p = base + t
            i0 = tl.load(idx0_ptr + p); i1 = tl.load(idx1_ptr + p)
            r0 = tl.load(flat_ptr + i0 * O + o, mask=omask, other=0.0)
            r1 = tl.load(flat_ptr + i1 * O + o, mask=omask, other=0.0)
            tl.store(gp0_ptr + p, tl.sum(g * r0))
            tl.store(gp1_ptr + p, tl.sum(g * r1))

    @triton.jit
    def _blend_bwd_gflat_kernel(idx0_ptr, idx1_ptr, c0_ptr, c1_ptr, g_ptr, gflat_ptr,
                                T: tl.constexpr, O: tl.constexpr, BLOCK_O: tl.constexpr):
        # grad_flat[idx0] += c0*g ; grad_flat[idx1] += c1*g   (atomic scatter-add)
        n = tl.program_id(0)
        o = tl.arange(0, BLOCK_O); omask = o < O
        g = tl.load(g_ptr + n * O + o, mask=omask, other=0.0)
        base = n * T
        for t in range(T):
            p = base + t
            i0 = tl.load(idx0_ptr + p); i1 = tl.load(idx1_ptr + p)
            c0 = tl.load(c0_ptr + p); c1 = tl.load(c1_ptr + p)
            tl.atomic_add(gflat_ptr + i0 * O + o, c0 * g, mask=omask)
            tl.atomic_add(gflat_ptr + i1 * O + o, c1 * g, mask=omask)


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


class LightBlendN2Fn(torch.autograd.Function):
    """autograd.Function for the n==2 blend: Triton fwd + Triton bwd. Grads to d (argmin/mv
    path only, matching eager _blend_bag), flat (tables), score, log_tau. index/mj/sign are
    non-differentiable. Not bit-exact (sigmoid vs eager softmax; atomic scatter order)."""

    @staticmethod
    def forward(ctx, d, flat, score, log_tau, idx0_abs, idx1_abs, mj, sign_dmj, n_bags, bag):
        N, T, O = n_bags, bag, flat.shape[1]
        tau = log_tau.exp()
        mvf = d.abs().gather(-1, mj.unsqueeze(-1)).squeeze(-1).reshape(N, T).to(torch.float32).contiguous()
        scoref = score.reshape(N, T).to(torch.float32).contiguous()
        w1f = torch.sigmoid(-2.0 * mvf / tau); w0f = 1.0 - w1f
        i0 = idx0_abs.reshape(N, T).to(torch.int32).contiguous()
        i1 = idx1_abs.reshape(N, T).to(torch.int32).contiguous()
        flat_c = flat.contiguous()
        y = torch.empty(N, O, device=d.device, dtype=torch.float32)
        BLOCK_O = 1 << (O - 1).bit_length()
        _blend_fwd_kernel[(N,)](i0, i1, mvf, scoref, flat_c, y, T=T, O=O,
                                tau=float(tau.item()), BLOCK_O=BLOCK_O)
        ctx.save_for_backward(flat_c, scoref, w0f, w1f, mvf, i0, i1, mj, sign_dmj)
        ctx.N, ctx.T, ctx.O, ctx.BLOCK_O = N, T, O, BLOCK_O
        ctx.tau = float(tau.item()); ctx.dshape = tuple(d.shape); ctx.sshape = tuple(score.shape)
        return y.to(flat.dtype)

    @staticmethod
    def backward(ctx, g):
        flat_c, scoref, w0f, w1f, mvf, i0, i1, mj, sign_dmj = ctx.saved_tensors
        N, T, O, BLOCK_O, tau = ctx.N, ctx.T, ctx.O, ctx.BLOCK_O, ctx.tau
        g = g.reshape(N, O).to(torch.float32).contiguous()
        gp0 = torch.empty(N, T, device=g.device, dtype=torch.float32)
        gp1 = torch.empty(N, T, device=g.device, dtype=torch.float32)
        _blend_bwd_gp_kernel[(N,)](i0, i1, g, flat_c, gp0, gp1, T=T, O=O, BLOCK_O=BLOCK_O)
        grad_score = (gp0 * w0f + gp1 * w1f)
        grad_w0 = gp0 * scoref; grad_w1 = gp1 * scoref
        grad_mv = (2.0 / tau * w0f * w1f) * (grad_w0 - grad_w1)
        grad_tau = ((grad_w1 - grad_w0) * (w0f * w1f * 2.0 * mvf / (tau * tau))).sum()
        grad_log_tau = (grad_tau * tau).to(flat_c.dtype).reshape(())
        gflat = torch.zeros_like(flat_c)
        c0 = (scoref * w0f).contiguous(); c1 = (scoref * w1f).contiguous()
        _blend_bwd_gflat_kernel[(N,)](i0, i1, c0, c1, g, gflat, T=T, O=O, BLOCK_O=BLOCK_O)
        grad_d = torch.zeros(ctx.dshape, device=g.device, dtype=flat_c.dtype)
        gd_at = (grad_mv.reshape(ctx.dshape[:-1]) * sign_dmj).unsqueeze(-1).to(flat_c.dtype)
        grad_d.scatter_(-1, mj.unsqueeze(-1), gd_at)
        return (grad_d, gflat.to(flat_c.dtype), grad_score.reshape(ctx.sshape).to(flat_c.dtype),
                grad_log_tau, None, None, None, None, None, None)


def blend_n2(d, index, offset, flat, score, log_tau, n_bags, bag):
    """Fused n==2 blend (fwd+bwd) as an autograd.Function. Falls back to None if Triton is
    unavailable/non-cuda (caller uses the eager path)."""
    if not _HAVE_TRITON or not d.is_cuda:
        return None
    NAP = d.shape[-1]
    powers = (2 ** torch.arange(NAP - 1, -1, -1, device=d.device, dtype=torch.int64))
    m = d.abs()
    _mv, mj = m.min(dim=-1)                                   # [B,H,T]
    bits = (d.detach() > 0).to(torch.int64)
    pw = powers[mj]
    bsel = torch.gather(bits, -1, mj.unsqueeze(-1)).squeeze(-1)
    idx1_rel = index + pw * (1 - 2 * bsel)
    idx0_abs = index + offset
    idx1_abs = idx1_rel + offset
    sign_dmj = d.gather(-1, mj.unsqueeze(-1)).squeeze(-1).sign()
    return LightBlendN2Fn.apply(d, flat, score, log_tau, idx0_abs, idx1_abs, mj, sign_dmj,
                                n_bags, bag)
