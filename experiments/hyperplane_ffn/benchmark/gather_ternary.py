"""Gather patch for the ternary hyperplane family (exp_g_0053 and friends).

TernaryHyperplaneMultiHeadLUT is NOT a FastMultiHeadLut subclass, and its addressing
is a dense GEMM -- x @ q.T with q in {-1, 0, +1} divided by the projection divisor --
rather than an anchor-pair bit-pack. So it needs its own patch: keep the GEMM, and
replace only the embedding_bag gather that follows it. `gather.patch()` will silently
match nothing on this family, which is why this module exists as a separate entry
point and why run_bench dispatches on the family rather than assuming.

TWO THINGS THAT ARE EASY TO GET WRONG HERE

1. The addressing GEMM must stay COMPILED TOGETHER with the sign-pack. An eager
   sign-pack over the [N, n_tables, nap] int64 intermediate costs more than the GEMM
   it follows; a first version of this patch left it eager and came out slower than
   the model it was meant to speed up.

2. addr_dtype='fp32' is BIT-EXACT against the shipped model. addr_dtype='bf16' is
   faster but NOT exact, and cannot be: near a_i ~ 0 the bf16 rounding flips
   1[a_i > 0] and selects a DIFFERENT table row. That is a discrete change in output,
   not a floating-point tolerance, so do not report it as a free speedup.
"""
import torch

from gather import gather_sum


@torch.compile(dynamic=True)
def _addr_signpack(x, w2, bias, powers, n_tables, nap, dt):
    a = torch.matmul(x.to(dt), w2.t()).view(x.shape[0], n_tables, nap)
    a = a.float() + bias
    return ((a > 0).to(torch.int64) * powers).sum(dim=-1)


def ternary_modules(model):
    from spiky.lutorch.ternary_hyperplane_multi_head_lut import (
        TernaryHyperplaneMultiHeadLUT)
    return [m for m in model.modules()
            if isinstance(m, TernaryHyperplaneMultiHeadLUT)]


def patch(model, addr_dtype='fp32'):
    """Route each ternary LUT's gather through the Triton kernel. Returns the count.

    addr_dtype: 'fp32' (bit-exact, the default) or 'bf16' (faster, not exact).
    """
    if addr_dtype not in ('fp32', 'bf16'):
        raise ValueError("addr_dtype must be 'fp32' or 'bf16'")
    dt = torch.float32 if addr_dtype == 'fp32' else torch.bfloat16
    n = 0
    for l in ternary_modules(model):
        q = l.hard_routing_weight().detach()
        n_tables, nap, _ = q.shape
        w2 = q.reshape(n_tables * nap, -1).contiguous().to(dt)
        bias = l.hyperplane_bias.detach().reshape(1, n_tables, nap).float()
        powers = l.soft_powers.view(1, 1, -1)
        W = l.weights.data.contiguous()
        heads, tph = l.n_heads, l.tables_per_head

        def fast(x, _w2=w2, _b=bias, _p=powers, _W=W, _h=heads, _t=tph,
                 _nt=n_tables, _na=nap, _dt=dt):
            idx = _addr_signpack(x, _w2, _b, _p, _nt, _na, _dt)
            # tables stay fp32 under hybrid-v2 while the model is bf16, so hand the
            # result back in x's dtype (a no-op in an all-fp32 model)
            return gather_sum(_W, idx, _h, _t).to(x.dtype)

        l._hard_eval = fast
        n += 1
    return n
