"""Act 1 + Act 2 of the p2_int8 read, in plain PyTorch: activations -> the uint8 cells
tensor [N, H, T, 3] that the Act-3 gather kernel consumes.

This is a deliberate transcription of `p2::table_scalars`
(src/spiky/lutorch/csrc/pow2_scalars.cuh:34-70), written to be BIT-EXACT against it
rather than merely equivalent. The CUDA side is compiled `--fmad=false`, so the
transcription has to match not just the formulas but the association order and the
NaN behaviour:

  * S and ls are LEFT FOLDS, `((m0 + m1) + m2) + ...`, not `.sum()`, whose CUDA
    reduction is a tree and gives a different last bit.
  * `mj` is the FIRST argmin -- the C loop takes `if (p == 0 || m < mv)`, strictly less,
    so ties keep the earliest p. `torch.argmin` does not promise that on CUDA, so the
    scan is written out.
  * `std::fmin(0.f, x)` returns 0 for a NaN x (fmin ignores NaN); `torch.minimum`
    propagates it. Written as `where(x < 0, x, 0)`, which matches fmin for every input
    including NaN and -inf.
  * `skip = !(kr >= lo)` is the NaN-safe form the header relies on (S == 0 gives
    log2(0) = -inf, and a NaN anywhere makes the comparison false, hence skip). Written
    as `~(kr >= lo)`, not `kr < lo`.
  * every multiply-then-add is two separate torch kernels, so nothing contracts into an
    FMA, matching --fmad=false.

The repo already ships a torch producer (`pow2_int8.table_integers` ->
`pow2_read`), but its own header says it is NOT bit-identical to the CUDA definition
("torch.compile's float32 reduction order and libdevice differ",
pow2_scalars.cuh:7-9). gate_cells.py measures both against the kernel's own output.
"""
import math

import torch

LN2 = float(torch.tensor(0.6931471805599453, dtype=torch.float32))
DISCARD = 15
FIXED_POINT_SHIFT = 6


def margins(z, anchor_a, anchor_b):
    """z [N, H, din] fp32 -> d [N, H, T, nap], d_p = z[a_p] - z[b_p].

    Mirrors the kernel's phase 1 (`pow2_int8_read.cu:137`), which reads both operands out
    of the shared Z tile. anchor_a / anchor_b are [H, T, nap] column indices inside the
    head, exactly as the kernel's Aa / Ab.
    """
    N, H, din = z.shape
    T, nap = anchor_a.shape[1], anchor_a.shape[2]
    ia = anchor_a.reshape(1, H, T * nap).expand(N, H, T * nap)
    ib = anchor_b.reshape(1, H, T * nap).expand(N, H, T * nap)
    return (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(N, H, T, nap)


def table_scalars(d, tau, g, beta, gamma, lo, hi, Q):
    """d [..., nap] fp32 -> (c1, c2, sh) int64 tensors [...], the kernel's integers."""
    nap = d.shape[-1]
    f32 = torch.float32
    d = d.to(f32)
    m = d.abs()

    # c1: MSB-first sign pattern (pow2_scalars.cuh:41)
    c1 = torch.zeros(d.shape[:-1], dtype=torch.int64, device=d.device)
    for p in range(nap):
        c1 = c1 | ((d[..., p] > 0).to(torch.int64) << (nap - 1 - p))

    # S and ls: LEFT folds, and the first strict argmin (pow2_scalars.cuh:42-50)
    zero = torch.zeros((), dtype=f32, device=d.device)
    bt = torch.as_tensor(beta, dtype=f32, device=d.device)

    def logsig(x):
        # min(0, x) - log1p(exp(-|x|)), with fmin's NaN rule
        return torch.where(x < 0, x, zero) - torch.log1p(torch.exp(-x.abs()))

    S = m[..., 0]
    ls = logsig(bt * m[..., 0])
    mv = m[..., 0]
    mj = torch.zeros(d.shape[:-1], dtype=torch.int64, device=d.device)
    for p in range(1, nap):
        mp = m[..., p]
        S = S + mp
        ls = ls + logsig(bt * mp)
        better = mp < mv                       # strict: ties keep the earlier p
        mv = torch.where(better, mp, mv)
        mj = torch.where(better, torch.full_like(mj, p), mj)

    c2 = c1 ^ (torch.ones_like(c1) << (nap - 1 - mj))

    # q, c_q, k' (pow2_scalars.cuh:51-56)
    tau_t = torch.as_tensor(tau, dtype=f32, device=d.device)
    inv = torch.as_tensor(2.0, dtype=f32, device=d.device) / (tau_t * LN2)
    qf = torch.clamp(torch.floor(mv * inv + 0.5), 0.0, 64.0)
    q = qf.to(torch.int64)
    cq = torch.where(q < 8, torch.log2(1.0 + torch.ldexp(torch.ones_like(qf), -q.to(torch.int32))),
                     torch.zeros_like(qf))
    g_t = torch.as_tensor(g, dtype=f32, device=d.device)
    gam = torch.as_tensor(gamma, dtype=f32, device=d.device)
    kr = torch.floor(torch.log2(S) + (g_t + gam * ls) / LN2 - cq + 0.5)

    # skip / clamp / drop / the shift byte (pow2_scalars.cuh:57-68)
    skip = ~(kr >= float(lo))                  # NaN and -inf both land here
    kc = torch.where(skip, torch.full_like(kr, float(lo)),
                     torch.where(kr > float(hi), torch.full_like(kr, float(hi)), kr))
    kc = kc.to(torch.int64)
    drop = q > Q
    sh1 = torch.where(skip, torch.full_like(kc, DISCARD), kc + FIXED_POINT_SHIFT)
    sh2 = torch.where(skip | drop, torch.full_like(kc, DISCARD),
                      kc + FIXED_POINT_SHIFT - q)
    return c1, c2, sh1 | (sh2 << 4)


def cells(z, anchor_a, anchor_b, tau, g, beta, gamma, lo, hi, Q):
    """The full Act 1 + Act 2 producer: z [N, H, din] -> uint8 cells [N, H, T, 3].

    Layout is the kernel's: (c1, c2, sh1 | sh2 << 4), cell numbers LOCAL to their table
    (no table offset -- the kernel adds base + t * pitch itself, pow2_int8_read.cu:167).
    """
    d = margins(z, anchor_a, anchor_b)
    c1, c2, sh = table_scalars(d, tau, g, beta, gamma, lo, hi, Q)
    return torch.stack([c1, c2, sh], dim=-1).to(torch.uint8).contiguous()


def cells_compiled():
    """torch.compile'd producer. Compilation changes the reduction order torch uses, so
    the bit-exactness gate is run on BOTH this and the eager one."""
    return torch.compile(cells, dynamic=True)
