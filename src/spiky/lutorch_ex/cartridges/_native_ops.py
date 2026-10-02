"""Tier-2 native fast-path: reuse the existing ``lutorch_cuda`` kernels.

The gen-1 lutorch kernels (native/lutorch/lutorch.cu, built as the ``lutorch_cuda``
extension) already implement the manifesto primitives. We reuse them for the fused
cartridges with thin adapters for lutorch_ex's representation:

* MSB-first addressing: lprojection reads ``weights[table, index]`` and is agnostic to how
  the index was packed, so we pass OUR MSB indices and OUR weight table directly — no
  bit-reversal needed.
* weight layout [G, tph, K, d_out] -> gen-1's flat [n_tables=G*tph, K, d_out];
* per-head input: z [B, G, d_in] is flattened to [B, G*d_in] and anchor coords are shifted
  to global (g*d_in + local), so the native input-gradient scatter lands in the right head;
* n_alternatives = 1 (the two-cell blend / single neighbour).

The native path is gated on ``lutorch_cuda`` being importable (torch must be imported
first so libc10 is loaded; we also ctypes-preload torch's libs as a fallback). When it is
not available the cartridges fall back to the pure/tier-1 path.

Single-anchor mode (``anchor_mode == "single"``) reuses the native forward and weight-grad
kernels unchanged — they are index-only and so correct for single anchors — and only the
input gradient differs: instead of the pairs kernel's +du-to-a / -du-to-b scatter it keeps
the +du-to-a half, formed cheaply from the same per-table gradient carriers the backward
already produces (``_single_input_grad``). No pairs-specific CUDA kernel is bypassed and no
large ``W[c]-W[c']`` tensor is materialised.
"""
from __future__ import annotations

import os

import torch

_THREADS = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))
_MANAGER = None
_TRIED = False


def native_manager():
    """Return the lutorch_cuda LUTorchManager, or None if unavailable. Robust to the
    libc10.so load issue (torch-first import; ctypes-preload fallback)."""
    global _MANAGER, _TRIED
    if _TRIED:
        return _MANAGER
    _TRIED = True
    try:
        import torch  # noqa: F401  (ensures libc10.so is loaded into the process)
        import lutorch_cuda
        _MANAGER = lutorch_cuda.get_lutorch_manager()
        return _MANAGER
    except Exception:
        pass
    try:  # fallback: ctypes-preload torch's shared libs so lutorch_cuda's dlopen resolves
        import ctypes
        libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
        for lib in ("libc10.so", "libc10_cuda.so", "libtorch_cpu.so",
                    "libtorch_cuda.so", "libtorch.so"):
            p = os.path.join(libdir, lib)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        import lutorch_cuda
        _MANAGER = lutorch_cuda.get_lutorch_manager()
    except Exception:
        _MANAGER = None
    return _MANAGER


def native_available(device: torch.device) -> bool:
    return device.type == "cuda" and native_manager() is not None


def _table_indices(B: int, nt: int, device) -> torch.Tensor:
    return torch.arange(nt, device=device).view(1, nt).expand(B, nt).reshape(-1).contiguous()


def _single_input_grad(gc_main, gc_alt, delta, a_glob, B, G, d_in, nt):
    """Single-anchor input gradient (anchor vs zero): scatter +du to coordinate a only.

    Reuses the native forward + weight-grad kernels unchanged (index-only, so correct for
    single mode) and the per-table gradient carriers they already produce: ``gc_main`` =
    ``grad . W[c]`` and ``gc_alt`` = ``grad . W[c']`` (both scalars per (b, table)). The pairs
    kernel forms ``du = (gc_main - gc_alt) * coeff`` and scatters +du to a, -du to b; the
    single case keeps only the +du-to-a half. ``coeff = 0.5*sign(delta)/(1+|delta|)^2`` is the
    same rational-uncertainty surrogate — so this is bit-equivalent to the pure single
    cartridge — and it costs O(B*nt): no [B, nt, d_out] gather of W. Returns ``[B, G, d_in]``.
    """
    gm = gc_main.reshape(B, nt)
    ga = gc_alt.reshape(B, nt)
    d = delta.reshape(B, nt)                          # signed deciding margin z[a]
    coeff = 0.5 * torch.sign(d) / (1.0 + d.abs()) ** 2
    du = (gm - ga) * coeff                            # [B, nt]
    grad_zf = torch.zeros(B, G * d_in, device=du.device, dtype=du.dtype)
    grad_zf.scatter_add_(1, a_glob.reshape(B, nt), du)
    return grad_zf.view(B, G, d_in)


def _flatten(weights, c, c_alt, u_signed_star):
    """Common reshapes to gen-1 flat layout. Returns (W, li, lai, lad, nt, B, G, tph, K, d_out)."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    nt = G * tph
    W = weights.reshape(nt, K, d_out).contiguous()
    li = c.reshape(B, nt).contiguous()
    lai = c_alt.reshape(B, nt, 1).contiguous()
    lad = u_signed_star.reshape(B, nt, 1).contiguous()
    return W, li, lai, lad, nt, B, G, tph, K, d_out


class NativeSoft(torch.autograd.Function):
    """Soft blend via lutorch_cuda lprojection_forward_smooth (+ its na1 smooth backward)."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single):
        mgr = native_manager()
        W, li, lai, lad, nt, B, G, tph, K, d_out = _flatten(weights, c, c_alt, u_signed_star)
        d_in = z.shape[2]
        tif = _table_indices(B, nt, z.device)
        out, mw, aw = mgr.lprojection_forward_smooth(W, li, lai, lad, tif, tif, True, _THREADS)
        grp = out.reshape(B, G, tph, d_out).sum(2)
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        ctx.save_for_backward(W, li, lai, lad, tif, mw, aw,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        return grp

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, lad, tif, mw, aw, a_g, b_g, batch_off, z_flat = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.unsqueeze(2).expand(B, G, tph, d_out).reshape(B, nt, d_out).contiguous()
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_smooth(
            grad_pt, W, li, lai, tif, tif, mw.contiguous(), aw.contiguous(), _THREADS)
        if ctx.single:
            xg = _single_input_grad(gc_main, gc_alt, lad, a_g, B, G, d_in, nt)
        else:
            xg = mgr.anchor_pairs_lookup_backward_all(
                z_flat, a_g.reshape(-1).contiguous(), b_g.reshape(-1).contiguous(), lad, batch_off,
                gc_main.contiguous(), gc_alt.reshape(-1).contiguous(), True, _THREADS).view(B, G, d_in)
        return wgrad.reshape(G, tph, K, d_out), xg, None, None, None, None, None, None


class NativeHard(torch.autograd.Function):
    """Hard value (sum_t W[c_t]) with the native na1 NONSMOOTH backward (weight-grad hard,
    input-grad via the uncertainty carriers)."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single):
        mgr = native_manager()
        W, li, lai, lad, nt, B, G, tph, K, d_out = _flatten(weights, c, c_alt, u_signed_star)
        d_in = z.shape[2]
        tif = _table_indices(B, nt, z.device)
        # Hard value: sum_t W[c_t]. lutorch_cuda has no nonsmooth forward, so gather+sum here.
        val = W[tif.reshape(B, nt), li].reshape(B, G, tph, d_out).sum(2)
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        ctx.save_for_backward(W, li, lai, lad, tif,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        return val

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, lad, tif, a_g, b_g, batch_off, z_flat = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.unsqueeze(2).expand(B, G, tph, d_out).reshape(B, nt, d_out).contiguous()
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_nonsmooth(
            grad_pt, W, li, lai, tif, tif, _THREADS)
        if ctx.single:
            xg = _single_input_grad(gc_main, gc_alt, lad, a_g, B, G, d_in, nt)
        else:
            xg = mgr.anchor_pairs_lookup_backward_all(
                z_flat, a_g.reshape(-1).contiguous(), b_g.reshape(-1).contiguous(), lad, batch_off,
                gc_main.contiguous(), gc_alt.reshape(-1).contiguous(), True, _THREADS).view(B, G, d_in)
        return wgrad.reshape(G, tph, K, d_out), xg, None, None, None, None, None, None
