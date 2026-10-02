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
the +du-to-a half, formed from the same per-table gradient carriers the backward already
produces (``_single_input_grad``), so no large ``W[c]-W[c']`` tensor is materialised. That
+du-to-a scatter is itself a dedicated one-launch CUDA kernel
(``native/lutorch/single_anchor_input_grad.cu``, JIT-built via ``cpp_extension.load`` — it is
standalone and does NOT rebuild the shared ``lutorch_cuda`` extension), mirroring the pairs
path's single fused input-grad kernel so the single backward is not launch-bound at small
batch. If that kernel cannot be built/loaded, the eager body is used (identical result).
"""
from __future__ import annotations

import os

import torch

from ._fused_ops import _acc_dtype

_THREADS = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))
_MANAGER = None
_TRIED = False

# The single-anchor input gradient is a short elementwise chain + one scatter_add. In eager
# PyTorch that is ~10 tiny kernel launches and, at small batch, the backward is launch-bound
# there (measured ~0.075 ms vs the pairs path's single fused input-grad kernel at ~0.018 ms).
# A dedicated one-launch CUDA kernel (native/lutorch/single_anchor_input_grad.cu, JIT-built via
# cpp_extension.load — standalone, it does NOT touch the shared lutorch_cuda extension) matches
# the pairs path, so single-anchor is <= pairs at every batch. If it cannot be built/loaded, or
# on CPU, the eager body is used (identical result, just the extra launches).
_SINGLE_IG_EXT = None
_SINGLE_IG_TRIED = False


def _single_ig_ext():
    """Lazily JIT-build/load the single-anchor input-grad kernel; None if unavailable."""
    global _SINGLE_IG_EXT, _SINGLE_IG_TRIED
    if _SINGLE_IG_TRIED:
        return _SINGLE_IG_EXT
    _SINGLE_IG_TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1":
        return None
    try:
        from pathlib import Path

        from torch.utils.cpp_extension import load

        # .../src/spiky/lutorch_ex/cartridges/_native_ops.py -> repo root is parents[4].
        repo_root = Path(__file__).resolve().parents[4]
        src = repo_root / "native" / "lutorch" / "single_anchor_input_grad.cu"
        _SINGLE_IG_EXT = load(name="lutorch_ex_single_anchor_ig", sources=[str(src)], verbose=False)
    except Exception:
        _SINGLE_IG_EXT = None
    return _SINGLE_IG_EXT


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


def _single_ig_body(gm, ga, d, ag, width):
    """du = (gm - ga) * 0.5*sign(d)/(1+|d|)^2, scattered +du to coordinate a. [B, nt] -> [B, width]."""
    coeff = 0.5 * torch.sign(d) / (1.0 + d.abs()) ** 2
    du = (gm - ga) * coeff                            # [B, nt]
    grad_zf = torch.zeros(gm.shape[0], width, device=du.device, dtype=du.dtype)
    grad_zf.scatter_add_(1, ag, du)
    return grad_zf


def _single_input_grad(gc_main, gc_alt, delta, a_glob, B, G, d_in, nt):
    """Single-anchor input gradient (anchor vs zero): scatter +du to coordinate a only.

    Reuses the native forward + weight-grad kernels unchanged (index-only, so correct for
    single mode) and the per-table gradient carriers they already produce: ``gc_main`` =
    ``grad . W[c]`` and ``gc_alt`` = ``grad . W[c']`` (both scalars per (b, table)). The pairs
    kernel forms ``du = (gc_main - gc_alt) * coeff`` and scatters +du to a, -du to b; the
    single case keeps only the +du-to-a half. ``coeff = 0.5*sign(delta)/(1+|delta|)^2`` is the
    same rational-uncertainty surrogate — so this is bit-equivalent to the pure single
    cartridge — and it costs O(B*nt): no [B, nt, d_out] gather of W. Returns ``[B, G, d_in]``.

    On CUDA the elementwise+scatter is a dedicated one-launch kernel (see _single_ig_ext), so
    the single-anchor backward is not launch-bound at small batch — it matches the pairs path's
    single fused input-grad kernel. On CPU, or if the kernel can't be built, the eager body is
    used (identical result). Returns ``[B, G, d_in]``.
    """
    gm = gc_main.reshape(B, nt).contiguous()
    ga = gc_alt.reshape(B, nt).contiguous()
    d = delta.reshape(B, nt).contiguous()             # signed deciding margin z[a]
    ag = a_glob.reshape(B, nt).contiguous()
    ext = _single_ig_ext() if gm.is_cuda else None
    if ext is not None:
        grad_zf = ext.single_anchor_input_grad(gm, ga, d, ag, G * d_in, _THREADS)
    else:
        grad_zf = _single_ig_body(gm, ga, d, ag, G * d_in)
    return grad_zf.view(B, G, d_in)


def _flatten(weights, c, c_alt, u_signed_star):
    """Common reshapes to gen-1 flat layout. Returns (W, li, lai, lad, nt, B, G, tph, K, d_out).

    lad (the signed deciding margin) is cast to the weight dtype: the native kernels dispatch
    on the weight dtype, so the margin must match it (addressing was computed in fp32 upstream;
    casting the magnitude to bf16 here costs only bf16 round-off, and the kernels re-accumulate
    in fp32 internally).
    """
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    nt = G * tph
    W = weights.reshape(nt, K, d_out).contiguous()
    li = c.reshape(B, nt).contiguous()
    lai = c_alt.reshape(B, nt, 1).contiguous()
    lad = u_signed_star.reshape(B, nt, 1).to(weights.dtype).contiguous()
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
        grp = out.reshape(B, G, tph, d_out).sum(2, dtype=_acc_dtype(out.dtype)).to(weights.dtype)
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        # z values are unused by the input-grad kernels (only its shape / dtype); keep it in the
        # weight dtype so the anchor kernel's dtype checks pass. grad_z is cast back to z's dtype.
        ctx.save_for_backward(W, li, lai, lad, tif, mw, aw,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in).to(weights.dtype))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        ctx.z_dtype = z.dtype
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
        return wgrad.reshape(G, tph, K, d_out), xg.to(ctx.z_dtype), None, None, None, None, None, None


class NativeHard(torch.autograd.Function):
    """Hard value (sum_t W[c_t]) with the native na1 NONSMOOTH backward (weight-grad hard,
    input-grad via the uncertainty carriers)."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single):
        mgr = native_manager()
        W, li, lai, lad, nt, B, G, tph, K, d_out = _flatten(weights, c, c_alt, u_signed_star)
        d_in = z.shape[2]
        tif = _table_indices(B, nt, z.device)
        # Hard value: sum_t W[c_t]. lutorch_cuda has no nonsmooth forward, so gather+sum here
        # (fp32-accumulated for bf16/fp16, cast back to the weight dtype).
        val = (W[tif.reshape(B, nt), li].reshape(B, G, tph, d_out)
               .sum(2, dtype=_acc_dtype(W.dtype)).to(weights.dtype))
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        ctx.save_for_backward(W, li, lai, lad, tif,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in).to(weights.dtype))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        ctx.z_dtype = z.dtype
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
        return wgrad.reshape(G, tph, K, d_out), xg.to(ctx.z_dtype), None, None, None, None, None, None
