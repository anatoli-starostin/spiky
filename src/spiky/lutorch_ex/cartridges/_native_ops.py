"""Tier-2 native fast-path: the gen-1 lprojection / anchor_pairs CUDA kernels.

These kernels are now vendored co-located under ``cartridges/csrc/`` (lprojection.cu +
lprojection_py.cpp + common_misc.cpp) and JIT-built as the ``lutorch_ex_lprojection`` extension
(the PRIMARY path, see :func:`_lprojection_ext`); the prebuilt ``lutorch_cuda`` library is kept
only as a last-resort fallback for one release. They implement the manifesto primitives, reused
for the fused cartridges with thin adapters for lutorch_ex's representation:

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
(``cartridges/csrc/single_anchor_input_grad.cu``, JIT-built via ``cpp_extension.load`` — it is
standalone and does NOT rebuild the shared ``lutorch_cuda`` extension), mirroring the pairs
path's single fused input-grad kernel so the single backward is not launch-bound at small
batch. If that kernel cannot be built/loaded, the eager body is used (identical result).
"""
from __future__ import annotations

import os

import torch

from ._fused_ops import _acc_dtype, fused_hard_read

_THREADS = int(os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256"))
_MANAGER = None
_TRIED = False

# Co-located CUDA sources (vendored from native/lutorch so lutorch_ex is self-contained) and a stable,
# persistent extension cache so the ~117 KB lprojection nvcc compile is paid once across processes.
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions_lutorch_ex"))


def _pick_gpp():
    """A host compiler for nvcc --compiler-bindir (parity with native/lutorch/setup.py); None if none."""
    import shutil
    for c in ("g++-13", "g++-12", "g++-11", "g++-10", "g++-9", "g++-8", "g++"):
        p = shutil.which(c)
        if p:
            return p
    return None


_LPROJ_EXT = None
_LPROJ_TRIED = False


def _lprojection_ext():
    """Build/load the co-located ``lutorch_ex_lprojection`` JIT extension (the vendored lprojection /
    anchor_pairs kernels + LUTorchManager). Mirrors the ``lutorch_ex_pow2_int8_read`` setup: lazy, cached,
    device-gated, never raises -> ``None`` falls through to the prebuilt lib / pure-torch path. nvcc flags
    match native/lutorch/setup.py (-std=c++20, -O3, -lcuda, --compiler-bindir, -I cuda/include)."""
    global _LPROJ_EXT, _LPROJ_TRIED
    if _LPROJ_TRIED:
        return _LPROJ_EXT
    _LPROJ_TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1":
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        from torch.utils.cpp_extension import load
        std = os.environ.get("SPIKY_CXX_STD", "c++20")
        # Mirror the working pow2_int8 build (and the prebuilt lutorch_cuda's semantics): cpp_extension
        # already supplies the correct CUDA_HOME include paths and a compatible host compiler, so we do
        # NOT hardcode -I/usr/local/cuda/include or --compiler-bindir (a stray -I mixes toolkits and
        # breaks the build: "CUDA compiler and CUDA toolkit headers are incompatible"). -lcuda matches
        # setup.py's libraries=["cuda"] (driver API); arch is auto-detected from the visible device.
        cpp = [f"-std={std}", "-O3"]
        cuda = [f"-std={std}", "-O3"]
        _LPROJ_EXT = load(name="lutorch_ex_lprojection",
                          sources=[os.path.join(_CSRC, "lprojection.cu"),
                                   os.path.join(_CSRC, "lprojection_py.cpp"),
                                   os.path.join(_CSRC, "common_misc.cpp")],
                          extra_cflags=cpp, extra_cuda_cflags=cuda, extra_ldflags=["-lcuda"],
                          verbose=False)
    except Exception:
        _LPROJ_EXT = None
    return _LPROJ_EXT


def warm_up_native():
    """Eagerly build/load the lprojection extension so the nvcc compile is paid once up front (e.g. at
    model build), not on the first forward. Never raises; a failure just leaves the fallbacks in place."""
    try:
        _lprojection_ext()
    except Exception:
        pass

# The single-anchor input gradient is a short elementwise chain + one scatter_add. In eager
# PyTorch that is ~10 tiny kernel launches and, at small batch, the backward is launch-bound
# there (measured ~0.075 ms vs the pairs path's single fused input-grad kernel at ~0.018 ms).
# A dedicated one-launch CUDA kernel (cartridges/csrc/single_anchor_input_grad.cu, JIT-built via
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
        from torch.utils.cpp_extension import load

        # Vendored, co-located source (no longer reaches into native/lutorch).
        src = os.path.join(_CSRC, "single_anchor_input_grad.cu")
        _SINGLE_IG_EXT = load(name="lutorch_ex_single_anchor_ig", sources=[src], verbose=False)
    except Exception:
        _SINGLE_IG_EXT = None
    return _SINGLE_IG_EXT


def native_manager():
    """Return a LUTorchManager, or None if unavailable. Resolution order:
    (1) the co-located ``lutorch_ex_lprojection`` JIT extension (primary; self-contained);
    (2) the prebuilt ``lutorch_cuda`` library (last-resort fallback, kept one release);
    (3) ``None`` -> callers use the pure/tier-1 torch path.
    Robust to the libc10.so dlopen issue for path (2) (torch-first import; ctypes-preload fallback)."""
    global _MANAGER, _TRIED
    if _TRIED:
        return _MANAGER
    _TRIED = True
    # (1) primary: the co-located JIT extension.
    ext = _lprojection_ext()
    if ext is not None:
        try:
            _MANAGER = ext.get_lutorch_manager()
            return _MANAGER
        except Exception:
            _MANAGER = None
    # (2) last-resort fallback: the prebuilt lutorch_cuda library.
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
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single, drop_mask=None):
        mgr = native_manager()
        W, li, lai, lad, nt, B, G, tph, K, d_out = _flatten(weights, c, c_alt, u_signed_star)
        d_in = z.shape[2]
        tif = _table_indices(B, nt, z.device)
        out, mw, aw = mgr.lprojection_forward_smooth(W, li, lai, lad, tif, tif, True, _THREADS)
        out_pt = out.reshape(B, G, tph, d_out)
        if drop_mask is not None:                 # table dropout: scale each table's blend by its keep
            out_pt = out_pt * drop_mask.unsqueeze(-1).to(out_pt.dtype)  # (same mask scales grad_pt below)
        grp = out_pt.sum(2, dtype=_acc_dtype(out.dtype)).to(weights.dtype)
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        # z is passed by the cartridge already in the weight dtype (the kernels dispatch on it;
        # its values are unused by the input-grad kernels, only its shape). The input gradient is
        # produced in that dtype and returned directly — no bf16->fp32->bf16 round trip; autograd
        # does the single cast to the model input dtype at the x boundary.
        ctx.save_for_backward(W, li, lai, lad, tif, mw, aw,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        ctx.drop_mask = drop_mask
        return grp

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, lad, tif, mw, aw, a_g, b_g, batch_off, z_flat = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.unsqueeze(2).expand(B, G, tph, d_out)
        if ctx.drop_mask is not None:            # scale each table's grad by its keep-mask
            grad_pt = grad_pt * ctx.drop_mask.unsqueeze(-1).to(grad_pt.dtype)
        grad_pt = grad_pt.reshape(B, nt, d_out).contiguous()
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_smooth(
            grad_pt, W, li, lai, tif, tif, mw.contiguous(), aw.contiguous(), _THREADS)
        if ctx.single:
            xg = _single_input_grad(gc_main, gc_alt, lad, a_g, B, G, d_in, nt)
        else:
            xg = mgr.anchor_pairs_lookup_backward_all(
                z_flat, a_g.reshape(-1).contiguous(), b_g.reshape(-1).contiguous(), lad, batch_off,
                gc_main.contiguous(), gc_alt.reshape(-1).contiguous(), True, _THREADS).view(B, G, d_in)
        return wgrad.reshape(G, tph, K, d_out), xg, None, None, None, None, None, None, None


class NativeHard(torch.autograd.Function):
    """Hard value (sum_t W[c_t]) with the native na1 NONSMOOTH backward (weight-grad hard,
    input-grad via the uncertainty carriers)."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single, drop_mask=None):
        mgr = native_manager()
        W, li, lai, lad, nt, B, G, tph, K, d_out = _flatten(weights, c, c_alt, u_signed_star)
        d_in = z.shape[2]
        tif = _table_indices(B, nt, z.device)
        # Hard value: sum_t W[c_t]. lutorch_cuda has no nonsmooth forward, so fuse the per-table
        # gather + tph-sum with embedding_bag (fp32-accumulated for bf16/fp16, cast back to the
        # weight dtype) instead of the eager W[tif,li].sum(2), which materialized the full
        # [B,G,tph,d_out] cell tensor (~2.4 GB at the champion batch). Numerically identical; the
        # saved tensors / backward are unchanged. drop_mask [B,G,tph] (optional) folds into the value
        # via per_sample_weights (sum_t mask_t W[c_t]); the SAME mask scales grad_pt in backward so
        # wgrad and the input-grad carriers inherit the mask_t factor -- table dropout honoured
        # through the native kernels with NO kernel change.
        val = fused_hard_read(weights, c, drop_mask=drop_mask).to(weights.dtype)
        batch_off = (torch.arange(B, device=z.device).repeat_interleave(nt) * (G * d_in)).contiguous()
        ctx.save_for_backward(W, li, lai, lad, tif,
                              a_glob.reshape(B, nt, 1), b_glob.reshape(B, nt, 1), batch_off,
                              z.reshape(B, G * d_in))
        ctx.dims = (G, tph, K, d_out, B, d_in, nt)
        ctx.single = single
        ctx.drop_mask = drop_mask
        return val

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, lad, tif, a_g, b_g, batch_off, z_flat = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.unsqueeze(2).expand(B, G, tph, d_out)
        if ctx.drop_mask is not None:            # scale each table's grad by its keep-mask (matches the
            grad_pt = grad_pt * ctx.drop_mask.unsqueeze(-1).to(grad_pt.dtype)  # masked forward value)
        grad_pt = grad_pt.reshape(B, nt, d_out).contiguous()
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_nonsmooth(
            grad_pt, W, li, lai, tif, tif, _THREADS)
        if ctx.single:
            xg = _single_input_grad(gc_main, gc_alt, lad, a_g, B, G, d_in, nt)
        else:
            xg = mgr.anchor_pairs_lookup_backward_all(
                z_flat, a_g.reshape(-1).contiguous(), b_g.reshape(-1).contiguous(), lad, batch_off,
                gc_main.contiguous(), gc_alt.reshape(-1).contiguous(), True, _THREADS).view(B, G, d_in)
        return wgrad.reshape(G, tph, K, d_out), xg, None, None, None, None, None, None, None
