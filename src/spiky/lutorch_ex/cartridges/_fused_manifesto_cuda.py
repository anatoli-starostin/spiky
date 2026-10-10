"""The 'cuda' backend of FusedManifestoHardLUT / FusedManifestoSoftLUT: hand-written forward + backward kernels
(``csrc/fused_manifesto.cu``), built the way FusedConfidenceLUT's are.

One forward and one backward kernel per step run the whole cartridge: addressing (margins, MSB-first address,
least-confident bit, neighbour), the uncertainty U = 0.5 / (1 + |u*|), the (weighted) read summed over the group's
tables, and on the backward the weight-gradient scatter (vector fp32 atomics) plus the j*-pair margin gradient into z.
Nothing per table goes through global memory; the backward recomputes the addressing from z. Compared with the
'native' (lprojection) backend this removes the ``[B, G*tph, d_out]`` expanded gradient, the materialised addressing
tensors, the per-element scalar atomics and most of the launches.

Hard and Soft share the kernels and differ only in the read weights (see the .cu header for the math):
  HARD  value  m W[c]                 table grad on c only        (straight-through: the input gradient is that of
                                                                   the (1-U)/U blend)
  SOFT  value  m ((1-U) W[c] + U W[c'])  table grad on c and c'   (the exact derivative of the blend)
with the table-dropout factor m.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from . import _fallback
from ._native_ops import _CSRC

EXT_NAME = "lutorch_ex_fused_manifesto"
MODE_HARD, MODE_SOFT = 0, 1
CUDA_TABLE_DTYPES = (torch.float32, torch.bfloat16)
CUDA_INPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
MAX_NAP = 16


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class ManifestoCudaKnobs:
    """Launch configuration of the Manifesto kernels (explicit; env overrides ``LUTORCH_EX_MANI_CUDA_*``)."""
    fwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_FWD_THREADS", 64))
    bwd_threads: int = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_BWD_THREADS", 64))
    rows_per_cta: int = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_ROWS_PER_CTA", 4))
    vec: int = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_VEC", 4))
    vec_bf16: int = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_VEC_BF16", 4))
    vec_atomics: bool = field(default_factory=lambda: _env_int("LUTORCH_EX_MANI_CUDA_VEC_ATOMICS", 1) == 1)


_EXT = None
_TRIED = False


def fused_manifesto_ext():
    """Build/load the ``lutorch_ex_fused_manifesto`` JIT extension; None if unavailable (never raises; the cause is
    recorded for the fallback banner)."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1":
        _fallback.record_build_failure(EXT_NAME, disabled=True)
        return None
    if not torch.cuda.is_available():
        _fallback.record_build_failure(EXT_NAME)
        return None
    try:
        from torch.utils.cpp_extension import load
        std = os.environ.get("SPIKY_CXX_STD", "c++20")
        _EXT = load(name=EXT_NAME, sources=[os.path.join(_CSRC, "fused_manifesto.cu")],
                    extra_cflags=[f"-std={std}", "-O3"], extra_cuda_cflags=[f"-std={std}", "-O3"], verbose=False)
    except Exception as e:
        _fallback.record_build_failure(EXT_NAME, e)
        _EXT = None
    return _EXT


def _row_launch(knobs, d_out: int, table_dtype: torch.dtype):
    """(vec, threads needed to cover one table row): the vector width is halved until it divides d_out (an odd d_out
    reads one element per load); one row needs d_out / vec threads, rounded up to a warp."""
    vec = knobs.vec if table_dtype == torch.float32 else knobs.vec_bf16
    while vec > 1 and d_out % vec:
        vec //= 2
    return vec, -(-(d_out // vec) // 32) * 32


#: The one remedy every ineligibility message gives (the case is fine on the other backends).
REMEDY = "Use backend='auto' to run on a supported path."
_MAX_CTA_THREADS = 1024


def _dt(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _ineligible(module, x: torch.Tensor):
    """``(what, internals)`` when this input / configuration is NOT a case the kernels cover, else None. These are
    deliberate fallbacks (there is no kernel for that case), not a broken fast path."""
    if not x.is_cuda:
        if not torch.cuda.is_available():
            return "This machine has no CUDA device, which the fused kernel needs.", "torch.cuda.is_available() is False"
        return "The input is on the CPU, but the fused kernel runs on CUDA only.", f"input device: {x.device}"
    if x.dtype not in CUDA_INPUT_DTYPES:
        return (f"The input dtype {_dt(x.dtype)} is not supported by the fused kernel.",
                "supported inputs: " + ", ".join(_dt(d) for d in CUDA_INPUT_DTYPES))
    if module.weights.dtype not in CUDA_TABLE_DTYPES:
        return (f"The table dtype {_dt(module.weights.dtype)} is not supported by the fused kernel.",
                "supported tables: " + ", ".join(_dt(d) for d in CUDA_TABLE_DTYPES))
    if module.spec.nap > MAX_NAP:
        return (f"nap={module.spec.nap} is more than the fused kernel supports (max {MAX_NAP}).",
                f"the cell index packs at most {MAX_NAP} bits")
    d_in_max = torch.iinfo(torch.int16).max
    if module.spec.d_in > d_in_max:
        return (f"d_in={module.spec.d_in} is too wide for the fused kernel (max {d_in_max}).",
                "anchor indices are stored as int16")
    vec, need = _row_launch(module.knobs, module.spec.d_out, module.weights.dtype)
    if need > _MAX_CTA_THREADS:
        return (f"d_out={module.spec.d_out} is too wide for the fused kernel (max {vec * _MAX_CTA_THREADS} at this "
                f"vector width).", f"needs {need} threads per CTA, limit {_MAX_CTA_THREADS}, vec {vec}")
    return None


def ineligible_reason(module, x: torch.Tensor):
    """Why this input / configuration is NOT a case the kernels cover (None when it is), as a user-facing message:
    what the kernel cannot handle, the remedy (the same sentence for every reason), then the internals in a trailing
    parenthetical. The cases: no CUDA device or a CPU input, an fp64 input, an fp16 / fp64 table, nap > 16, d_in beyond
    int16 anchors, or a d_out one CTA cannot cover."""
    r = _ineligible(module, x)
    return None if r is None else f"{r[0]} {REMEDY} ({r[1]})"


def shape_ok(module, x: torch.Tensor) -> bool:
    """Everything the kernels need EXCEPT the extension itself (see ineligible_reason)."""
    return ineligible_reason(module, x) is None


def cuda_ok(module, x: torch.Tensor) -> bool:
    return shape_ok(module, x) and fused_manifesto_ext() is not None


def auto_wants_cuda(module, x: torch.Tensor, next_backend: str) -> bool:
    """'auto' (train and eval): True when the 'cuda' backend can run for this input. Otherwise 'auto' moves on to
    ``next_backend``, and the three reasons are kept apart:
    - the extension is missing / failed to build for a case the kernels cover: reported loudly with its classified
      cause (once per process), or NativeUnavailableError under SPIKY_LUTORCH_REQUIRE_NATIVE=1;
    - not a case the kernels cover (CPU input, fp64, fp16 table, nap > 16, ...): quiet, a debug log line only;
    - no CUDA device on this machine: quiet, a debug log line only."""
    r = _ineligible(module, x)
    if r is None:
        if fused_manifesto_ext() is not None:
            return True
        _fallback.report_involuntary_fallback(type(module).__name__, EXT_NAME, "auto", next_backend)
        return False
    # 'auto' already IS the remedy, so the debug line carries what and why, without the remedy sentence.
    _fallback.log.debug("%s: auto -> %s instead of 'cuda'. %s (%s)", type(module).__name__, next_backend, r[0], r[1])
    return False


def require_cuda(module, x: torch.Tensor) -> None:
    """backend='cuda' was requested explicitly: raise (with the classified cause when only the extension is missing)
    rather than silently run something else."""
    reason = ineligible_reason(module, x)
    if reason is None and fused_manifesto_ext() is not None:
        return
    name = type(module).__name__
    if reason is None:
        cause = _fallback.recorded(EXT_NAME) or _fallback.Cause(
            "unclassified", "the extension loader did not record a cause", remedy=_fallback.CAUSES["unclassified"][1])
        raise _fallback.NativeUnavailableError(
            _fallback.banner(name, EXT_NAME, "cuda", "(none: explicit request)", cause))
    raise RuntimeError(f"{name}(backend='cuda'): {reason}")


def _launch(knobs, d_out: int, table_dtype: torch.dtype):
    """The knobs, made legal for this d_out (see _row_launch): each thread count is raised to cover one table row. At
    the canonical d_out 48 the defaults pass through unchanged. ineligible_reason keeps 'auto' away from a d_out that
    needs more than 1024 threads; an explicit request for one fails here."""
    vec, need = _row_launch(knobs, d_out, table_dtype)
    if need > 1024:
        raise RuntimeError(f"fused_manifesto: d_out {d_out} needs {need} threads per CTA (> 1024) at vec {vec}")
    return max(knobs.fwd_threads, need), max(knobs.bwd_threads, need), vec


class _FusedManifesto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weights, anc_a, anc_b, mask, nap, eps, mode, knobs, out_bf16=False):
        # z: fp32 or bf16 [B, G, d_in]. weights: fp32 or bf16 [G, tph, K, d_out] (read in place). mask: fp32 [B, G, tph]
        # or None. out_bf16: the kernel stores its (fp32-accumulated) output as bf16; only for a bf16 z.
        ext = fused_manifesto_ext()
        W2 = weights.reshape(-1, weights.shape[-1])
        fwd_threads, bwd_threads, vec = _launch(knobs, W2.shape[1], weights.dtype)
        out = ext.manifesto_fwd(z, anc_a, anc_b, W2, mask, nap, eps, mode, fwd_threads, knobs.rows_per_cta, vec,
                                out_bf16)
        ctx.save_for_backward(z, weights, anc_a, anc_b, mask)
        ctx.cfg = (nap, eps, mode, knobs, vec, bwd_threads)
        return out                                                   # fp32 [B, G, d_out]

    @staticmethod
    def backward(ctx, go):
        z, weights, anc_a, anc_b, mask = ctx.saved_tensors
        nap, eps, mode, knobs, vec, bwd_threads = ctx.cfg
        W2 = weights.reshape(-1, weights.shape[-1])
        gW, gz = fused_manifesto_ext().manifesto_bwd(
            go.float().contiguous(), z, anc_a, anc_b, W2, mask, nap, eps, mode,
            bwd_threads, knobs.rows_per_cta, vec, knobs.vec_atomics)
        # gW accumulates in fp32 whatever the table dtype; cast once to the parameter's dtype.
        # gz comes back in z's dtype (a bf16 z: rounded once, in the kernel's store); the .to is then a no-op.
        return gz.to(z.dtype), gW.view_as(weights).to(weights.dtype), None, None, None, None, None, None, None, None


def init_cuda_backend(module, knobs) -> None:
    """Per-instance setup for the 'cuda' backend: launch knobs and int16 copies of the anchors (non-persistent buffers:
    they follow ``.to()`` / ``.cuda()`` and stay out of the state_dict; the int64 buffers remain the source of truth)."""
    module.knobs = knobs if knobs is not None else ManifestoCudaKnobs()
    if module.spec.d_in <= torch.iinfo(torch.int16).max:
        module.register_buffer("_anc_a16", module.anchor_a.to(torch.int16).contiguous(), persistent=False)
        if module.anchor_b is not None:
            module.register_buffer("_anc_b16", module.anchor_b.to(torch.int16).contiguous(), persistent=False)
        else:
            module._anc_b16 = None


def manifesto_cuda_forward(module, x: torch.Tensor, mode: int) -> torch.Tensor:
    """The 'cuda' backend: ``[B, h_in, d_in]`` -> ``[B, h_out, d_out]`` in the input dtype."""
    module._check_input(x)
    # Group g reads input head g % h_in. When h_in == n_groups that map is the identity, so x already IS the per-group
    # input: skip the gather (eager it is a full extra pass + intermediate, which the compiled path used to fuse away).
    zin = x if module.spec.h_in == module.spec.n_groups else x[:, module.in_head, :]
    # A bf16 input goes to the kernels as is: they upconvert it while staging each row (fp32 addressing, no fp32 copy of
    # the input in global memory). fp16 / fp64 are cast to fp32 here as before.
    z = (zin if zin.dtype in (torch.float32, torch.bfloat16) else zin.float()).contiguous()
    # Table dropout: the cartridge's own mask (one source of truth with the other backends: same draw, and an
    # overridden _table_dropout_mask is honoured), fp32, applied per table to the value and every gradient.
    mask = module._table_dropout_mask(x.shape[0], x.device, torch.float32)
    mask = None if mask is None else mask.contiguous()
    # Routing gate: the kernel may store bf16 directly only when _route is the identity (h_out == n_groups: per-head and
    # fan-out). Fan-in sums groups in _route, which must happen on the fp32 output before the single cast below.
    out_bf16 = z.dtype == torch.bfloat16 and module.spec.h_out == module.spec.n_groups
    grp = _FusedManifesto.apply(z, module.weights, module._anc_a16, module._anc_b16, mask,
                                module.spec.nap, module.cmp_eps, mode, module.knobs, out_bf16)
    return module._route(grp, x).to(x.dtype)
