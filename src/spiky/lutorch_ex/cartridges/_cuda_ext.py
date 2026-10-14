"""Shared plumbing for the co-located CUDA extensions: where the sources live, the JIT build, and the launch knobs of
the two hand-written kernel families (FusedManifesto* and FusedConfidenceLUT).

* :func:`load_extension` is the one JIT build / load of the fused cartridges' extensions (fused_manifesto,
  fused_confidence; FusedQuantisedConfidenceLUT's p2_scalars extension has its own architecture-gated loader in
  ``_pow2_int8``). It never raises:
  when the extension is unavailable it records the classified cause (:mod:`._fallback`) and returns None, and the
  fused cartridge that needs it raises with that cause.
* :class:`KernelKnobs` is the launch configuration of the two kernel families. :class:`ManifestoCudaKnobs` and
  :class:`CudaKnobs` (FusedConfidenceLUT) are the same fields, defaulting from ``LUTORCH_EX_MANI_CUDA_*`` resp.
  ``LUTORCH_EX_CONF_CUDA_*``.
* :func:`_row_launch` / :func:`_launch` fit the knobs to a ``d_out``: the vector width is halved until it divides
  ``d_out``, each thread count is raised to cover one table row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import torch

from . import _fallback

#: Co-located CUDA sources.
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
# A persistent extension cache, so each nvcc compile is paid once across processes (unless the caller chose one).
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions_lutorch_ex"))


def load_extension(name: str, sources: Sequence[str]):
    """JIT-build / load extension ``name`` from ``sources`` (file names under ``csrc/``); the module, or None after
    recording the classified cause (no CUDA device, or a failed build). Never raises."""
    if not torch.cuda.is_available():
        _fallback.record_build_failure(name)
        return None
    try:
        from torch.utils.cpp_extension import load
        # torch's default -std=c++17 does not compile its own headers here (ATen/core/List_inl.h). cpp_extension
        # supplies the CUDA_HOME include paths and a compatible host compiler, and targets the visible device's arch.
        std = os.environ.get("SPIKY_CXX_STD", "c++20")
        flags = [f"-std={std}", "-O3"]
        return load(name=name, sources=[os.path.join(CSRC, s) for s in sources], extra_cflags=flags,
                    extra_cuda_cflags=flags, verbose=False)
    except Exception as e:
        _fallback.record_build_failure(name, e)
        return None


_KNOB_DEFAULTS = (("fwd_threads", "FWD_THREADS", 64), ("bwd_threads", "BWD_THREADS", 64),
                  ("rows_per_cta", "ROWS_PER_CTA", 4), ("vec", "VEC", 4), ("vec_bf16", "VEC_BF16", 4),
                  ("vec_atomics", "VEC_ATOMICS", 1))


@dataclass(kw_only=True)
class KernelKnobs:
    """Launch configuration of a hand-written kernel family (all explicit). A field left at None takes its default from
    ``LUTORCH_EX_<ENV>_CUDA_<FIELD>`` (``ENV`` set by the subclass), else the built-in default: 64 / 64 threads per CTA
    (forward / backward), 4 rows per CTA, vector width 4 (float32 table) and 4 (bfloat16 table: 8 is slower on the RTX
    5090, mostly in the backward), vector atomics on (sm_90+). Defaults = the sweep winners on the RTX 5090 and the
    H100."""
    ENV: ClassVar[str] = ""
    fwd_threads: Optional[int] = None
    bwd_threads: Optional[int] = None
    rows_per_cta: Optional[int] = None
    vec: Optional[int] = None
    vec_bf16: Optional[int] = None
    vec_atomics: Optional[bool] = None

    def __post_init__(self):
        for f, env, default in _KNOB_DEFAULTS:
            if getattr(self, f) is None:
                v = int(os.environ.get(f"LUTORCH_EX_{self.ENV}_CUDA_{env}", default))
                setattr(self, f, v == 1 if f == "vec_atomics" else v)


@dataclass(kw_only=True)
class ManifestoCudaKnobs(KernelKnobs):
    """The FusedManifesto* 'cuda' kernels' knobs (env defaults ``LUTORCH_EX_MANI_CUDA_*``)."""
    ENV: ClassVar[str] = "MANI"


@dataclass(kw_only=True)
class CudaKnobs(KernelKnobs):
    """FusedConfidenceLUT's 'cuda' kernels' knobs (env defaults ``LUTORCH_EX_CONF_CUDA_*``)."""
    ENV: ClassVar[str] = "CONF"


#: Threads per CTA the kernels may launch.
_MAX_CTA_THREADS = 1024


def _row_launch(knobs, d_out: int, table_dtype: torch.dtype):
    """(vec, threads needed to cover one table row): the vector width is halved until it divides d_out (an odd d_out
    reads one element per load); one row needs d_out / vec threads, rounded up to a warp."""
    vec = knobs.vec if table_dtype == torch.float32 else knobs.vec_bf16
    while vec > 1 and d_out % vec:
        vec //= 2
    return vec, -(-(d_out // vec) // 32) * 32


def _launch(knobs, d_out: int, table_dtype: torch.dtype):
    """The knobs, made legal for this d_out (see _row_launch): each thread count is raised to cover one table row. At
    the default knobs nothing changes for a d_out divisible by 4 and at most 256 (e.g. the canonical 48). The fused
    cartridges reject a d_out that needs more than 1024 threads before launching (with the reason)."""
    vec, need = _row_launch(knobs, d_out, table_dtype)
    if need > _MAX_CTA_THREADS:
        raise RuntimeError(f"d_out {d_out} needs {need} threads per CTA (> {_MAX_CTA_THREADS}) at vec {vec}")
    return max(knobs.fwd_threads, need), max(knobs.bwd_threads, need), vec
