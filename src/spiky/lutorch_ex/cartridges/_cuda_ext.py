"""Shared plumbing for the co-located CUDA extensions: where the sources live, the JIT build, and the launch knobs of
the two knob-driven kernel families (FusedManifesto* and FusedConfidenceLUT).

* :class:`Extension` is one extension's JIT build / load, once per process, for every fused family (fused_manifesto,
  fused_confidence, and the quantised family's pow2_int8_read, which passes its architecture allowlist and its extra
  nvcc flag). It never raises: when the extension is unavailable it records the classified cause (:mod:`._fallback`)
  and returns None, and the fused cartridge that needs it raises with that cause.
* :func:`not_covered`, :func:`device_problem`, :func:`row_threads` and :func:`row_problem` are the one phrasing of a
  case a fused family's kernels do not cover (the dtype part is :func:`._dtypes.unsupported_dtype`).
* :class:`KernelKnobs` is the launch configuration of the two kernel families. :class:`ManifestoCudaKnobs` and
  :class:`CudaKnobs` (FusedConfidenceLUT) are the same fields, defaulting from ``LUTORCH_EX_MANI_CUDA_*`` resp.
  ``LUTORCH_EX_CONF_CUDA_*``.
* :func:`_row_launch` / :func:`_launch` fit the knobs to a ``d_out``: the vector width is halved until it divides
  ``d_out``, each thread count is raised to cover one table row.
* :data:`_ARCH_DEFAULTS` is the one table of per-architecture launch defaults (today: the forward kernels' CTA order),
  resolved per device by :func:`forward_group_major`.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import torch

from . import _fallback

#: Co-located CUDA sources.
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
# A persistent extension cache, so each nvcc compile is paid once across processes (unless the caller chose one).
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions_lutorch_ex"))


class Extension:
    """One co-located CUDA extension, JIT-built / loaded on the first :meth:`get` of the process.

    ``sources`` are file names under ``csrc/``; ``extra_cuda_cflags`` are added to the common ``-std=<SPIKY_CXX_STD,
    default c++20> -O3``; ``validated_arches`` (compute capabilities), when given, enables the extension only on those
    devices. :meth:`get` returns the module, or None after recording the classified cause -- no CUDA device, a device
    outside ``validated_arches``, or a failed build. Never raises."""

    def __init__(self, name: str, sources: Sequence[str], *, extra_cuda_cflags: Sequence[str] = (),
                 validated_arches: Optional[tuple] = None):
        self.name = name
        self.sources = tuple(sources)
        self.extra_cuda_cflags = tuple(extra_cuda_cflags)
        self.validated_arches = validated_arches
        self.module = None
        self.tried = False

    def get(self):
        if not self.tried:
            self.tried = True
            self.module = self._build()
        return self.module

    def _build(self):
        if not torch.cuda.is_available():
            _fallback.record_build_failure(self.name)                       # -> no_cuda_device
            return None
        if self.validated_arches is not None:
            cap = torch.cuda.get_device_capability()
            if cap not in self.validated_arches:
                _fallback.record_cause(self.name, _fallback.Cause(
                    "unvalidated_arch", f"device compute capability {cap} is not in {self.validated_arches}"))
                return None
        try:
            from torch.utils.cpp_extension import load
            # torch's default -std=c++17 does not compile its own headers here (ATen/core/List_inl.h).
            # cpp_extension supplies the CUDA_HOME include paths and a compatible host compiler, and targets the
            # visible device's architecture.
            std = os.environ.get("SPIKY_CXX_STD", "c++20")
            flags = [f"-std={std}", "-O3"]
            return load(name=self.name, sources=[os.path.join(CSRC, s) for s in self.sources], extra_cflags=flags,
                        extra_cuda_cflags=flags + list(self.extra_cuda_cflags), verbose=False)
        except Exception as e:                                           # no nvcc, compile error, ...: recorded
            _fallback.record_build_failure(self.name, e)
            return None

    def reset(self) -> None:
        """Forget the build (tests): the next :meth:`get` builds / loads again."""
        self.module, self.tried = None, False


def not_covered(cartridge: str, problems: Sequence[str], reference: Optional[str]) -> str:
    """The one message for a case a fused family's kernels do not cover: ``"<cartridge>: not a case the fused kernels
    cover: <problem>; <problem>."``, then ``" Construct <reference> (the reference class) for this case."`` when a
    reference class runs it (None: none does, e.g. a low-precision quantised case)."""
    msg = f"{cartridge}: not a case the fused kernels cover: {'; '.join(problems)}."
    return msg + (f" Construct {reference} (the reference class) for this case." if reference else "")


def device_problem(x: torch.Tensor) -> Optional[str]:
    """The problem text for an input the CUDA kernels cannot take for its device, else None."""
    if x.is_cuda:
        return None
    if not torch.cuda.is_available():
        return "this machine has no CUDA device, which the kernels need"
    return f"the input is on {x.device}, the kernels run on CUDA only"


def row_threads(d_out: int, widths: Sequence[int]) -> tuple:
    """``(vec, need)``: the widest vector width in ``widths`` dividing ``d_out``, and the threads one table row needs at
    it (``d_out / vec``, rounded up to a warp). One CTA covers a row only if ``need <= _MAX_CTA_THREADS``."""
    vec = next(v for v in widths if d_out % v == 0)
    return vec, -(-(d_out // vec) // 32) * 32


def row_problem(d_out: int, vec: int, need: int) -> Optional[str]:
    """The problem text for a d_out one CTA cannot cover, else None."""
    if need <= _MAX_CTA_THREADS:
        return None
    return f"d_out={d_out} needs {need} threads per CTA at vec {vec} (limit {_MAX_CTA_THREADS})"


_KNOB_DEFAULTS = (("fwd_threads", "FWD_THREADS", 64), ("bwd_threads", "BWD_THREADS", 64),
                  ("rows_per_cta", "ROWS_PER_CTA", 4), ("vec", "VEC", 4), ("vec_bf16", "VEC_BF16", 4),
                  ("vec_atomics", "VEC_ATOMICS", 1))


@dataclass(kw_only=True)
class KernelKnobs:
    """Launch configuration of a hand-written kernel family (all explicit). A field left at None takes its default from
    ``LUTORCH_EX_<ENV>_CUDA_<FIELD>`` (``ENV`` set by the subclass), else the built-in default: 64 / 64 threads per CTA
    (forward / backward), 4 rows per CTA, vector width 4 (float32 table) and 4 (bfloat16 table: 8 is slower on the RTX
    5090, mostly in the backward), vector atomics on (sm_90+). The float32-table defaults are the
    sweep winners on the RTX 5090 and the H100 (0d8a66d5); vec_bf16 was swept on the RTX 5090 only (9472c8ef)."""
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


#: Per-architecture launch defaults, keyed by compute capability: ONE table for every fused family.
#:
#: ``cta_order`` is the FORWARD kernels' CTA order. ``"group"`` (group-major) keeps one group's CTAs consecutive, so the
#: CTAs in flight read ~1-2 groups' table slices and those stay in L2; ``"interleaved"`` lets consecutive CTAs walk the
#: groups (``g = CTA % G``), so every group's slice is live at once. Measured (h16 tph64 nap8 d48, 8k / 32k): on the
#: H100 (9.0) group-major takes the forward from 6-37x its compulsory DRAM reads at ~50% L2 hits to <= 1x at 92-99%
#: (16-38% faster); on the RTX 5090 (12.0), whose 96 MB L2 holds the whole table either way, it makes the large-batch
#: forward 1.6-8.4% slower. The BACKWARD kernels are always group-major (faster on both cards); not a knob.
#:
#: An architecture not listed takes :data:`_ARCH_FALLBACK`: interleaved, the order every kernel used before this table,
#: so an unlisted card never silently changes. ``LUTORCH_EX_<FAMILY>_CUDA_CTA_ORDER`` (``group`` / ``interleaved``;
#: FAMILY = MANI, CONF or QUANT) overrides the table.
_ARCH_DEFAULTS = {(9, 0): {"cta_order": "group"}, (12, 0): {"cta_order": "interleaved"}}
_ARCH_FALLBACK = {"cta_order": "interleaved"}
_CTA_ORDERS = ("interleaved", "group")


def resolve_cta_order(capability, env_value: Optional[str] = None) -> str:
    """The forward CTA order for a device of compute ``capability`` (major, minor): ``env_value`` when given (the
    ``LUTORCH_EX_<FAMILY>_CUDA_CTA_ORDER`` override), else the :data:`_ARCH_DEFAULTS` entry, else the fallback."""
    if env_value is not None:
        if env_value not in _CTA_ORDERS:
            raise ValueError(f"CTA order {env_value!r}: expected one of {', '.join(_CTA_ORDERS)}")
        return env_value
    return _ARCH_DEFAULTS.get(tuple(capability), _ARCH_FALLBACK)["cta_order"]


@functools.lru_cache(maxsize=None)
def _forward_group_major(family: str, device_index: int) -> bool:
    env = f"LUTORCH_EX_{family}_CUDA_CTA_ORDER"
    try:
        order = resolve_cta_order(torch.cuda.get_device_capability(device_index), os.environ.get(env))
    except ValueError as e:
        raise ValueError(f"{env}: {e}") from None
    return order == "group"


def forward_group_major(family: str, device: torch.device) -> bool:
    """Whether ``family``'s forward kernels (MANI, CONF or QUANT) run group-major on ``device``: resolved once per
    (family, device) per process from the env override, else the per-architecture table (see :data:`_ARCH_DEFAULTS`)."""
    return _forward_group_major(family, device.index if device.index is not None else torch.cuda.current_device())


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
