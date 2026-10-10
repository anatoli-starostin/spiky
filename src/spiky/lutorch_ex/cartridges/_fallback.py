"""Loud, specific reporting when a cartridge's fast CUDA path is unavailable and it falls back to a slow path.

The CUDA extensions (``lutorch_ex_lprojection`` for the Fused Manifesto / SoftSign native backends,
``lutorch_ex_fused_confidence`` for FusedConfidenceLUT) are JIT-built at first use and never raise on a failed build:
the cartridges fall back to correct but far slower pure-torch / tier-1 paths. That is the right behaviour for
correctness, and the wrong one for anybody timing the code, so this module makes the fallback impossible to miss.

* :func:`record_build_failure` is called by the extension loaders when a build/load fails (or is skipped). It
  classifies the CAUSE once, from the exception torch raised plus direct checks, and stores it.
* :func:`report_involuntary_fallback` is called by a cartridge at the point where it would have used the CUDA path
  for a CUDA input but cannot. It emits a multi-line banner through ``warnings.warn`` AND ``logging.error`` (once per
  process per extension and cause), or raises :class:`NativeUnavailableError` when ``SPIKY_LUTORCH_REQUIRE_NATIVE=1``.

Deliberate choices stay quiet: an explicit ``backend=`` other than the CUDA one, ``LUTORCH_EX_NO_CUDA_EXT=1``, a CPU
input, or a geometry / dtype the kernels do not support by design (that is not a failure of the fast path).
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("spiky.lutorch_ex")

STRICT_ENV = "SPIKY_LUTORCH_REQUIRE_NATIVE"
MIN_TORCH = (2, 1)

# Measured 2026-10-10 on the RTX 5090, h16 / tph64 / nap8, r = 128 (d 8), batch 32,768 vectors, fp32 table.
COST_TEXT = (
    "results stay CORRECT but are much SLOWER. Measured 2026-10-10 (RTX 5090, h16/tph64/nap8, r=128,\n"
    "            batch 32,768): FusedManifestoHard native 6.5 ms vs tier1 50.7 ms forward (~8x);\n"
    "            FusedConfidenceLUT CUDA 4.1 ms vs the ConfidenceLUT path 20.7 ms per train step (~5x).\n"
    "            ANY PERFORMANCE NUMBER MEASURED IN THIS STATE IS INVALID."
)


class NativeUnavailableError(RuntimeError):
    """Raised instead of the fallback warning when SPIKY_LUTORCH_REQUIRE_NATIVE=1."""


@dataclass
class Cause:
    """Why an extension is unavailable. ``kind`` is one of the CAUSES keys; ``deliberate`` causes never warn."""
    kind: str
    detail: str = ""
    remedy: str = ""
    log_path: Optional[str] = None
    deliberate: bool = False


# kind -> (headline, remedy). Each entry is only produced when the evidence for it is present (see classify()).
CAUSES = {
    "setuptools_missing": (
        "setuptools is not installed in this Python environment, so torch.utils.cpp_extension cannot JIT-build\n"
        "            the CUDA extension.",
        "pip install setuptools into this venv (uv: uv pip install --python <venv>/bin/python setuptools)."),
    "ninja_missing": (
        "ninja is not installed; torch.utils.cpp_extension needs it to JIT-build CUDA extensions.",
        "pip install ninja into this venv."),
    "no_cuda_toolkit": (
        "no CUDA toolkit was found (CUDA_HOME unset and no nvcc on PATH), so the extension cannot be compiled.",
        "install a CUDA toolkit whose major version matches torch's, and set CUDA_HOME to it."),
    "cuda_version_mismatch": (
        "the CUDA toolkit used to compile the extension does not match the CUDA version torch was built with.",
        "point CUDA_HOME at a toolkit with the same major version as torch.version.cuda."),
    "unsupported_arch": (
        "the compiler / build does not support this GPU's architecture.",
        "use a CUDA toolkit new enough for this GPU (or set TORCH_CUDA_ARCH_LIST to a supported arch)."),
    "compile_failed": (
        "compiling the CUDA extension failed.",
        "read the build log below; fix the toolchain error and retry (delete the stale build directory if needed)."),
    "torch_too_old": (
        "this torch is older than lutorch_ex's minimum.",
        "upgrade torch to >= %d.%d." % MIN_TORCH),
    "no_cuda_device": (
        "torch sees no CUDA device.",
        "run on a CUDA machine (CPU runs use the pure path by design)."),
    "unclassified": (
        "the extension failed to build or load for a reason this check does not recognise.",
        "read the error below and the build log."),
    "disabled": ("disabled by LUTORCH_EX_NO_CUDA_EXT=1 (deliberate).", ""),
}

_RECORDED: dict = {}          # extension name -> Cause
_WARNED: set = set()           # (extension, cause kind) already reported in this process


def strict() -> bool:
    return os.environ.get(STRICT_ENV, "0") == "1"


def _torch_version() -> tuple:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _first_error_lines(msg: str, n: int = 6) -> str:
    lines = [ln for ln in msg.splitlines() if ln.strip()]
    errs = [ln for ln in lines if re.search(r"error|fatal|Error", ln)]
    pick = (errs or lines)[:n]
    return "\n".join("              " + ln.strip()[:150] for ln in pick)


def _write_log(ext_name: str, text: str) -> Optional[str]:
    base = os.environ.get("TORCH_EXTENSIONS_DIR") or tempfile.gettempdir()
    path = os.path.join(base, ext_name, "lutorch_ex_build_error.log")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return path
    except OSError:
        try:
            fd, path = tempfile.mkstemp(prefix=f"{ext_name}_build_error_", suffix=".log")
            with os.fdopen(fd, "w") as f:
                f.write(text)
            return path
        except OSError:
            return None


def classify(exc: Optional[BaseException], ext_name: str = "extension") -> Cause:
    """Name the cause from evidence only: the exception torch raised, and direct environment checks."""
    import importlib.util
    if _torch_version() < MIN_TORCH:
        return Cause("torch_too_old", f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        return Cause("no_cuda_device", "torch.cuda.is_available() is False")
    msg = f"{type(exc).__name__}: {exc}" if exc is not None else ""
    if (isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "setuptools") or \
            importlib.util.find_spec("setuptools") is None:
        return Cause("setuptools_missing", msg.splitlines()[0] if msg else "setuptools not importable")
    if "Ninja is required" in msg:
        return Cause("ninja_missing", msg.splitlines()[0])
    m = re.search(r"detected CUDA version \(([\d.]+)\) mismatches the version that was used to compile\s+"
                  r"PyTorch \(([\d.]+)\)", msg)
    if m:
        return Cause("cuda_version_mismatch", f"toolkit CUDA {m.group(1)} vs torch CUDA {m.group(2)}")
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:          # pragma: no cover - cpp_extension import itself failing is unclassified below
        CUDA_HOME = "?"
    if CUDA_HOME is None or "CUDA_HOME environment variable is not set" in msg:
        return Cause("no_cuda_toolkit", "torch.utils.cpp_extension.CUDA_HOME is None")
    if re.search(r"Unsupported gpu architecture|no kernel image is available|nvcc fatal\s*:.*arch", msg):
        line = next((ln for ln in msg.splitlines() if re.search(r"arch|kernel image", ln)), msg.splitlines()[0])
        return Cause("unsupported_arch", line.strip()[:200])
    if "Error building extension" in msg or "error:" in msg:
        path = _write_log(ext_name, msg)
        return Cause("compile_failed", _first_error_lines(msg), log_path=path)
    return Cause("unclassified", (msg.splitlines()[0] if msg else "no exception captured")[:200],
                 log_path=_write_log(ext_name, msg) if msg else None)


def record_build_failure(ext_name: str, exc: Optional[BaseException] = None, *, disabled: bool = False) -> Cause:
    """Called by an extension loader when it returns None. Classifies and stores the cause (no output here: the
    banner fires where a cartridge actually falls back, so a deliberately unused extension stays silent)."""
    if disabled:
        cause = Cause("disabled", deliberate=True)
    else:
        cause = classify(exc, ext_name)
    cause.remedy = cause.remedy or CAUSES[cause.kind][1]
    _RECORDED[ext_name] = cause
    return cause


def recorded(ext_name: str) -> Optional[Cause]:
    return _RECORDED.get(ext_name)


def banner(cartridge: str, ext_name: str, wanted: str, used: str, cause: Cause) -> str:
    headline = CAUSES[cause.kind][0]
    lines = [
        "=" * 100,
        "LUTORCH_EX: THE FAST CUDA PATH IS NOT AVAILABLE - THIS CARTRIDGE IS RUNNING A SLOW FALLBACK",
        "=" * 100,
        f"Cartridge : {cartridge} (backend '{wanted}' -> '{used}' because the CUDA extension is unavailable)",
        f"Extension : {ext_name}",
        f"Cause     : [{cause.kind}] {headline}",
        f"Remedy    : {cause.remedy or CAUSES[cause.kind][1]}",
    ]
    if cause.detail:
        detail = cause.detail if "\n" in cause.detail else "              " + cause.detail
        lines += ["Detail    :", detail]
    if cause.log_path:
        lines.append(f"Build log : {cause.log_path}")
    lines += [
        f"Cost      : {COST_TEXT}",
        f"Strict    : set {STRICT_ENV}=1 to make this a hard error (the benchmark harnesses do).",
        "Shown once per process per cause.",
        "=" * 100,
    ]
    return "\n".join(lines)


def report_involuntary_fallback(cartridge: str, ext_name: str, wanted: str, used: str) -> None:
    """A cartridge wanted the CUDA path for a CUDA input and cannot have it. Raise (strict) or warn once."""
    cause = _RECORDED.get(ext_name)
    if cause is None:          # the loader never recorded anything: say so rather than guess
        cause = Cause("unclassified", "the extension loader did not record a cause",
                      remedy=CAUSES["unclassified"][1])
    if cause.deliberate:
        return
    text = banner(cartridge, ext_name, wanted, used, cause)
    if strict():
        raise NativeUnavailableError(text)
    key = (ext_name, cause.kind)
    if key in _WARNED:
        return
    _WARNED.add(key)
    log.error("\n%s", text)
    warnings.warn("\n" + text, RuntimeWarning, stacklevel=3)


def backend_of(module) -> str:
    """The backend that actually ran in ``module``'s last forward, for labelling measurements. Raises ValueError when
    it cannot be determined, so a harness never reports a timing of unknown provenance."""
    if not hasattr(module, "_BACKENDS"):                 # a plain cartridge has exactly one implementation
        return "pure (single implementation)"
    used = getattr(module, "last_backend", None)
    if used is not None:
        return used
    chosen = getattr(module, "backend", None)
    if chosen not in (None, "auto"):                     # an explicit choice runs that backend (or raises)
        return chosen
    raise ValueError(f"cannot determine which backend {type(module).__name__} ran (backend='auto' and no "
                     f"last_backend recorded); refusing to report the timing")


class strict_native:
    """Context manager: turn SPIKY_LUTORCH_REQUIRE_NATIVE on for its duration unless the caller set it explicitly,
    so a timing run can never silently measure a fallback. Restores the previous value."""

    def __enter__(self):
        self._prev = os.environ.get(STRICT_ENV)
        if self._prev is None:
            os.environ[STRICT_ENV] = "1"
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop(STRICT_ENV, None)
        return False


def _reset_for_tests() -> None:
    _RECORDED.clear()
    _WARNED.clear()
