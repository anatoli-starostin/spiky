"""The error a fused cartridge raises when its CUDA extension is unavailable, with the classified cause.

The CUDA extensions (``lutorch_ex_fused_manifesto`` for FusedManifestoHardLUT / FusedManifestoSoftLUT,
``lutorch_ex_fused_confidence`` for FusedConfidenceLUT) are JIT-built at first use. A fused class has no other
implementation, so when its extension cannot be built or loaded it raises :class:`NativeUnavailableError` naming why.
There is no fallback and no switch to get one: to run without the extension (a CPU-only box, a CI job), construct
the reference class instead.

* :func:`record_build_failure` is called by the extension loader when a build/load fails. It classifies the CAUSE
  once, from the exception torch raised plus direct checks, and stores it.
* :func:`raise_unavailable` is called by a fused cartridge that needs the extension: it raises with a multi-line
  banner naming the cause and the remedy.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch

MIN_TORCH = (2, 1)

class NativeUnavailableError(RuntimeError):
    """Raised when a fused cartridge's CUDA extension is unavailable."""


@dataclass
class Cause:
    """Why an extension is unavailable. ``kind`` is one of the CAUSES keys."""
    kind: str
    detail: str = ""
    remedy: str = ""
    log_path: Optional[str] = None


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
    "unvalidated_arch": (
        "the extension is enabled only on GPU architectures it has been validated on, and this one is not among them.",
        "run on a validated GPU (the extension's VALIDATED_ARCHES), or construct the reference class."),
    "no_cuda_device": (
        "torch sees no CUDA device.",
        "run on a CUDA machine, or construct the reference class (it needs no extension)."),
    "unclassified": (
        "the extension failed to build or load for a reason this check does not recognise.",
        "read the error below and the build log."),
}

_RECORDED: dict = {}          # extension name -> Cause


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


def record_build_failure(ext_name: str, exc: Optional[BaseException] = None) -> Cause:
    """Called by an extension loader when it returns None. Classifies and stores the cause (no output here: the
    banner fires where a fused cartridge actually needs the extension)."""
    cause = classify(exc, ext_name)
    cause.remedy = cause.remedy or CAUSES[cause.kind][1]
    _RECORDED[ext_name] = cause
    return cause


def record_cause(ext_name: str, cause: Cause) -> Cause:
    """Store a cause the loader determined itself (e.g. an architecture outside its validated list)."""
    cause.remedy = cause.remedy or CAUSES[cause.kind][1]
    _RECORDED[ext_name] = cause
    return cause


def recorded(ext_name: str) -> Optional[Cause]:
    return _RECORDED.get(ext_name)


def banner(cartridge: str, ext_name: str, cause: Cause) -> str:
    headline = CAUSES[cause.kind][0]
    lines = [
        "=" * 100,
        "LUTORCH_EX: THE CUDA EXTENSION THIS FUSED CARTRIDGE RUNS ON IS NOT AVAILABLE",
        "=" * 100,
        f"Cartridge : {cartridge} (a fused class: it runs only on its CUDA kernels)",
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
        "Without   : construct the reference class (ManifestoHardLUT / ManifestoSoftLUT / ConfidenceLUT /",
        "            QuantisedConfidenceLUT), which needs no extension.",
        "=" * 100,
    ]
    return "\n".join(lines)


def raise_unavailable(cartridge: str, ext_name: str) -> None:
    """A fused cartridge needs ``ext_name`` and it is unavailable: raise NativeUnavailableError with the cause."""
    cause = _RECORDED.get(ext_name)
    if cause is None:          # the loader never recorded anything: say so rather than guess
        cause = Cause("unclassified", "the extension loader did not record a cause", remedy=CAUSES["unclassified"][1])
    raise NativeUnavailableError(banner(cartridge, ext_name, cause))


def _reset_for_tests() -> None:
    _RECORDED.clear()
