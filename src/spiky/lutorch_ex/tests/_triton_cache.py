"""Triton kernel-cache precondition for the lutorch_ex test session (used by conftest.py).

torch.compile'd CUDA paths make Triton write kernels to its cache: ``TRITON_CACHE_DIR`` if set, else
$TRITON_HOME/.triton/cache (TRITON_HOME defaults to ~). If that dir is read-only, every compiling test fails with
"OSError: [Errno 30] Read-only file system" - and only on some runs: when inductor compiles from scratch it points
Triton at its own writable cache dir, but on an FX-graph-cache hit it skips that and Triton writes to the dir
above. On a GPU machine the suite checks this up front and refuses to run, rather than working around it. On a
CPU-only machine nothing is compiled with Triton and the cache is never written, so the check is skipped.
"""
import os
import tempfile

import pytest


def cuda_available() -> bool:
    """``torch.cuda.is_available()``: a driver probe (``cudaGetDeviceCount``) that leaves torch's CUDA state
    uninitialised; the test modules call it at import anyway."""
    import torch
    return torch.cuda.is_available()


def should_enforce_writable_cache(cuda_probe=None) -> bool:
    """Enforce the writable-cache precondition only where Triton kernels get compiled, i.e. when CUDA is available.
    Fails open: if the probe itself breaks (torch import, driver), skip the check rather than block a CPU run.
    ``cuda_probe`` defaults to :func:`cuda_available`, looked up at call time."""
    probe = cuda_probe if cuda_probe is not None else cuda_available
    try:
        return bool(probe())
    except Exception:
        return False


def probe_writable(path: str) -> bool:
    """True iff a file can actually be created, written and deleted in ``path`` (created if missing).
    A real write, not ``os.access``, which can say yes on read-only mounts / overlayfs."""
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-probe-") as f:
            f.write(b"probe")
            f.flush()
        return True
    except OSError:
        return False


def effective_triton_cache_dir() -> str:
    """The dir Triton will write kernels to: ``TRITON_CACHE_DIR`` if set, else Triton's default."""
    explicit = os.environ.get("TRITON_CACHE_DIR")
    if explicit:
        return explicit
    return os.path.join(os.environ.get("TRITON_HOME") or os.path.expanduser("~"), ".triton", "cache")


class TritonCacheNotWritableError(pytest.UsageError, RuntimeError):
    """A RuntimeError that pytest also treats as a usage error: raised from pytest_configure it aborts the session
    with a single ``ERROR: <message>`` line instead of an INTERNALERROR traceback."""


def unwritable_cache_error(path: str) -> TritonCacheNotWritableError:
    how = "from TRITON_CACHE_DIR" if os.environ.get("TRITON_CACHE_DIR") else "Triton's default; TRITON_CACHE_DIR is unset"
    return TritonCacheNotWritableError(
        f"lutorch_ex tests: the Triton kernel cache dir {path!r} ({how}) is not writable. The CUDA tests "
        "torch.compile kernels and Triton must write the compiled kernels there - otherwise they fail with "
        "'OSError: [Errno 30] Read-only file system'. Fix: export TRITON_CACHE_DIR to a writable dir, e.g. "
        "export TRITON_CACHE_DIR=/tmp/triton-cache"
    )
