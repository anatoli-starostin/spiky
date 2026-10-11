"""Skip or fail when a test needs a lutorch_ex CUDA extension.

Two different reasons a CUDA-extension test cannot run, kept apart:
* no CUDA device: a CPU-only machine legitimately cannot run it -> always a skip;
* a CUDA device, but the extension is unavailable (failed to build or load, e.g. setuptools missing): by default a
  FAILURE naming the extension and its recorded cause, so a broken build cannot pass as a green run. It is a skip
  only when SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE=1 is set (e.g. a deliberate run without the extensions).

The library's runtime strict mode, SPIKY_LUTORCH_REQUIRE_NATIVE=1 (cartridges/_fallback.py: a cartridge raises
instead of warning when it falls back), is a separate knob and does not change these gates.
"""
import os

import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback

OPT_OUT_ENV = "SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE"


def native_optional() -> bool:
    """True when SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE=1: an unavailable extension skips instead of failing."""
    return os.environ.get(OPT_OUT_ENV, "0") == "1"


def require_cuda() -> None:
    """Skip when there is no CUDA device (never a failure: that is the machine, not the build)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def _unavailable(ext_name: str) -> str:
    cause = _fallback.recorded(ext_name)
    why = f"[{cause.kind}] {cause.detail}".strip() if cause is not None else "no cause recorded"
    return f"the {ext_name} CUDA extension is unavailable: {why}"


def optional_extension(available: bool, ext_name: str) -> bool:
    """For a test that runs without the extension but uses it as an extra reference when present (CUDA device
    present): returns ``available``. An unavailable extension fails the test instead of being silently dropped from
    the comparison, unless SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE=1."""
    if not available and not native_optional():
        pytest.fail(f"{_unavailable(ext_name)} (set {OPT_OUT_ENV}=1 to run without it)", pytrace=False)
    return available


def require_extension(available: bool, ext_name: str) -> None:
    """Skip when there is no CUDA device. With one: fail (default) or skip (SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE=1)
    when ``ext_name`` is unavailable. ``available`` is the caller's availability probe (e.g.
    ``fused_manifesto_ext() is not None``)."""
    require_cuda()
    if available:
        return
    msg = _unavailable(ext_name)
    if native_optional():
        pytest.skip(msg)
    pytest.fail(f"{msg} (set {OPT_OUT_ENV}=1 to skip instead)", pytrace=False)
