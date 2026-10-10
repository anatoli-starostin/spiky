"""Skip or fail when a test needs a lutorch_ex CUDA extension.

Two different reasons a CUDA-extension test cannot run, kept apart:
* no CUDA device: a CPU-only machine legitimately cannot run it -> always a skip;
* a CUDA device, but the extension is unavailable (failed to build or load, e.g. setuptools missing): by default a
  skip, but under SPIKY_LUTORCH_REQUIRE_NATIVE=1 a FAILURE naming the extension and its recorded cause, so a broken
  build cannot pass as a green run.
"""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback


def require_cuda() -> None:
    """Skip when there is no CUDA device (never a failure: that is the machine, not the build)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def _unavailable(ext_name: str) -> str:
    cause = _fallback.recorded(ext_name)
    why = f"[{cause.kind}] {cause.detail}".strip() if cause is not None else "no cause recorded"
    return f"the {ext_name} CUDA extension is unavailable: {why}"


def optional_extension(available: bool, ext_name: str) -> bool:
    """For a test that runs without the extension but uses it as an extra reference when present: returns
    ``available``; under SPIKY_LUTORCH_REQUIRE_NATIVE=1 an unavailable extension fails the test instead of being
    silently dropped from the comparison."""
    if not available and _fallback.strict():
        pytest.fail(f"{_fallback.STRICT_ENV}=1 and {_unavailable(ext_name)}", pytrace=False)
    return available


def require_extension(available: bool, ext_name: str) -> None:
    """Call with a CUDA device present: skip (default) or fail (SPIKY_LUTORCH_REQUIRE_NATIVE=1) when ``ext_name`` is
    unavailable. ``available`` is the caller's availability probe (e.g. ``fused_manifesto_ext() is not None``)."""
    require_cuda()
    if available:
        return
    msg = _unavailable(ext_name)
    if _fallback.strict():
        pytest.fail(f"{_fallback.STRICT_ENV}=1 and {msg}", pytrace=False)
    pytest.skip(msg)
