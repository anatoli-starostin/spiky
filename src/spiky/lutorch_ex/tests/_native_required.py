"""Skip or fail when a test needs a lutorch_ex CUDA extension.

Three cases, kept apart:
* no CUDA device: a CPU-only machine legitimately cannot run it -> a skip;
* a CUDA device, the extension deliberately disabled (LUTORCH_EX_NO_CUDA_EXT=1, recorded as a deliberate cause)
  -> a skip: the run said on purpose that it has no extensions;
* a CUDA device, the extension unavailable for any other reason (failed to build or load, e.g. setuptools missing)
  -> a FAILURE naming the extension and its recorded cause, so a broken build cannot pass as a green run.
"""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback


def require_cuda() -> None:
    """Skip when there is no CUDA device (never a failure: that is the machine, not the build)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def _cause(ext_name: str):
    return _fallback.recorded(ext_name)


def _unavailable(ext_name: str) -> str:
    cause = _cause(ext_name)
    why = f"[{cause.kind}] {cause.detail}".strip() if cause is not None else "no cause recorded"
    return f"the {ext_name} CUDA extension is unavailable: {why}"


def _deliberate(ext_name: str) -> bool:
    return _fallback.is_deliberate(ext_name)


def optional_extension(available: bool, ext_name: str) -> bool:
    """For a test that runs without the extension but uses it as an extra reference when present (CUDA device
    present): returns ``available``. An unavailable extension fails the test instead of being silently dropped from
    the comparison, unless it was disabled deliberately (LUTORCH_EX_NO_CUDA_EXT=1)."""
    if not available and not _deliberate(ext_name):
        pytest.fail(_unavailable(ext_name), pytrace=False)
    return available


def require_extension(available: bool, ext_name: str) -> None:
    """Skip when there is no CUDA device. With one, when ``ext_name`` is unavailable: skip if it was disabled
    deliberately (LUTORCH_EX_NO_CUDA_EXT=1), otherwise fail naming the extension and its cause. ``available`` is the
    caller's availability probe (e.g. ``fused_manifesto_ext() is not None``)."""
    require_cuda()
    if available:
        return
    msg = _unavailable(ext_name)
    if _deliberate(ext_name):
        pytest.skip(msg)
    pytest.fail(msg, pytrace=False)
