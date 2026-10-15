"""Skip (or fail) a test that needs a lutorch_ex CUDA extension, i.e. one that runs a fused class.

The fused classes have no fallback, so a test that runs one needs its extension. Two cases:
* no CUDA device: the extension cannot exist on this machine -> a skip (a CPU-only run is legitimate);
* a CUDA device, the extension unavailable (failed to build or load, e.g. setuptools missing) -> a FAILURE naming the
  extension and its recorded cause, so a broken build cannot pass as a green run.

Use :func:`requires_extension` (``requires_extension("manifesto")`` / ``("confidence")`` / ``("quantised")``) at the
top of a test, or the :data:`needs_manifesto_ext` / :data:`needs_confidence_ext` / :data:`needs_quantised_ext`
fixtures.
"""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback


def require_cuda() -> None:
    """Skip when there is no CUDA device (never a failure: that is the machine, not the build)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")


def _probe(which: str):
    """``(available, ext_name)`` for ``which`` in {'manifesto', 'confidence', 'quantised'}; loads (JIT-builds) the
    extension."""
    if which == "manifesto":
        from spiky.lutorch_ex.cartridges._fused_manifesto_cuda import EXT
        return EXT.get() is not None, EXT.name
    if which == "confidence":
        from spiky.lutorch_ex.cartridges._fused_confidence_cuda import EXT
        return EXT.get() is not None, EXT.name
    if which == "quantised":
        from spiky.lutorch_ex.cartridges import _fused_quantised_cuda as fq
        return fq.ensure_registered(), fq.EXT.name
    raise ValueError(f"unknown extension {which!r} (expected 'manifesto', 'confidence' or 'quantised')")


def _unavailable(ext_name: str) -> str:
    cause = _fallback.recorded(ext_name)
    why = f"[{cause.kind}] {cause.detail}".strip() if cause is not None else "no cause recorded"
    return f"the {ext_name} CUDA extension is unavailable: {why}"


def require_extension(available: bool, ext_name: str) -> None:
    """Skip when there is no CUDA device; with one, fail naming ``ext_name`` and its cause unless ``available``."""
    require_cuda()
    if not available:
        pytest.fail(_unavailable(ext_name), pytrace=False)


def requires_extension(which: str) -> None:
    """:func:`require_extension` for the 'manifesto' (FusedManifestoHard/Soft) or 'confidence' (FusedConfidenceLUT)
    extension. The CUDA check comes first, so a CPU-only run skips without trying to build anything."""
    require_cuda()
    require_extension(*_probe(which))


@pytest.fixture
def needs_manifesto_ext():
    requires_extension("manifesto")


@pytest.fixture
def needs_confidence_ext():
    requires_extension("confidence")


@pytest.fixture
def needs_quantised_ext():
    requires_extension("quantised")
