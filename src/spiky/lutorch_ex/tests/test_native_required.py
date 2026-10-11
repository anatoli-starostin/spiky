"""The CUDA-extension test gates (tests/_native_required.py), every branch exercised directly: no CUDA device ->
skip; a CUDA device with the extension deliberately disabled (LUTORCH_EX_NO_CUDA_EXT=1) -> skip; with it unavailable
for any other reason -> fail, naming the extension and its recorded cause. Runs anywhere (the CUDA probe is
patched)."""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback
from spiky.lutorch_ex.tests import _native_required as nr

EXT = "lutorch_ex_test_ext"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _fallback._reset_for_tests()
    monkeypatch.delenv("LUTORCH_EX_NO_CUDA_EXT", raising=False)   # the deliberate case is set per test
    yield
    _fallback._reset_for_tests()


def _cuda(monkeypatch, present, deliberate=False):
    """Patch the CUDA probe, then record EXT as unavailable: disabled on purpose, or a setuptools build failure (the
    cause classifier probes CUDA too, hence the order)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: present)
    if deliberate:
        _fallback.record_build_failure(EXT, disabled=True)
    else:
        _fallback.record_build_failure(EXT, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))


@pytest.mark.parametrize("deliberate", [False, True])
def test_no_cuda_device_is_always_a_skip(monkeypatch, deliberate):
    _cuda(monkeypatch, False, deliberate)
    with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
        nr.require_extension(False, EXT)


def test_cuda_without_the_extension_fails_naming_extension_and_cause(monkeypatch):
    _cuda(monkeypatch, True)
    with pytest.raises(pytest.fail.Exception) as e:
        nr.require_extension(False, EXT)
    assert EXT in str(e.value) and "[setuptools_missing]" in str(e.value)
    with pytest.raises(pytest.fail.Exception, match=r"\[setuptools_missing\]"):
        nr.optional_extension(False, EXT)


def test_cuda_with_the_extension_deliberately_disabled_skips(monkeypatch):
    _cuda(monkeypatch, True, deliberate=True)
    with pytest.raises(pytest.skip.Exception, match=r"lutorch_ex_test_ext CUDA extension is unavailable: \[disabled\]"):
        nr.require_extension(False, EXT)
    assert nr.optional_extension(False, EXT) is False


def test_the_env_escape_hatch_skips_even_without_a_recorded_cause(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LUTORCH_EX_NO_CUDA_EXT", "1")
    with pytest.raises(pytest.skip.Exception, match="no cause recorded"):
        nr.require_extension(False, EXT)


def test_no_recorded_cause_is_a_failure(monkeypatch):
    """An unavailable extension whose loader recorded nothing is not deliberate: it fails rather than skips."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(pytest.fail.Exception, match="no cause recorded"):
        nr.require_extension(False, EXT)


def test_an_available_extension_passes(monkeypatch):
    _cuda(monkeypatch, True)
    nr.require_extension(True, EXT)
    assert nr.optional_extension(True, EXT) is True
