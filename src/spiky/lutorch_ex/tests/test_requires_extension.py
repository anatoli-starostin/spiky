"""The CUDA-extension test gate (tests/_requires_extension.py), every branch exercised directly: no CUDA device ->
skip; a CUDA device without the extension -> fail, naming the extension and its recorded cause; available -> pass.
Runs anywhere (the CUDA probe is patched)."""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback
from spiky.lutorch_ex.tests import _requires_extension as rx

EXT = "lutorch_ex_test_ext"


@pytest.fixture(autouse=True)
def _clean():
    _fallback._reset_for_tests()
    yield
    _fallback._reset_for_tests()


def _cuda(monkeypatch, present):
    """Patch the CUDA probe, then record EXT as unavailable (a setuptools build failure; the cause classifier probes
    CUDA too, hence the order)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: present)
    _fallback.record_build_failure(EXT, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))


def test_no_cuda_device_is_a_skip(monkeypatch):
    _cuda(monkeypatch, False)
    with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
        rx.require_extension(False, EXT)
    for which in ("manifesto", "confidence", "quantised"):     # skips before any build is attempted
        with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
            rx.requires_extension(which)


def test_cuda_without_the_extension_fails_naming_extension_and_cause(monkeypatch):
    _cuda(monkeypatch, True)
    with pytest.raises(pytest.fail.Exception) as e:
        rx.require_extension(False, EXT)
    assert EXT in str(e.value) and "[setuptools_missing]" in str(e.value)


def test_no_recorded_cause_is_a_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(pytest.fail.Exception, match="no cause recorded"):
        rx.require_extension(False, EXT)


def test_an_available_extension_passes(monkeypatch):
    _cuda(monkeypatch, True)
    rx.require_extension(True, EXT)


def test_unknown_extension_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="unknown extension"):
        rx.requires_extension("lprojection")
