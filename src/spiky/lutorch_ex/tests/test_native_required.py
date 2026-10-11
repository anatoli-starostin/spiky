"""The CUDA-extension test gates (tests/_native_required.py), every branch exercised directly: no CUDA device ->
skip; a CUDA device with the extension unavailable -> fail by default, naming the extension and its recorded cause,
or skip under SPIKY_LUTORCH_DOESNT_REQUIRE_NATIVE=1. Runs anywhere (the CUDA probe is patched)."""
import pytest
import torch

from spiky.lutorch_ex.cartridges import _fallback
from spiky.lutorch_ex.tests import _native_required as nr

EXT = "lutorch_ex_test_ext"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(nr.OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(_fallback.STRICT_ENV, raising=False)
    _fallback._reset_for_tests()
    yield
    _fallback._reset_for_tests()


def _cuda(monkeypatch, present):
    """Patch the CUDA probe, then record a setuptools build failure for EXT (the cause classifier probes CUDA too)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: present)
    _fallback.record_build_failure(EXT, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))


@pytest.mark.parametrize("opt_out", [False, True])
def test_no_cuda_device_is_always_a_skip(monkeypatch, opt_out):
    _cuda(monkeypatch, False)
    if opt_out:
        monkeypatch.setenv(nr.OPT_OUT_ENV, "1")
    with pytest.raises(pytest.skip.Exception, match="needs a CUDA device"):
        nr.require_extension(False, EXT)


def test_cuda_without_the_extension_fails_by_default_naming_extension_and_cause(monkeypatch):
    _cuda(monkeypatch, True)
    with pytest.raises(pytest.fail.Exception) as e:
        nr.require_extension(False, EXT)
    msg = str(e.value)
    assert EXT in msg and "[setuptools_missing]" in msg and nr.OPT_OUT_ENV in msg
    with pytest.raises(pytest.fail.Exception, match=r"\[setuptools_missing\]"):
        nr.optional_extension(False, EXT)


def test_cuda_without_the_extension_skips_under_the_opt_out(monkeypatch):
    _cuda(monkeypatch, True)
    monkeypatch.setenv(nr.OPT_OUT_ENV, "1")
    with pytest.raises(pytest.skip.Exception, match=r"lutorch_ex_test_ext CUDA extension is unavailable: \[setuptools"):
        nr.require_extension(False, EXT)
    assert nr.optional_extension(False, EXT) is False


def test_the_runtime_strict_mode_does_not_change_the_gates(monkeypatch):
    """SPIKY_LUTORCH_REQUIRE_NATIVE is the library's runtime strict mode, not a test gate."""
    _cuda(monkeypatch, True)
    monkeypatch.setenv(_fallback.STRICT_ENV, "1")
    monkeypatch.setenv(nr.OPT_OUT_ENV, "1")
    with pytest.raises(pytest.skip.Exception):
        nr.require_extension(False, EXT)


def test_an_available_extension_passes(monkeypatch):
    _cuda(monkeypatch, True)
    nr.require_extension(True, EXT)
    assert nr.optional_extension(True, EXT) is True
