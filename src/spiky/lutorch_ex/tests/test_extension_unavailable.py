"""A fused class without its CUDA extension (cartridges/_fallback.py): each build-failure cause it claims to detect is
detected from the evidence for it, and a fused class raises NativeUnavailableError naming the cause -- there is no
fallback (spec rule 3). A case the kernels do not cover raises a RuntimeError naming the case and the reference class.
"""
import importlib.util

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges import _fallback, _fused_manifesto_cuda, fused_confidence
from spiky.lutorch_ex.cartridges._fallback import NativeUnavailableError, classify, record_build_failure
from spiky.lutorch_ex.lut_spec import LUTSpec
from spiky.lutorch_ex.tests._weights import softsign

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _clean():
    _fallback._reset_for_tests()
    yield
    _fallback._reset_for_tests()


@pytest.fixture
def cuda_present(monkeypatch):
    """classify() checks for a CUDA device first; pretend one exists so the other causes are reachable on any host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


# -- cause classification: each from its own evidence -------------------------------------------------------------

def test_setuptools_missing_from_the_exception(cuda_present):
    assert classify(ModuleNotFoundError("No module named 'setuptools'", name="setuptools")).kind == "setuptools_missing"


def test_setuptools_missing_from_the_environment(cuda_present, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "setuptools" else real(n, *a))
    c = classify(RuntimeError("anything"))
    assert c.kind == "setuptools_missing"
    assert "setuptools" in record_build_failure("x", RuntimeError("anything")).remedy


def test_ninja_missing(cuda_present):
    assert classify(RuntimeError("Ninja is required to load C++ extensions")).kind == "ninja_missing"


def test_cuda_version_mismatch_names_both_versions(cuda_present):
    e = RuntimeError("\nThe detected CUDA version (13.0) mismatches the version that was used to compile\n"
                     "PyTorch (12.8). Please make sure to use the same CUDA versions.\n")
    c = classify(e)
    assert c.kind == "cuda_version_mismatch"
    assert "13.0" in c.detail and "12.8" in c.detail


def test_no_cuda_toolkit(cuda_present, monkeypatch):
    import torch.utils.cpp_extension as ce
    monkeypatch.setattr(ce, "CUDA_HOME", None)
    assert classify(RuntimeError("could not find nvcc")).kind == "no_cuda_toolkit"


def test_unsupported_arch(cuda_present):
    c = classify(RuntimeError("nvcc fatal   : Unsupported gpu architecture 'compute_120'"))
    assert c.kind == "unsupported_arch" and "compute_120" in c.detail


def test_compile_failed_keeps_the_error_lines_and_writes_the_log(cuda_present, monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    e = RuntimeError("Error building extension 'lutorch_ex_x':\n[1/2] nvcc ... -c x.cu\n"
                     "x.cu(10): error: expected a \";\"\n1 error detected in the compilation of \"x.cu\".\n")
    c = classify(e, "lutorch_ex_x")
    assert c.kind == "compile_failed"
    assert 'expected a ";"' in c.detail
    assert c.log_path and open(c.log_path).read().startswith("RuntimeError: Error building extension")


def test_torch_too_old(monkeypatch):
    monkeypatch.setattr(_fallback, "_torch_version", lambda: (1, 13))
    assert classify(RuntimeError("x")).kind == "torch_too_old"


def test_no_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert classify(None).kind == "no_cuda_device"


def test_unrecognised_errors_are_not_misattributed(cuda_present, tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    assert classify(OSError("libcuda.so: cannot open shared object file")).kind == "unclassified"


# -- the error: raised every time, naming the cause and the reference class ------------------------------------------

BANNER_NEEDLES = ("THE CUDA EXTENSION THIS FUSED CARTRIDGE RUNS ON IS NOT AVAILABLE", "[setuptools_missing]",
                  "pip install setuptools", "construct the reference class")


def test_raise_unavailable_names_the_cause_every_time(cuda_present):
    record_build_failure("lutorch_ex_fused_manifesto", ModuleNotFoundError("No module named 'setuptools'",
                                                                           name="setuptools"))
    for _ in range(2):                                          # every time, not once
        with pytest.raises(NativeUnavailableError) as e:
            _fallback.raise_unavailable("FusedManifestoHardLUT", "lutorch_ex_fused_manifesto")
        for needle in BANNER_NEEDLES:
            assert needle in str(e.value), needle


def test_raise_unavailable_without_a_recorded_cause_says_so(cuda_present):
    with pytest.raises(NativeUnavailableError, match=r"(?s)\[unclassified\].*did not record a cause"):
        _fallback.raise_unavailable("FusedConfidenceLUT", "lutorch_ex_fused_confidence")


# -- the cartridges -------------------------------------------------------------------------------------------------

def _spec():
    return LUTSpec(h_in=4, h_out=4, tph=8, nap=4, d_in=8, d_out=8, anchor_mode="pairs")


#: Both fused Manifesto twins, with both weight functions.
MANIFESTO = [lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT, softsign(lx.FusedManifestoHardLUT),
             softsign(lx.FusedManifestoSoftLUT)]


@pytest.fixture
def no_manifesto_ext(monkeypatch):
    """The fused_manifesto extension is unavailable, with a recorded cause (setuptools missing)."""
    monkeypatch.setattr(_fused_manifesto_cuda, "_TRIED", True)
    monkeypatch.setattr(_fused_manifesto_cuda, "_EXT", None)
    record_build_failure(_fused_manifesto_cuda.EXT_NAME,
                         ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("cls", MANIFESTO, ids=lambda c: c.__name__)
def test_fused_manifesto_without_extension_raises_with_cause(cls, no_manifesto_ext):
    m = cls(_spec(), seed=0).cuda().train()
    for _ in range(2):
        with pytest.raises(NativeUnavailableError, match=r"\[setuptools_missing\]"):
            m(torch.randn(64, 4, 8, device="cuda"))


@pytest.mark.parametrize("cls", MANIFESTO, ids=lambda c: c.__name__)
def test_fused_manifesto_cpu_input_raises_naming_the_reference_class(cls):
    """No fallback on the CPU either: a CPU input is a case the kernels do not cover."""
    m = cls(_spec(), seed=0).train()
    with pytest.raises(RuntimeError, match=r"ManifestoHardLUT / ManifestoSoftLUT"):
        m(torch.randn(16, 4, 8))


@pytest.fixture
def no_confidence_ext(monkeypatch):
    monkeypatch.setattr(fused_confidence, "fused_confidence_ext", lambda: None)
    record_build_failure(fused_confidence.EXT_NAME, RuntimeError(
        "\nThe detected CUDA version (13.0) mismatches the version that was used to compile\nPyTorch (12.8)."))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_confidence_without_extension_raises_with_cause(no_confidence_ext):
    m = lx.FusedConfidenceLUT(_spec(), seed=0).cuda().train()
    with pytest.raises(NativeUnavailableError, match=r"\[cuda_version_mismatch\]") as e:
        m(torch.randn(32, 4, 8, device="cuda"))
    assert "13.0" in str(e.value)


def test_fused_confidence_cpu_input_raises_naming_the_reference_class():
    m = lx.FusedConfidenceLUT(_spec(), seed=0).train()
    with pytest.raises(RuntimeError, match=r"Construct ConfidenceLUT"):
        m(torch.randn(16, 4, 8))


@pytest.mark.parametrize("cls", [lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT, lx.FusedConfidenceLUT],
                         ids=lambda c: c.__name__)
def test_no_backend_argument(cls):
    """Spec rule 3: the class is the implementation; there is nothing to select."""
    with pytest.raises(TypeError, match="backend"):
        cls(_spec(), seed=0, backend="cuda")
