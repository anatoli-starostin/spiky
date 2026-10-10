"""The loud fallback warning (cartridges/_fallback.py): each cause it claims to detect is detected from the evidence
for it, explicit backend choices stay quiet, strict mode raises, and the warning fires once per process per cause."""
import importlib.util
import logging
import warnings

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges import _fallback, _fused_manifesto_cuda, _native_ops, fused_confidence, fused_manifesto_hard
from spiky.lutorch_ex.cartridges import fused_manifesto_soft
from spiky.lutorch_ex.cartridges._fallback import Cause, NativeUnavailableError, classify, record_build_failure
from spiky.lutorch_ex.lut_spec import LUTSpec

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _fallback._reset_for_tests()
    monkeypatch.delenv(_fallback.STRICT_ENV, raising=False)
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


# -- reporting: loud, once per cause, strict raises, deliberate is quiet ---------------------------------------------

def _report():
    _fallback.report_involuntary_fallback("FusedManifestoHardLUT", "lutorch_ex_lprojection", "auto", "tier1")


def test_banner_fires_once_per_cause_via_warnings_and_logging(cuda_present, caplog):
    record_build_failure("lutorch_ex_lprojection", ModuleNotFoundError("No module named 'setuptools'",
                                                                       name="setuptools"))
    with caplog.at_level(logging.ERROR, logger="spiky.lutorch_ex"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(5):                                   # a training loop: still one warning
                _report()
    msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(msgs) == 1
    text = msgs[0]
    for needle in ("FAST CUDA PATH IS NOT AVAILABLE", "[setuptools_missing]", "pip install setuptools",
                   "ANY PERFORMANCE NUMBER MEASURED IN THIS STATE IS INVALID", "6.5 ms vs tier1 50.7 ms",
                   "SPIKY_LUTORCH_REQUIRE_NATIVE=1"):
        assert needle in text, needle
    assert sum("FAST CUDA PATH" in r.getMessage() for r in caplog.records) == 1


def test_a_different_cause_or_extension_warns_again(cuda_present):
    record_build_failure("lutorch_ex_lprojection", RuntimeError("Ninja is required to load C++ extensions"))
    record_build_failure("lutorch_ex_fused_confidence", RuntimeError("Ninja is required to load C++ extensions"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _report(); _report()
        _fallback.report_involuntary_fallback("FusedConfidenceLUT", "lutorch_ex_fused_confidence", "auto", "pure")
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 2


def test_strict_mode_raises_with_the_same_text(cuda_present, monkeypatch):
    monkeypatch.setenv(_fallback.STRICT_ENV, "1")
    record_build_failure("lutorch_ex_lprojection", RuntimeError("Ninja is required to load C++ extensions"))
    with pytest.raises(NativeUnavailableError, match=r"\[ninja_missing\]"):
        _report()


def test_disabled_by_env_is_deliberate_and_quiet():
    record_build_failure("lutorch_ex_lprojection", disabled=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _report()
    assert not [x for x in w if issubclass(x.category, RuntimeWarning)]


# -- the cartridges: involuntary vs explicit (CUDA) ------------------------------------------------------------------

def _spec():
    return LUTSpec(h_in=4, h_out=4, tph=8, nap=4, d_in=8, d_out=8, anchor_mode="pairs")


@pytest.fixture
def no_native(monkeypatch):
    """Neither Manifesto CUDA extension is available (lprojection for 'native', fused_manifesto for 'cuda'), with a
    recorded cause: the setuptools case, which breaks every JIT build at once."""
    monkeypatch.setattr(_native_ops, "native_available", lambda device: False)
    monkeypatch.setattr(fused_manifesto_hard, "native_available", lambda device: False)
    monkeypatch.setattr(fused_manifesto_soft, "native_available", lambda device: False)
    monkeypatch.setattr(_fused_manifesto_cuda, "_TRIED", True)
    monkeypatch.setattr(_fused_manifesto_cuda, "_EXT", None)
    for ext in ("lutorch_ex_lprojection", _fused_manifesto_cuda.EXT_NAME):
        record_build_failure(ext, ModuleNotFoundError("No module named 'setuptools'", name="setuptools"))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("cls", [lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT])
def test_manifesto_auto_fallback_warns_once_and_records_provenance(cls, no_native):
    m = cls(_spec(), seed=0).cuda().train()
    x = torch.randn(64, 4, 8, device="cuda")          # small batch: Soft's auto would pick native here
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            m(x)
    assert len([x_ for x_ in w if "FAST CUDA PATH" in str(x_.message)]) == 1
    assert _fallback.backend_of(m) == "tier1"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("cls", [lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT])
def test_manifesto_explicit_tier1_is_quiet(cls, no_native):
    m = cls(_spec(), seed=0, backend="tier1").cuda().train()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m(torch.randn(64, 4, 8, device="cuda"))
    assert not [x for x in w if "FAST CUDA PATH" in str(x.message)]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("cls", [lx.FusedManifestoHardLUT, lx.FusedManifestoSoftLUT])
def test_manifesto_explicit_native_unavailable_raises_with_cause(cls, no_native):
    m = cls(_spec(), seed=0, backend="native").cuda().train()
    with pytest.raises(NativeUnavailableError, match=r"\[setuptools_missing\]"):
        m(torch.randn(64, 4, 8, device="cuda"))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_manifesto_strict_mode_raises(no_native, monkeypatch):
    monkeypatch.setenv(_fallback.STRICT_ENV, "1")
    m = lx.FusedManifestoHardLUT(_spec(), seed=0).cuda().train()
    with pytest.raises(NativeUnavailableError):
        m(torch.randn(64, 4, 8, device="cuda"))


def test_cpu_input_is_a_deliberate_path(no_native):
    m = lx.FusedManifestoHardLUT(_spec(), seed=0).train()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m(torch.randn(16, 4, 8))
    assert not [x for x in w if "FAST CUDA PATH" in str(x.message)]


@pytest.fixture
def no_fused_confidence(monkeypatch):
    monkeypatch.setattr(fused_confidence, "fused_confidence_ext", lambda: None)
    record_build_failure("lutorch_ex_fused_confidence", RuntimeError(
        "\nThe detected CUDA version (13.0) mismatches the version that was used to compile\nPyTorch (12.8)."))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_confidence_auto_fallback_warns_with_cause(no_fused_confidence):
    m = lx.FusedConfidenceLUT(_spec(), seed=0).cuda().train() if hasattr(lx, "FusedConfidenceLUT") else \
        fused_confidence.FusedConfidenceLUT(_spec(), seed=0).cuda().train()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            m(torch.randn(32, 4, 8, device="cuda"))
    hits = [str(x.message) for x in w if "FAST CUDA PATH" in str(x.message)]
    assert len(hits) == 1 and "[cuda_version_mismatch]" in hits[0] and "13.0" in hits[0]
    assert _fallback.backend_of(m) == "pure"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_confidence_explicit_pure_is_quiet_and_explicit_cuda_raises(no_fused_confidence):
    FC = fused_confidence.FusedConfidenceLUT
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FC(_spec(), seed=0, backend="pure").cuda().train()(torch.randn(32, 4, 8, device="cuda"))
    assert not [x for x in w if "FAST CUDA PATH" in str(x.message)]
    with pytest.raises(NativeUnavailableError, match=r"\[cuda_version_mismatch\]"):
        FC(_spec(), seed=0, backend="cuda").cuda().train()(torch.randn(32, 4, 8, device="cuda"))


# -- provenance for benchmarks ---------------------------------------------------------------------------------------

def test_backend_of():
    assert _fallback.backend_of(lx.ManifestoHardLUT(_spec(), seed=0)).startswith("pure")
    with pytest.raises(ValueError, match="refusing to report"):
        _fallback.backend_of(lx.FusedManifestoHardLUT(_spec(), seed=0))      # auto, never run
    assert _fallback.backend_of(lx.FusedManifestoHardLUT(_spec(), seed=0, backend="tier1")) == "tier1"
    assert isinstance(Cause("unclassified"), Cause)
