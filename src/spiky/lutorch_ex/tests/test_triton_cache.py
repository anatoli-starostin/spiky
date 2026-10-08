"""The conftest's Triton-cache precondition: a real-write probe, the effective dir, and a clear error.

Tested directly on temp dirs; nothing here touches the session's actual cache dir (env changes are monkeypatched
and restored)."""
import os

import pytest

from spiky.lutorch_ex.tests._triton_cache import (
    effective_triton_cache_dir, probe_writable, unwritable_cache_error,
)

needs_non_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                    reason="root bypasses permission bits, so a chmod 0500 dir stays writable")


@pytest.fixture
def read_only_dir(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    yield d
    d.chmod(0o700)                                  # so tmp_path cleanup can remove it


def test_probe_writable_creates_missing_dir_and_leaves_nothing(tmp_path):
    target = tmp_path / "a" / "b"
    assert probe_writable(str(target))
    assert target.is_dir() and list(target.iterdir()) == []


@needs_non_root
def test_probe_unwritable_dir(read_only_dir):
    assert not probe_writable(str(read_only_dir))
    assert not probe_writable(str(read_only_dir / "missing"))   # cannot be created either


def test_effective_dir_is_explicit_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "mine"))
    assert effective_triton_cache_dir() == str(tmp_path / "mine")


def test_effective_dir_defaults_to_triton_home_and_sets_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv("TRITON_HOME", str(tmp_path))
    assert effective_triton_cache_dir() == str(tmp_path / ".triton" / "cache")
    assert "TRITON_CACHE_DIR" not in os.environ                 # resolving never redirects


def test_error_names_path_reason_and_fix(monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", "/ro/cache")
    err = unwritable_cache_error("/ro/cache")
    assert isinstance(err, RuntimeError) and isinstance(err, pytest.UsageError)
    msg = str(err)
    assert "'/ro/cache'" in msg and "from TRITON_CACHE_DIR" in msg and "Triton" in msg
    assert "export TRITON_CACHE_DIR=/tmp/triton-cache" in msg
