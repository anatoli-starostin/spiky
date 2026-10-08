import pytest

from spiky.lutorch_ex.tests._triton_cache import (
    effective_triton_cache_dir, probe_writable, should_enforce_writable_cache, unwritable_cache_error,
)


@pytest.hookimpl(trylast=True)      # after pytest's own pytest_configure has registered the terminal reporter
def pytest_configure(config):
    # Precondition, checked once: on a GPU machine the Triton kernel cache must be writable. Abort the whole session
    # with one clear error instead of a cascade of "Read-only file system" failures from every compiling test. Never
    # worked around: the fix is the caller's (export TRITON_CACHE_DIR). See _triton_cache.py for why it matters.
    path = effective_triton_cache_dir()
    if not should_enforce_writable_cache():
        # CPU-only (or the CUDA probe failed): nothing compiles Triton kernels, so the cache is never written.
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(f"lutorch_ex: no CUDA detected, Triton cache writability check skipped ({path})")
        return
    if not probe_writable(path):
        raise unwritable_cache_error(path)
