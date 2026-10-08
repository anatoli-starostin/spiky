from spiky.lutorch_ex.tests._triton_cache import effective_triton_cache_dir, probe_writable, unwritable_cache_error


def pytest_configure(config):
    # Precondition, checked once: the Triton kernel cache must be writable. Abort the whole session with one clear
    # error instead of a cascade of "Read-only file system" failures from every compiling test. Never worked
    # around: the fix is the caller's (export TRITON_CACHE_DIR). See _triton_cache.py for why it matters.
    path = effective_triton_cache_dir()
    if not probe_writable(path):
        raise unwritable_cache_error(path)
