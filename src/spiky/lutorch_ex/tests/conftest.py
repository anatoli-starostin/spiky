import getpass
import os
import tempfile

# torch.compile'd CUDA paths make Triton write kernels to its cache, by default $TRITON_HOME/.triton/cache
# (TRITON_HOME defaults to ~). Where that is read-only (a sandbox, a read-only home), a cold cache fails
# every such test with "OSError: [Errno 30] Read-only file system" - and a warm cache hides it, since
# cached kernels need no write. Redirect to a per-user temp dir, but only then: an explicitly set
# TRITON_CACHE_DIR is never overridden, and a writable default is left alone. Must run before Triton
# reads the knob, i.e. at conftest import, before the test modules import spiky.lutorch_ex.


def _writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            return True
    except OSError:
        return False


_TRITON_CACHE_REDIRECT = None
if "TRITON_CACHE_DIR" not in os.environ:
    _default = os.path.join(os.environ.get("TRITON_HOME") or os.path.expanduser("~"), ".triton", "cache")
    if not _writable(_default):
        _TRITON_CACHE_REDIRECT = os.path.join(tempfile.gettempdir(), f"triton_cache_{getpass.getuser()}")
        os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE_REDIRECT


def pytest_report_header(config):
    if _TRITON_CACHE_REDIRECT is not None:
        return f"lutorch_ex: default Triton cache not writable, TRITON_CACHE_DIR -> {_TRITON_CACHE_REDIRECT}"
