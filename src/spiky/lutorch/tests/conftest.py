import os

# Must be set before lutorch modules are imported.
os.environ.setdefault("SPIKY_GT_NO_COMPILE", "1")

# The sandbox cage mounts ~/.triton (triton's default JIT cache) read-only, so compiling a
# triton kernel there fails with OSError(Errno 30, read-only file system). Point the cache at a
# writable dir before any triton / torch.compile import, matching the ffn_replacement run wrapper
# (TRITON_CACHE_DIR=~/.cache/triton). setdefault respects an explicit override from the shell.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))
try:
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
except OSError:
    pass

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_MARK = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")

_LUTORCH_COMPILE_MODULES = [
    "spiky.lutorch.anchor_pairs_lookup",
    "spiky.lutorch.wta_lookup",
    "spiky.lutorch.l_projection",
]


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=_CUDA_MARK),
        pytest.param("cuda-no-compile", marks=_CUDA_MARK),
    ]
)
def device(request, monkeypatch):
    d = request.param
    if d == "cuda-no-compile":
        import importlib
        for mod_name in _LUTORCH_COMPILE_MODULES:
            mod = importlib.import_module(mod_name)
            monkeypatch.setattr(mod, "_USE_LUTORCH_COMPILE", False)
        return "cuda"
    return d
