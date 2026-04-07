import os

# Must be set before lutorch modules are imported.
os.environ.setdefault("SPIKY_GT_NO_COMPILE", "1")

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_MARK = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")

_LUTORCH_COMPILE_MODULES = [
    "spiky.lutorch.anchor_pairs_lookup",
    "spiky.lutorch.wta_lookup",
    "spiky.lutorch.l_projection",
]


@pytest.fixture(autouse=True, params=[False, True], ids=["recompute-off", "recompute-on"])
def _force_recompute_in_backward(request, monkeypatch):
    if request.param:
        import spiky.lutorch.multi_head_lut as mod
        monkeypatch.setattr(mod, "_FORCE_RECOMPUTE_IN_BACKWARD", True)


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
