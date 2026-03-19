import os

# Must be set before lutorch modules are imported.
os.environ.setdefault("SPIKY_LUTORCH_NO_COMPILE", "1")
os.environ.setdefault("SPIKY_GT_NO_COMPILE", "1")

import pytest
import torch

@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ]
)
def device(request):
    return request.param
