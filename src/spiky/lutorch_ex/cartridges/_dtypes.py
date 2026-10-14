"""The dtype vocabulary shared by the cartridges: the one phrasing of a dtype a fused CUDA kernel refuses."""
from __future__ import annotations

from typing import Sequence

import torch


def unsupported_dtype(role: str, dtype: torch.dtype, supported: Sequence[torch.dtype], note: str = "") -> str:
    """``"<role> dtype torch.X is not supported by the fused kernel (supported: torch.A, torch.B[; note])"``: the one
    phrasing every fused class uses for a dtype its CUDA kernels do not take (``role``: ``"input"`` or ``"table"``).
    Dtypes are printed as torch prints them (``torch.float32``)."""
    listed = ", ".join(str(d) for d in supported)
    return f"{role} dtype {dtype} is not supported by the fused kernel (supported: {listed}{'; ' + note if note else ''})"
