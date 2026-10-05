"""Sign-bit address packing for lutorch_ex cartridges.

The convention is **MSB-first**: anchor pair ``0`` is the most-significant bit. For ``nap`` pairs the bit weights are
``powers[i] = 2 ** (nap - 1 - i)``, and a table's cell index is

    c = sum_i bit_i * powers[i],   bit_i = [margin_i > eps]

so a single bit flip at pair ``i`` toggles ``c`` by ``powers[i]`` (used for the
two-alternative backward's neighbour cell). Defining it here keeps the convention in one
place for every cartridge.
"""
from __future__ import annotations

from typing import Optional

import torch


def msb_first_powers(
    nap: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """MSB-first bit weights ``[nap]``: ``powers[i] = 2 ** (nap - 1 - i)`` (pair 0 = high bit)."""
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=dtype))
