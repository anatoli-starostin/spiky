"""Deterministic-mode flag for lutorch dispatch points.

When `True`, kernels that would otherwise use float `atomicAdd` (or any other
nondeterministic reduction) are replaced with deterministic equivalents
(typically PyTorch ops under `torch.use_deterministic_algorithms(True)`).
When `False` (default), the faster kernels are used.

Usage:

    from spiky.lutorch.determinism import set_deterministic, is_deterministic

    set_deterministic(True)
    ...
    # inside a dispatch site:
    if is_deterministic():
        # deterministic fallback
    else:
        # fast kernel path

Every place that currently has a known nondeterminism source should branch on
this flag so flipping one switch gives reproducible training.
"""

_DETERMINISTIC: bool = False


def set_deterministic(value: bool) -> None:
    """Set global deterministic mode for lutorch dispatch points."""
    global _DETERMINISTIC
    _DETERMINISTIC = bool(value)


def is_deterministic() -> bool:
    """Return the current deterministic-mode flag (default: False)."""
    return _DETERMINISTIC
