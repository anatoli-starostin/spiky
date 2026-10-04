"""Leaf registry mapping a deployment *format tag* to a rebuild callable.

This module is deliberately a **leaf**: it imports nothing from :mod:`.deploy` or :mod:`.cartridges`,
so a cartridge module can register its reconstruction hook here without creating an import cycle
(``deploy`` imports cartridges; cartridges must not import ``deploy``). Each rebuilder has the
signature ``(spec, tensors, meta, device) -> nn.Module`` and complements the export-side
``SupportsDeploymentExport.to_deployment()``: :func:`spiky.lutorch_ex.deploy.load_deployment` looks up
``meta["format"]`` here and calls the registered callable, so ``deploy`` holds no cartridge-specific
knowledge of which inference class to build.
"""
from __future__ import annotations

from typing import Callable

_REBUILDERS: dict[str, Callable] = {}


def register_rebuilder(fmt: str, fn: Callable) -> None:
    """Register ``fn`` as the rebuilder for deployment format tag ``fmt`` (idempotent overwrite)."""
    _REBUILDERS[fmt] = fn


def get_rebuilder(fmt: str) -> Callable:
    """Return the rebuilder registered for ``fmt``; raise a clear ``KeyError`` listing known tags."""
    try:
        return _REBUILDERS[fmt]
    except KeyError:
        raise KeyError(
            f"no deployment rebuilder registered for format {fmt!r}; known tags: {sorted(_REBUILDERS)}"
        ) from None
