"""Test helper: a Manifesto cartridge class with ``uncertainty_func="softsign"`` bound, for tests that parametrize over classes."""
import functools


def softsign(cls):
    """``cls`` with ``uncertainty_func="softsign"``; ``__name__`` names both, for test ids and failure messages."""
    f = functools.partial(cls, uncertainty_func="softsign")
    f.__name__ = f"{cls.__name__}[softsign]"
    f.cls = cls
    return f
