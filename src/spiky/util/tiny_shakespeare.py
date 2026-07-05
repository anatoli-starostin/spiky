"""Test/workbook fixture helper: the tiny-shakespeare corpus.

The corpus (~1.1MB) is a fixture, not source, so it is gitignored rather than
committed. Tests and workbooks call :func:`ensure_tinyshakespeare` to fetch it on
first use, so a fresh checkout is self-sufficient without carrying data in git.
"""
import os
import urllib.request

# Canonical Karpathy char-rnn corpus (the file every tinyshakespeare copy derives from).
TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
_EXPECTED_BYTES = 1_115_394


def ensure_tinyshakespeare(dest: str) -> str:
    """Return ``dest``, downloading the tiny-shakespeare corpus there if absent.

    ``dest`` may be relative (the tests pass a path relative to the test dir).
    Returns the same path so call sites can wrap it inline:
    ``TextSnippetSampler(ensure_tinyshakespeare(path), ...)``.
    """
    if not os.path.exists(dest):
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, dest)
    return dest
