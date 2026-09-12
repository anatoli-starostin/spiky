"""The metric-glossary protocol: what tracker.py and workspace.py need from an injected glossary.

A glossary says what every logged key measures. This package never imports one. Callers pass an object -- a plain
module works, as long as it has these names -- to Tracker.start(glossary=...) (optional) and to the workspace
publish / verify / audit commands (required):

    PANEL_TITLE: str                      title of the workspace section and its Markdown Panel, e.g. "About these metrics"
    SOURCE: str                           OPTIONAL: where the glossary is defined; shown in messages and artifact metadata
    undocumented(keys) -> list[str]       the sorted keys that have no entry. Keys wandb owns (a leading underscore,
                                          system/*) should count as documented.
    glossary_hash() -> str                a short content hash that changes exactly when the content changes
    table_rows() -> list[[key, description, unit]]
                                          the rows of the per-run `metric_glossary` artifact table
    panel_markdown() -> str               the workspace panel text. It must depend on the glossary content only (no commit
                                          sha, no timestamp): verify compares it with the published text exactly.
    stale(seen_keys) -> list[str]         the entries that match none of seen_keys (audit only)

Who uses what: the tracker's drift check uses undocumented(); its per-run artifact uses glossary_hash() and table_rows();
publish / verify use PANEL_TITLE, panel_markdown() and glossary_hash(); audit uses undocumented(), stale() and
glossary_hash().
"""
import importlib
import importlib.util
import os
from typing import Iterable, List, Protocol

TRACKER_NEEDS = ('undocumented', 'glossary_hash', 'table_rows')
PUBLISH_NEEDS = ('PANEL_TITLE', 'panel_markdown', 'glossary_hash')
AUDIT_NEEDS = ('undocumented', 'stale', 'glossary_hash')


class Glossary(Protocol):                                                    # documentation; nothing checks against it
    PANEL_TITLE: str

    def undocumented(self, keys: Iterable[str]) -> List[str]: ...

    def glossary_hash(self) -> str: ...

    def table_rows(self) -> List[List[str]]: ...

    def panel_markdown(self) -> str: ...

    def stale(self, seen_keys: Iterable[str]) -> List[str]: ...


def missing(glossary, needs):
    """The names in `needs` that `glossary` lacks."""
    return [n for n in needs if not hasattr(glossary, n)]


def require(glossary, needs, what):
    """`glossary` if it has every name in `needs`, else TypeError naming what is missing."""
    if glossary is None:
        raise TypeError(f'{what} needs a glossary (see spiky.util.wandb_integration.glossary)')
    lacks = missing(glossary, needs)
    if lacks:
        raise TypeError(f'{what}: the glossary lacks {", ".join(lacks)} (see spiky.util.wandb_integration.glossary)')
    return glossary


def load(spec):
    """A glossary from a .py file path or an importable module name (for the workspace command line)."""
    if spec.endswith('.py') or os.sep in spec:
        path = os.path.abspath(spec)
        mod_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        if mod_spec is None:
            raise ImportError(f'cannot load a glossary from {spec!r}')
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(spec)
