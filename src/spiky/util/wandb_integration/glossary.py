"""Metric glossaries for spiky.util.wandb_integration: DictGlossary (the default path) and the protocol it implements.

A glossary says what every logged key measures. This package never imports one. Callers pass it to
Tracker.start(glossary=...) (optional: the drift check and the per-run glossary artifact) and to the workspace
publish / verify / audit commands (required).

DEFAULT PATH -- DictGlossary: the project supplies the data, the class supplies the whole protocol.

    from spiky.util.wandb_integration.glossary import DictGlossary

    GLOSSARY = DictGlossary({
        'train/loss':    dict(unit='nats/token', section='train', desc='Cross-entropy of the step, ...', short='Step CE.'),
        'ln2_norm_L{i}': dict(unit='L2 norm', section='per layer', desc="L2 norm of block i's ln2 gain."),
    }, sections={'train': 'Training', 'per layer': 'Per layer ({i} = layer index)'},
       source='experiments/<family>/<this file>.py')

  Per entry: `unit` and `desc` (the full definition, written from the code) are required; `short` (the one-line panel
  form, default desc) and `section` (default "metrics") are optional; any other field is an error, because nothing
  reads it. A key containing {i} documents that key with any non-negative integer in place of {i}. See DictGlossary.

ESCAPE HATCH -- any object with these names (a plain module works), when data alone is not enough (a panel laid out
differently, two-tier config notes, ...):

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

A glossary never decides whether a run's notes link to a published panel: that is Tracker.start(glossary_panel=...),
set explicitly. PANEL_TITLE only names the panel publish writes.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import re
from typing import Iterable, List, Protocol

TRACKER_NEEDS = ('undocumented', 'glossary_hash', 'table_rows')
PUBLISH_NEEDS = ('PANEL_TITLE', 'panel_markdown', 'glossary_hash')
AUDIT_NEEDS = ('undocumented', 'stale', 'glossary_hash')
DEFAULT_TITLE = 'About these metrics'


class Glossary(Protocol):                                                    # documentation; nothing checks against it
    PANEL_TITLE: str

    def undocumented(self, keys: Iterable[str]) -> List[str]: ...

    def glossary_hash(self) -> str: ...

    def table_rows(self) -> List[List[str]]: ...

    def panel_markdown(self) -> str: ...

    def stale(self, seen_keys: Iterable[str]) -> List[str]: ...


_MD_CELL = re.compile(r'([\\`*_\[\]|])')                # not < >: some wandb markdown renderers show "\<" literally


def _cell(text):
    """Literal text inside a markdown table cell (an underscore in a name such as clip_grad_norm_ would italicise)."""
    return _MD_CELL.sub(r'\\\1', text).replace('\n', ' ')


class DictGlossary:
    """The reference glossary: {key: fields} in, the whole protocol out.

    metrics    {key: dict(unit=..., desc=..., short=..., section=...)}; unit and desc required. A key containing {i}
               matches that key with any non-negative integer in place of {i} (per-layer keys: 'ln2_norm_L{i}').
    title      PANEL_TITLE, the published section's name (default "About these metrics").
    sections   {section: panel heading}, in panel order. Given: only these sections are shown in the panel (entries in
               other sections are still documented and still in the artifact, e.g. columns that exist only in a local
               metrics file). Omitted: every section is shown, in first-seen order, headed by its own name.
    notes      optional {name: text}: a bullet list at the bottom of the panel (under notes_title) and `note: <name>`
               rows in the artifact -- e.g. config keys whose meaning is not what the name suggests.
    source     SOURCE: where this glossary is defined (shown in drift messages, the panel and the artifact metadata).

    Keys wandb owns (a leading underscore, system/*) count as documented.
    """

    FIELDS = ('unit', 'desc', 'short', 'section')

    def __init__(self, metrics, *, title=DEFAULT_TITLE, sections=None, notes=None, source=None, notes_title='Notes'):
        entries = {}
        for key, fields in metrics.items():
            unknown = sorted(set(fields) - set(self.FIELDS))
            if unknown:
                raise ValueError(f'glossary entry {key!r}: unknown field(s) {unknown}; allowed: {list(self.FIELDS)}')
            for req in ('unit', 'desc'):
                if not isinstance(fields.get(req), str) or not fields[req].strip():
                    raise ValueError(f'glossary entry {key!r}: {req!r} must be a non-empty string')
            entries[key] = dict(unit=fields['unit'], desc=fields['desc'], short=fields.get('short') or fields['desc'],
                                section=fields.get('section') or 'metrics')
        self.METRICS = entries
        seen = list(dict.fromkeys(e['section'] for e in entries.values()))
        self.SECTIONS = dict(sections) if sections is not None else {s: s for s in seen}
        self._order = list(self.SECTIONS) + [s for s in seen if s not in self.SECTIONS]
        self.NOTES = dict(notes or {})
        self.PANEL_TITLE, self.SOURCE, self.NOTES_TITLE = title, source, notes_title
        self._patterns = [(re.compile('^' + re.escape(k).replace(re.escape('{i}'), r'\d+') + '$'), k)
                          for k in entries if '{i}' in k]

    def entry_key(self, key):
        """The entry (exact key or {i} pattern) that documents `key`, or None."""
        if key in self.METRICS:
            return key
        for rx, k in self._patterns:
            if rx.match(key):
                return k
        return None

    def lookup(self, key):
        """The entry's fields for `key` (unit, desc, short, section), or None."""
        k = self.entry_key(key)
        return None if k is None else dict(self.METRICS[k])

    def is_documented(self, key):
        return key.startswith('_') or key.startswith('system/') or self.entry_key(key) is not None

    def undocumented(self, keys):
        return sorted({k for k in keys if not self.is_documented(k)})

    def stale(self, seen_keys):
        hit = {self.entry_key(k) for k in seen_keys}
        return sorted(k for k in self.METRICS if k not in hit)

    def glossary_hash(self):
        blob = json.dumps({'metrics': self.METRICS, 'sections': self.SECTIONS, 'notes': self.NOTES}, sort_keys=True,
                          ensure_ascii=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def _sorted_entries(self):
        return sorted(self.METRICS.items(), key=lambda kv: (self._order.index(kv[1]['section']), kv[0]))

    def table_rows(self):
        rows = [[k, e['desc'], e['unit']] for k, e in self._sorted_entries()]
        return rows + [[f'note: {k}', v, '-'] for k, v in self.NOTES.items()]

    def panel_markdown(self):
        intro = f'One line per logged key (glossary `{self.glossary_hash()}`'
        intro += (f', from `{self.SOURCE}`' if self.SOURCE else '') + "). Full definitions: the run's `metric_glossary` "
        intro += 'artifact (Artifacts tab).'
        out = [f'### {self.PANEL_TITLE}', '', intro, '']
        for section, heading in self.SECTIONS.items():
            items = [(k, e) for k, e in self._sorted_entries() if e['section'] == section]
            if not items:
                continue
            out += [f'**{heading}**', '', '| key | unit | what it measures |', '|---|---|---|']
            out += [f'| `{k}` | {_cell(e["unit"])} | {_cell(e["short"])} |' for k, e in items]
            out.append('')
        if self.NOTES:
            out += [f'**{self.NOTES_TITLE}**', '']
            out += [f'- `{k}`: {_cell(v)}' for k, v in self.NOTES.items()]
            out.append('')
        return '\n'.join(out)


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
    """A glossary from a .py file path or an importable module name, optionally `:ATTRIBUTE` (for the workspace command
    line). Without an attribute: the module's GLOSSARY if it has one (e.g. a DictGlossary), else the module itself."""
    head, sep, attr = spec.rpartition(':')
    if sep and attr.isidentifier():
        spec = head
    else:
        attr = ''
    if spec.endswith('.py') or os.sep in spec:
        path = os.path.abspath(spec)
        mod_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        if mod_spec is None:
            raise ImportError(f'cannot load a glossary from {spec!r}')
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(spec)
    if attr:
        return getattr(mod, attr)
    return getattr(mod, 'GLOSSARY', mod)
