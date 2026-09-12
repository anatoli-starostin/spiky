"""Metric legends for spiky.util.wandb_integration: the markdown that describes every metric a run logs.

Binding rule (claude/wandb.md section 5): every run's W&B notes describe the experiment AND every metric it logs. The
tracker writes that one markdown blob at run start; the metric part is a glossary's legend_markdown(). Keep the
glossary as data in code:

    from spiky.util.wandb_integration.glossary import DictGlossary

    GLOSSARY = DictGlossary({
        'train/loss':    dict(unit='nats/tok', section='train', desc='Cross-entropy of the step, ...'),
        'ln2_norm_L{i}': dict(unit='L2', section='per layer', desc="L2 norm of block i's ln2 gain."),
    }, sections={'train': 'Training', 'per layer': 'Per layer ({i} = layer index)'},
       notes={'val_bpb': 'Same key, different quantity in other families: ...'},
       source='experiments/<family>/<this file>.py')

Per entry: `unit` and `desc` (the definition, written from the code that computes the value) are required; `section`
(default "metrics") is optional; any other field is an error, because nothing reads it. A key containing {i} describes
that key with any non-negative integer in place of {i}. Keep units short (nats/tok, bits/byte, L2, s/step): they are
printed in brackets after the description.

What the tracker needs from a glossary -- a DictGlossary, or any object with these two names:

    undocumented(keys) -> list[str]   the sorted keys it does not describe; keys wandb owns (a leading underscore,
                                      system/*) count as described
    legend_markdown() -> str          the legend written into every run's notes; SOURCE (optional) names the file
"""
import re

NEEDS = ('undocumented', 'legend_markdown')
DEFAULT_TITLE = 'Metrics'

_MD_CELL = re.compile(r'([\\`*_\[\]|])')                # not < >: some wandb markdown renderers show "\<" literally


def _cell(text):
    """Literal text inside a markdown table cell (an underscore in a name such as clip_grad_norm_ would italicise)."""
    return _MD_CELL.sub(r'\\\1', text).replace('\n', ' ')


def is_wandb_key(key):
    """Keys wandb itself logs (leading underscore, system/*): never ours to describe."""
    return key.startswith('_') or key.startswith('system/')


def missing(glossary, needs=NEEDS):
    """The names in `needs` that `glossary` lacks."""
    return [n for n in needs if not hasattr(glossary, n)]


class DictGlossary:
    """The reference glossary: {key: fields} in, the legend out.

    metrics    {key: dict(unit=..., desc=..., section=...)}; unit and desc required. A key containing {i} matches that key
               with any non-negative integer in place of {i} (per-layer keys: 'ln2_norm_L{i}').
    title      the legend's heading (default "Metrics").
    sections   {section: heading}, in legend order. Given: only these sections are rendered (entries in other sections
               still count as described, e.g. columns that exist only in a local metrics file). Omitted: every section,
               in first-seen order, headed by its own name.
    notes      optional {name: text}: a bullet list at the end of the legend (under notes_title), e.g. a key that means
               something different from a same-named key elsewhere.
    source     SOURCE: where this glossary is defined (shown in the legend and in undescribed-key warnings).
    """

    FIELDS = ('unit', 'desc', 'section')

    def __init__(self, metrics, *, title=DEFAULT_TITLE, sections=None, notes=None, source=None, notes_title='Notes'):
        entries = {}
        for key, fields in metrics.items():
            unknown = sorted(set(fields) - set(self.FIELDS))
            if unknown:
                raise ValueError(f'glossary entry {key!r}: unknown field(s) {unknown}; allowed: {list(self.FIELDS)}')
            for req in ('unit', 'desc'):
                if not isinstance(fields.get(req), str) or not fields[req].strip():
                    raise ValueError(f'glossary entry {key!r}: {req!r} must be a non-empty string')
            entries[key] = dict(unit=fields['unit'], desc=fields['desc'], section=fields.get('section') or 'metrics')
        self.METRICS = entries
        seen = list(dict.fromkeys(e['section'] for e in entries.values()))
        self.SECTIONS = dict(sections) if sections is not None else {s: s for s in seen}
        self._order = list(self.SECTIONS) + [s for s in seen if s not in self.SECTIONS]
        self.NOTES = dict(notes or {})
        self.TITLE, self.SOURCE, self.NOTES_TITLE = title, source, notes_title
        self._patterns = [(re.compile('^' + re.escape(k).replace(re.escape('{i}'), r'\d+') + '$'), k)
                          for k in entries if '{i}' in k]

    def entry_key(self, key):
        """The entry (exact key or {i} pattern) that describes `key`, or None."""
        if key in self.METRICS:
            return key
        for rx, k in self._patterns:
            if rx.match(key):
                return k
        return None

    def lookup(self, key):
        """The entry's fields for `key` (unit, desc, section), or None."""
        k = self.entry_key(key)
        return None if k is None else dict(self.METRICS[k])

    def undocumented(self, keys):
        return sorted({k for k in keys if not (is_wandb_key(k) or self.entry_key(k) is not None)})

    def legend_markdown(self):
        """The legend: one table per section (key | what it measures [unit]), then the notes. Content only."""
        out = [f'### {self.TITLE}', '']
        if self.SOURCE:
            out += [f'Defined in `{self.SOURCE}`.', '']
        ordered = sorted(self.METRICS.items(), key=lambda kv: (self._order.index(kv[1]['section']), kv[0]))
        for section, heading in self.SECTIONS.items():
            items = [(k, e) for k, e in ordered if e['section'] == section]
            if not items:
                continue
            out += [f'**{heading}**', '', '| key | what it measures [unit] |', '|---|---|']
            out += [f'| `{k}` | {_cell(e["desc"])} [{_cell(e["unit"])}] |' for k, e in items]
            out.append('')
        if self.NOTES:
            out += [f'**{self.NOTES_TITLE}**', '']
            out += [f'- `{k}`: {_cell(v)}' for k, v in self.NOTES.items()]
            out.append('')
        return '\n'.join(out)
