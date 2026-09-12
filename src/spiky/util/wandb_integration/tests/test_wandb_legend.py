"""DictGlossary, the metric legend: data in, undocumented() and legend_markdown() out ({i} per-layer patterns
included). CPU only, no server, no project content."""
import re

import pytest

from spiky.util.wandb_integration import glossary as G

METRICS = {
    'train/loss': dict(unit='nats/tok', section='train', desc='Cross-entropy of the step, mean over positions.'),
    'train/lr': dict(unit='lr', section='train', desc='Learning rate of the step.'),
    'val_bpb': dict(unit='bits/byte', section='eval', desc='Bits per byte | bytes_weighted.'),
    'ln2_norm_L{i}': dict(unit='L2', section='per layer', desc="L2 norm of block i's ln2 gain."),
    'step': dict(unit='step', section='csv', desc='metrics.csv row step; not a wandb key.'),
}
SECTIONS = {'train': 'Training', 'eval': 'Evaluation', 'per layer': 'Per layer ({i} = layer index)'}


def _g(**kw):
    return G.DictGlossary(METRICS, sections=SECTIONS, source='tests/example.py', **kw)


def test_meets_what_the_tracker_needs():
    g = _g()
    assert G.missing(g) == [] and g.TITLE == G.DEFAULT_TITLE and g.SOURCE == 'tests/example.py'
    assert G.missing(object()) == ['undocumented', 'legend_markdown']


def test_lookup_undocumented_and_layer_patterns():
    g = _g()
    assert g.entry_key('ln2_norm_L0') == g.entry_key('ln2_norm_L11') == 'ln2_norm_L{i}'
    for k in ('ln2_norm_Lx', 'ln2_norm_L', 'xln2_norm_L0', 'ln2_norm_L0_extra'):
        assert g.entry_key(k) is None, k
    assert g.lookup('ln2_norm_L3') == dict(unit='L2', desc="L2 norm of block i's ln2 gain.", section='per layer')
    assert g.lookup('nope') is None
    assert g.undocumented(['val_bpb', 'brand/new', 'ln2_norm_L3', '_step', 'system/gpu.0.gpu', 'brand/new']) == ['brand/new']
    assert G.DictGlossary({'k': dict(unit='u', desc='d')}).lookup('k')['section'] == 'metrics'


def test_legend_is_grouped_content_only_and_escaped():
    g = _g(notes={'val_bpb': 'same_key, different quantity elsewhere'})
    md = g.legend_markdown()
    assert md.startswith('### Metrics\n') and md == _g(notes={'val_bpb': 'same_key, different quantity elsewhere'}).legend_markdown()
    assert '`tests/example.py`' in md
    heads = [md.index(h) for h in ('**Training**', '**Evaluation**', '**Per layer ({i} = layer index)**', '**Notes**')]
    assert heads == sorted(heads)
    assert '| `step` |' not in md                              # its section is not in `sections`: described, not shown
    assert '| `train/lr` | Learning rate of the step. [lr] |' in md                       # unit after the description
    assert '| `val_bpb` | Bits per byte \\| bytes\\_weighted. [bits/byte] |' in md
    assert '- `val_bpb`: same\\_key, different quantity elsewhere' in md
    for line in md.splitlines():                                    # two columns; escaping never breaks the table
        if line.startswith('| `'):
            assert len(re.split(r'(?<!\\)\|', line)) == 4, line


def test_sections_default_to_first_seen_order():
    g = G.DictGlossary({'b': dict(unit='u', desc='B.', section='second'), 'a': dict(unit='u', desc='A.', section='first'),
                        'c': dict(unit='u', desc='C.')}, title='Legend')
    md = g.legend_markdown()
    assert md.startswith('### Legend') and md.index('**second**') < md.index('**first**') < md.index('**metrics**')


def test_bad_entries_are_rejected():
    with pytest.raises(ValueError, match="unknown field"):
        G.DictGlossary({'k': dict(unit='u', desc='d', short='no longer read')})
    with pytest.raises(ValueError, match="'desc' must be"):
        G.DictGlossary({'k': dict(unit='u')})
    with pytest.raises(ValueError, match="'unit' must be"):
        G.DictGlossary({'k': dict(desc='d', unit='')})
