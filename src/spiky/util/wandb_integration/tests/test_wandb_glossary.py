"""DictGlossary, the reference glossary: data in, the whole protocol out ({i} per-layer patterns included), and
glossary.load for the workspace command line. CPU only, no server, no project content."""
import re

import pytest

from spiky.util.wandb_integration import glossary as G
from spiky.util.wandb_integration import workspace as W

METRICS = {
    'train/loss': dict(unit='nats', section='train', desc='Cross-entropy of the step, mean over positions.',
                       short='Step CE.'),
    'train/lr': dict(unit='lr', section='train', desc='Learning rate of the step.'),
    'val_bpb': dict(unit='bits/byte', section='eval', desc='Bits per byte on the validation window.',
                    short='Val bpb | bytes_weighted'),
    'ln2_norm_L{i}': dict(unit='L2', section='per layer', desc="L2 norm of block i's ln2 gain."),
    'step': dict(unit='step', section='csv', desc='metrics.csv row step; not a wandb key.'),
}
SECTIONS = {'train': 'Training', 'eval': 'Evaluation', 'per layer': 'Per layer ({i} = layer index)'}


def _g(**kw):
    return G.DictGlossary(METRICS, sections=SECTIONS, source='tests/example.py', **kw)


def test_satisfies_the_whole_protocol():
    g = _g()
    assert G.missing(g, G.TRACKER_NEEDS + G.PUBLISH_NEEDS + G.AUDIT_NEEDS) == []
    assert g.PANEL_TITLE == G.DEFAULT_TITLE and g.SOURCE == 'tests/example.py'
    assert G.DictGlossary(METRICS, title='Custom').PANEL_TITLE == 'Custom'


def test_lookup_undocumented_and_layer_patterns():
    g = _g()
    assert g.entry_key('ln2_norm_L0') == g.entry_key('ln2_norm_L11') == 'ln2_norm_L{i}'
    for k in ('ln2_norm_Lx', 'ln2_norm_L', 'xln2_norm_L0', 'ln2_norm_L0_extra'):
        assert g.entry_key(k) is None, k
    assert g.lookup('ln2_norm_L3') == dict(unit='L2', desc="L2 norm of block i's ln2 gain.",
                                           short="L2 norm of block i's ln2 gain.", section='per layer')
    assert g.lookup('train/lr')['short'] == 'Learning rate of the step.' and g.lookup('nope') is None
    assert g.undocumented(['val_bpb', 'brand/new', 'ln2_norm_L3', '_step', 'system/gpu.0.gpu', 'brand/new']) == ['brand/new']
    assert G.DictGlossary({'k': dict(unit='u', desc='d')}).lookup('k')['section'] == 'metrics'


def test_stale():
    assert _g().stale(['train/loss', 'train/lr', 'val_bpb', 'ln2_norm_L0', 'ln2_norm_L5']) == ['step']
    assert _g().stale([]) == sorted(METRICS)


def test_hash_is_stable_and_content_sensitive():
    h = _g().glossary_hash()
    assert h == _g().glossary_hash() and len(h) == 12
    assert _g(notes={'eval_steps': 'legacy'}).glossary_hash() != h
    edited = dict(METRICS, val_bpb=dict(METRICS['val_bpb'], desc='Something else.'))
    assert G.DictGlossary(edited, sections=SECTIONS, source='tests/example.py').glossary_hash() != h
    assert G.DictGlossary(METRICS, sections=SECTIONS, source='elsewhere.py').glossary_hash() == h   # content only


def test_table_rows_follow_section_order_then_notes():
    rows = _g(notes={'eval_steps': 'legacy, ignored'}).table_rows()
    assert [r[0] for r in rows] == ['train/loss', 'train/lr', 'val_bpb', 'ln2_norm_L{i}', 'step', 'note: eval_steps']
    assert rows[0] == ['train/loss', 'Cross-entropy of the step, mean over positions.', 'nats']
    assert rows[-1] == ['note: eval_steps', 'legacy, ignored', '-'] and all(len(r) == 3 for r in rows)


def test_panel_markdown_is_content_only_grouped_and_escaped():
    g = _g(notes={'eval_steps': 'legacy_ignored'})
    md = g.panel_markdown()
    assert md.startswith(f'### {G.DEFAULT_TITLE}\n') and md == _g(notes={'eval_steps': 'legacy_ignored'}).panel_markdown()
    assert g.glossary_hash() in md and '`tests/example.py`' in md
    heads = [md.index(h) for h in ('**Training**', '**Evaluation**', '**Per layer ({i} = layer index)**', '**Notes**')]
    assert heads == sorted(heads)
    assert '| `step` |' not in md                              # its section is not in `sections`: documented, not shown
    assert '| `train/lr` | lr | Learning rate of the step. |' in md                        # short defaults to desc
    assert '| `val_bpb` | bits/byte | Val bpb \\| bytes\\_weighted |' in md
    assert '- `eval_steps`: legacy\\_ignored' in md
    for line in md.splitlines():                                    # escaping never breaks the table
        if line.startswith('| `'):
            assert len(re.split(r'(?<!\\)\|', line)) == 5, line


def test_sections_default_to_first_seen_order():
    g = G.DictGlossary({'b': dict(unit='u', desc='B.', section='second'), 'a': dict(unit='u', desc='A.', section='first'),
                        'c': dict(unit='u', desc='C.')})
    md = g.panel_markdown()
    assert md.index('**second**') < md.index('**first**') < md.index('**metrics**')
    assert [r[0] for r in g.table_rows()] == ['b', 'a', 'c']


def test_bad_entries_are_rejected():
    with pytest.raises(ValueError, match="unknown field"):
        G.DictGlossary({'k': dict(unit='u', desc='d', descr='typo')})
    with pytest.raises(ValueError, match="'desc' must be"):
        G.DictGlossary({'k': dict(unit='u')})
    with pytest.raises(ValueError, match="'unit' must be"):
        G.DictGlossary({'k': dict(desc='d', unit='')})


def test_a_dict_glossary_publishes_and_verifies(capsys, monkeypatch):
    g = _g()
    monkeypatch.delenv('WANDB_BASE_URL', raising=False)
    assert W.main(['publish', '--dry-run', '--project', 'P'], glossary=g) == 0
    assert capsys.readouterr().out.startswith(g.panel_markdown())
    spec = W.apply_section({'section': {'panelBankConfig': {'sections': []}}},
                           W.glossary_section(g.panel_markdown(), g.PANEL_TITLE))
    assert W.section_state(spec, g.panel_markdown(), g.PANEL_TITLE) == (True, 'present, first, current')
    stale = _g(notes={'k': 'changed'})
    assert W.section_state(spec, stale.panel_markdown(), stale.PANEL_TITLE)[0] is False


def test_load_returns_a_modules_GLOSSARY_or_a_named_attribute(tmp_path):
    p = tmp_path / 'proj_glossary.py'
    p.write_text("from spiky.util.wandb_integration.glossary import DictGlossary\n"
                 "GLOSSARY = DictGlossary({'k': dict(unit='u', desc='d')}, title='T')\n"
                 "OTHER = DictGlossary({'j': dict(unit='u', desc='d')})\n")
    assert isinstance(G.load(str(p)), G.DictGlossary) and G.load(str(p)).PANEL_TITLE == 'T'
    assert G.load(f'{p}:OTHER').entry_key('j') == 'j'
    plain = tmp_path / 'plain_glossary.py'
    plain.write_text("PANEL_TITLE = 'P'\n")
    assert G.load(str(plain)).PANEL_TITLE == 'P'
