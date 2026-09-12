"""Workspace panel: built and placed correctly (idempotent, other sections untouched), the glossary protocol checks,
and the command line's no-server paths. CPU only, no server."""
import copy
import types

import pytest

from spiky.util.wandb_integration import glossary as G
from spiky.util.wandb_integration import workspace as W

TITLE = 'About these metrics'


def _spec(extra_sections=()):
    return {'section': {'panelBankConfig': {
        'sections': [{'__id__': 'a1', 'name': 'Charts', 'isPanelsAuto': True, 'panels': []},
                     {'__id__': 'b2', 'name': 'train', 'isPanelsAuto': True, 'panels': []}] + list(extra_sections)
        + [{'__id__': 'h3', 'name': 'Hidden Panels', 'isPanelsAuto': False, 'panels': []}],
        'panelConfigOverrides': {'val/loss': {'config': {'metrics': ['val/loss'], 'legendTemplate': 'x'}}},
        'settings': {'searchQuery': ''}}, 'runSets': [{'id': 'r'}]}}


def test_apply_section_is_first_idempotent_and_leaves_the_rest_alone():
    before = _spec()
    once = W.apply_section(before, W.glossary_section('# v1', TITLE))
    twice = W.apply_section(once, W.glossary_section('# v2', TITLE))
    for spec, text in ((once, '# v1'), (twice, '# v2')):
        secs = spec['section']['panelBankConfig']['sections']
        assert secs[0]['__id__'] == W.SECTION_ID and secs[0]['isPanelsAuto'] is False and secs[0]['pinned'] is True
        assert [s['__id__'] for s in secs[1:]] == ['a1', 'b2', 'h3']          # others untouched, in order
        assert sum(s['name'] == TITLE for s in secs) == 1
        assert secs[0]['panels'][0]['viewType'] == 'Markdown Panel' and secs[0]['panels'][0]['config']['value'] == text
        assert spec['section']['panelBankConfig']['panelConfigOverrides'] == before['section']['panelBankConfig']['panelConfigOverrides']
        assert spec['section']['runSets'] == before['section']['runSets']
    assert before == _spec()                                                  # input not mutated


def test_apply_section_replaces_a_renamed_or_moved_copy():
    moved = copy.deepcopy(W.glossary_section('# old', TITLE))
    moved['__id__'] = 'regenerated-by-the-ui'                                 # matched by name too
    spec = W.apply_section(_spec([moved]), W.glossary_section('# new', TITLE))
    names = [s['name'] for s in spec['section']['panelBankConfig']['sections']]
    assert names == [TITLE, 'Charts', 'train', 'Hidden Panels']


def test_section_state():
    md = '# current'
    assert W.section_state(_spec(), md, TITLE) == (False, 'the section is absent')
    ok, why = W.section_state(W.apply_section(_spec(), W.glossary_section(md, TITLE)), md, TITLE)
    assert ok and why == 'present, first, current'
    assert not W.section_state(W.apply_section(_spec(), W.glossary_section('# old', TITLE)), md, TITLE)[0]
    late = _spec([W.glossary_section(md, TITLE)])
    assert W.section_state(late, md, TITLE) == (False, 'the section is at position 2, not first')
    dup = W.apply_section(_spec(), W.glossary_section(md, TITLE))
    dup['section']['panelBankConfig']['sections'].append(W.glossary_section(md, TITLE))
    assert W.section_state(dup, md, TITLE)[1] == '2 copies of the section'
    assert W.section_state({}, md, TITLE)[0] is False


def test_shared_view_names():
    assert W.shared_nw_id('Project — described') == 'projectdescribed'
    assert W.shared_view_name('Project — described') == 'nw-projectdescribed-v'
    assert W.shared_view_name('Scratch 2') == 'nw-scratch2-v'
    with pytest.raises(ValueError):
        W.shared_nw_id('— —')


def test_readme_is_built_from_the_injected_glossary_and_project(glossary):
    md = W.readme('http://h/e/p?nw=x', glossary, 'Proj', 'T', intro='One project for everything.')
    assert md.startswith('# Proj\n\nOne project for everything.\n') and md.count('http://h/e/p?nw=x') == 1
    assert f'[{TITLE}](http://h/e/p?nw=x)' in md and glossary.glossary_hash() in md and glossary.SOURCE in md
    assert 'saved view "T"' in md
    assert W.readme('http://h/e/p/workspace', glossary, 'Proj').startswith('# Proj\n\n- **[')


def test_glossary_protocol_require_and_load(glossary, tmp_path):
    assert G.require(glossary, G.TRACKER_NEEDS + G.PUBLISH_NEEDS + G.AUDIT_NEEDS, 'all') is glossary
    partial = types.SimpleNamespace(PANEL_TITLE='T', glossary_hash=lambda: 'h')
    with pytest.raises(TypeError, match='publish: the glossary lacks panel_markdown'):
        G.require(partial, G.PUBLISH_NEEDS, 'publish')
    with pytest.raises(TypeError, match='needs a glossary'):
        G.require(None, G.PUBLISH_NEEDS, 'publish')
    p = tmp_path / 'my_glossary.py'
    p.write_text("PANEL_TITLE = 'T'\n\ndef panel_markdown():\n    return '# T'\n\ndef glossary_hash():\n    return 'abc'\n")
    mod = G.load(str(p))
    assert mod.PANEL_TITLE == 'T' and G.missing(mod, G.PUBLISH_NEEDS) == [] and G.missing(mod, G.AUDIT_NEEDS) == [
        'undocumented', 'stale']


def test_publish_dry_run_needs_no_server(capsys, monkeypatch, glossary):
    monkeypatch.delenv('WANDB_BASE_URL', raising=False)
    assert W.main(['publish', '--dry-run', '--project', 'P'], glossary=glossary) == 0
    out = capsys.readouterr().out
    assert out.startswith(glossary.panel_markdown()) and W.SECTION_ID in out and repr(TITLE) in out


def test_main_requires_a_glossary_and_a_project(monkeypatch, capsys):
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    with pytest.raises(SystemExit, match='--glossary'):
        W.main(['verify'])
    with pytest.raises(SystemExit, match='--project'):
        W.main(['verify'], glossary=object())
    with pytest.raises(TypeError, match='verify: the glossary lacks'):
        W.main(['verify', '--project', 'P'], glossary=object())
    assert W.main(['bogus']) == 2 and 'publish' in capsys.readouterr().out


def test_csv_header_keys(tmp_path):
    for run, header in (('r1', 'step,val/loss,norm_L0'), ('r2', 'step,val/loss,new_key')):
        d = tmp_path / run
        d.mkdir()
        (d / 'metrics.csv').write_text(header + '\n1,2,3\n')
    keys = W.csv_header_keys([str(tmp_path / 'r2' / 'metrics.csv'), str(tmp_path / 'r1' / 'metrics.csv')])
    assert keys == {'step': 'r1', 'val/loss': 'r1', 'norm_L0': 'r1', 'new_key': 'r2'}
