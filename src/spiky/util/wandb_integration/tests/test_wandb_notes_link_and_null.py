"""The notes' glossary-panel link is explicit (glossary_panel), the shipped NullTracker, and the documented import guard
for this package failing to import. CPU only, no server (a fake wandb module)."""
import sys
import types

import pytest

from spiky.util.wandb_integration import backfill as B
from spiky.util.wandb_integration import glossary as G
from spiky.util.wandb_integration import tracker as T

URL = 'http://wandb.invalid:8080'


def _start(monkeypatch, tmp_path, fake_wandb, **kw):
    monkeypatch.setenv('WANDB_BASE_URL', URL)
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_ENTITY', 'ent')
    t = T.Tracker.start({'description': 'D.'}, str(tmp_path), project='Proj', **kw)
    return t, fake_wandb.runs[-1]


def test_a_publishable_glossary_adds_no_panel_link_unless_asked(monkeypatch, tmp_path, fake_wandb, glossary):
    assert G.missing(glossary, G.PUBLISH_NEEDS) == []                     # publishable, PANEL_TITLE and all
    t, run = _start(monkeypatch, tmp_path, fake_wandb, glossary=glossary)
    assert 'at the top of the' not in run.notes and 'artifact [metric\\_glossary:' in run.notes
    t.finish()


def test_panel_link_to_a_shared_saved_view(monkeypatch, tmp_path, fake_wandb, glossary):
    t, run = _start(monkeypatch, tmp_path, fake_wandb, glossary=glossary, glossary_panel='Proj — described')
    assert (f'"About these metrics" at the top of the [saved view "Proj — described"]({URL}/ent/Proj?nw=projdescribed)'
            in run.notes)
    t.finish()


def test_panel_link_to_the_personal_workspace(monkeypatch, tmp_path, fake_wandb, glossary):
    t, run = _start(monkeypatch, tmp_path, fake_wandb, glossary=glossary, glossary_panel=T.PERSONAL_WORKSPACE)
    assert f'"About these metrics" at the top of the [project workspace]({URL}/ent/Proj/workspace)' in run.notes
    t.finish()


def test_panel_link_without_a_glossary_uses_the_default_title(monkeypatch, tmp_path, fake_wandb):
    t, run = _start(monkeypatch, tmp_path, fake_wandb, glossary_panel='V')
    assert f'"{G.DEFAULT_TITLE}" at the top of the [saved view "V"]({URL}/ent/Proj?nw=v)' in run.notes
    assert run.artifacts == [] and 'artifact' not in run.notes
    t.finish()


def test_a_bad_glossary_panel_is_reported_not_raised(monkeypatch, capsys, tmp_path, fake_wandb, glossary):
    t, run = _start(monkeypatch, tmp_path, fake_wandb, glossary=glossary, glossary_panel=42)
    assert t.active and 'glossary_panel ignored' in capsys.readouterr().out
    assert 'at the top of the' not in run.notes and 'artifact [metric\\_glossary:' in run.notes
    t.finish()


def test_panel_link_helper():
    assert T.panel_link('http://h/', 'e', 'p', None) == (None, None)
    assert T.panel_link('http://h/', 'e', 'p', T.PERSONAL_WORKSPACE) == ('http://h/e/p/workspace', 'project workspace')
    assert T.panel_link('http://h', 'e', 'p', 'P — described') == ('http://h/e/p?nw=pdescribed', 'saved view "P — described"')
    assert T.panel_link('', 'e', 'p', 'V') == (None, None) and T.panel_link('http://h', None, 'p', T.PERSONAL_WORKSPACE) == (None, None)
    with pytest.raises(ValueError):
        T.panel_link('http://h', 'e', 'p', 3)
    with pytest.raises(ValueError):
        T.panel_link('http://h', 'e', 'p', '— —')
    assert T.shared_view_name('P — described') == 'nw-pdescribed-v'


def test_backfill_panel_link_is_explicit_and_panel_title_alone_keeps_the_workspace_link():
    assert B._panel('http://h', 'e', 'p', None, None) == (None, None, None)
    assert B._panel('http://h', 'e', 'p', 'Title', None) == ('http://h/e/p/workspace', 'Title', 'project workspace')
    assert B._panel('http://h', 'e', 'p', None, 'V') == ('http://h/e/p?nw=v', G.DEFAULT_TITLE, 'saved view "V"')
    assert B._panel('http://h', 'e', 'p', 'Title', 'V') == ('http://h/e/p?nw=v', 'Title', 'saved view "V"')
    url, title, where = B._panel('http://h', 'e', 'p', None, 'V')
    md = B.backfill_notes({'description': 'D.'}, name='e', run_dir='/tmp', commit=None, branch=None, host='h',
                          backfilled=True, dirty=None, shown_git_commit=None, workspace=url, panel_title=title,
                          panel_where=where)
    assert f'"{G.DEFAULT_TITLE}" at the top of the [saved view "V"](http://h/e/p?nw=v)' in md


def test_null_tracker_has_the_tracker_surface_and_does_nothing():
    n = T.NullTracker('rank 1')
    assert not n.active and n.mode is None and n.dropped == 0 and n.reason == 'rank 1' and n.undocumented == []
    assert n.train_step(1, {'x': 1.0}) is None and n.eval_step(1, {'y': 1.0}, model=object()) is None
    assert n.log({'z': 1.0}, 1) is None and n.finish({'s': 1.0}) is None
    public = {m for m in dir(T.Tracker) if not m.startswith('_')} - {'start'}
    assert public <= set(dir(n)), public - set(dir(n))


def _my_glossary_module(monkeypatch):
    """The `my_glossary` module IMPORT_GUARD imports: a DictGlossary, as a project would define it."""
    mod = types.ModuleType('my_glossary')
    mod.GLOSSARY = G.DictGlossary({'train/loss': dict(unit='nats', desc='Step loss.')})
    monkeypatch.setitem(sys.modules, 'my_glossary', mod)
    return mod


@pytest.mark.parametrize('blocked', ['spiky.util.wandb_integration.tracker', 'my_glossary'])
def test_import_guard_survives_this_package_being_unimportable(monkeypatch, capsys, blocked):
    _my_glossary_module(monkeypatch)
    monkeypatch.setitem(sys.modules, blocked, None)                          # that import now raises
    ns = dict(cfg={}, exp_dir='/tmp', PROJECT='P', GROUP='g')
    exec(T.IMPORT_GUARD, ns)
    tracker = ns['tracker']
    assert not tracker.active and '[wandb] off: ModuleNotFoundError' in capsys.readouterr().out
    assert tracker.train_step(1, {'a': 1.0}) is None and tracker.eval_step(1, {}, None) is None
    assert tracker.log({'b': 1.0}, 2) is None and tracker.finish({}) is None


def test_import_guard_uses_the_real_tracker_when_importable(monkeypatch, fake_wandb, tmp_path):
    mod = _my_glossary_module(monkeypatch)
    monkeypatch.setenv('WANDB_BASE_URL', URL)
    monkeypatch.setenv('WANDB_MODE', 'offline')
    ns = dict(cfg={}, exp_dir=str(tmp_path), PROJECT='P', GROUP='g')
    exec(T.IMPORT_GUARD, ns)
    tracker = ns['tracker']
    assert isinstance(tracker, T.Tracker) and tracker.active and tracker.glossary is mod.GLOSSARY
    assert fake_wandb.runs[0].init['project'] == 'P'
    tracker.finish()
