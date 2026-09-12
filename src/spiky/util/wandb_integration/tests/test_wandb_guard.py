"""The shipped NullTracker and the documented import guard for this package failing to import. CPU only, no server."""
import sys
import types

import pytest

from spiky.util.wandb_integration import glossary as G
from spiky.util.wandb_integration import tracker as T

URL = 'http://wandb.invalid:8080'


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
    mod.GLOSSARY = G.DictGlossary({'train/loss': dict(unit='nats/tok', desc='Step loss.')})
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
    assert fake_wandb.runs[0].init['project'] == 'P' and '| `train/loss` |' in fake_wandb.runs[0].notes
    tracker.finish()
