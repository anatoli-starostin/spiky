"""Stubs for the wandb_integration tests: a stub glossary (undocumented + legend_markdown), fake wandb runs and a fake
`wandb` module. CPU only: no server, no real wandb calls, no project content."""
import os
import re
import sys
import threading
import time
import types

import pytest


class StubGlossary:
    SOURCE = 'tests/conftest.py'
    METRICS = {                                                              # key -> (unit, description)
        'train/loss': ('nats/tok', 'Training loss of the step.'),
        'time/sec_per_step': ('s', 'Wall seconds per step.'),
        'val/loss': ('nats/tok', 'Validation loss.'),
        'norm_L{i}': ('L2', 'Weight norm of layer i.'),
        'final_loss': ('nats/tok', 'val/loss at the last eval.'),
    }

    def _entry(self, key):
        if key in self.METRICS:
            return key
        for k in self.METRICS:
            if '{i}' in k and re.fullmatch(re.escape(k).replace(re.escape('{i}'), r'\d+'), key):
                return k
        return None

    def undocumented(self, keys):
        return sorted({k for k in keys if not (k.startswith('_') or k.startswith('system/') or self._entry(k))})

    def legend_markdown(self):
        rows = ''.join(f'| `{k}` | {d} [{u}] |\n' for k, (u, d) in sorted(self.METRICS.items()))
        return f'### Metrics\n\n| key | what it measures [unit] |\n|---|---|\n{rows}'


class FakeRun:
    def __init__(self, **init):
        self.init = init
        self._tags = tuple(init.get('tags') or ('a',))
        self.summary, self.logged, self.artifacts = {}, [], []
        self.notes = init.get('notes')
        self.project, self.entity = init.get('project'), init.get('entity')
        self.dir = os.path.join(init.get('dir') or '/tmp', 'run-fake', 'files')

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, v):
        self._tags = tuple(v)

    def log(self, row, step=None):
        self.logged.append((step, dict(row)))

    def log_artifact(self, art, aliases=()):                                # must never be called any more
        self.artifacts.append((art, list(aliases)))


class BlockingRun(FakeRun):
    """run.log() blocks until release is set: a server that stopped accepting rows."""

    def __init__(self, **init):
        super().__init__(**init)
        self.release = threading.Event()

    def log(self, row, step=None):
        self.release.wait(10)
        super().log(row, step)


class FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__('wandb')
        self.runs, self.finished, self.finish_delay = [], 0, 0.0

    def Settings(self, **kw):
        return dict(kw)

    def init(self, **kw):
        run = FakeRun(**kw)
        self.runs.append(run)
        return run

    def finish(self):
        time.sleep(self.finish_delay)
        self.finished += 1


@pytest.fixture
def glossary():
    return StubGlossary()


@pytest.fixture
def fakes():
    return types.SimpleNamespace(Run=FakeRun, BlockingRun=BlockingRun)


@pytest.fixture
def fake_wandb(monkeypatch, tmp_path):
    """`import wandb` returns a fake; the wandb environment is cleared and its directories point into tmp_path."""
    fw = FakeWandb()
    monkeypatch.setitem(sys.modules, 'wandb', fw)
    for var in ('WANDB_BASE_URL', 'WANDB_MODE', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_RUN_GROUP'):
        monkeypatch.delenv(var, raising=False)
    for var, sub in (('WANDB_DIR', 'wandb'), ('WANDB_CONFIG_DIR', 'config'), ('WANDB_DATA_DIR', 'data')):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return fw
