"""Backfill: metrics.csv rows -> wandb rows with the same notes blob (legend included), run-specific values injected,
incomplete runs refused, and the notes-only path sends nothing but notes. CPU only, no server (fake wandb and API)."""
import json
import os
import types

import pytest

from spiky.util.wandb_integration import backfill as B
from spiky.util.wandb_integration import tracker as T

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = os.path.abspath(os.path.join(HERE, '..', '..'))                         # src/spiky/util, committed on main
ROWS = ((1, '', 3.0), (500, '1.5', 2.5), (1000, '1.4', 2.4))


def _run_dir(tmp_path, name='exp_3', rows=ROWS, summary=True):
    d = tmp_path / name
    d.mkdir()
    (d / 'config.json').write_text(json.dumps({'exp_name': name, 'n_steps': 1000, 'legacy': 1,
                                               'description': 'Backfill test.'}))
    (d / 'metrics.csv').write_text('\n'.join(['step,val/loss,train_loss'] + [f'{s},{v},{t}' for s, v, t in rows]) + '\n')
    if summary:
        (d / 'summary.json').write_text(json.dumps({'final_loss': 1.4, 'eval_protocol': {'x': 1}}))
    return d


def _last_step_is_n_steps(rows, cfg):
    return rows[-1][0] == cfg['n_steps'], f'last step {rows[-1][0]} of {cfg["n_steps"]}'


def test_read_metrics(tmp_path):
    d = _run_dir(tmp_path)
    assert B.read_metrics(d, require_col='val/loss') == [(500, {'val/loss': 1.5, 'train_loss': 2.5}),
                                                         (1000, {'val/loss': 1.4, 'train_loss': 2.4})]
    every = B.read_metrics(d)
    assert [s for s, _ in every] == [1, 500, 1000] and every[0][1] == {'train_loss': 3.0}


def test_backfill_run_logs_rows_summary_tags_config_and_the_legend(monkeypatch, capsys, fake_wandb, glossary, tmp_path):
    monkeypatch.setenv('WANDB_BASE_URL', 'http://wandb.invalid')
    d = _run_dir(tmp_path)
    out = B.backfill_run(str(d), project='Throwaway', entity='ent', group='family', branch='research/x', host='pasta-box',
                         tags=['line:a'], config_extra={'total_params': 7}, config_renames={'legacy': 'legacy_ignored'},
                         summary_keys=('final_loss', 'missing'), require_col='val/loss',
                         is_complete=_last_step_is_n_steps, glossary=glossary, wandb=fake_wandb)
    run = fake_wandb.runs[0]
    kw = run.init
    assert out == os.path.dirname(run.dir) and fake_wandb.finished == 1
    assert run.logged == [(500, {'val/loss': 1.5, 'train_loss': 2.5}), (1000, {'val/loss': 1.4, 'train_loss': 2.4})]
    assert run.summary == {'final_loss': 1.4}
    assert kw['tags'] == ['backfilled', 'family', 'research/x', 'box', 'line:a']   # commit unknown: not a git checkout
    assert (kw['project'], kw['entity'], kw['group'], kw['name'], kw['id'], kw['mode']) == (
        'Throwaway', 'ent', 'family', 'exp_3', 'exp_3', 'offline')
    c = kw['config']
    assert c['backfilled'] is True and c['host'] == 'box' and c['total_params'] == 7 and c['branch'] == 'research/x'
    assert c['legacy_ignored'] == 1 and 'legacy' not in c and 'description' not in c
    assert c['backfill_source'] == 'metrics.csv + summary.json' and c['commit_kind'] == 'artefacts commit'
    assert '**Backfilled**' in kw['notes'] and 'Backfill test.' in kw['notes']
    assert glossary.legend_markdown().rstrip('\n') in kw['notes'] and run.artifacts == []
    assert 'backfilled exp_3: 2 rows' in capsys.readouterr().out


def test_backfill_refuses_incomplete_runs(monkeypatch, capsys, fake_wandb, tmp_path):
    monkeypatch.setenv('WANDB_BASE_URL', 'http://wandb.invalid')
    short = _run_dir(tmp_path, rows=ROWS[:2])
    assert B.backfill_run(str(short), project='P', require_col='val/loss', is_complete=_last_step_is_n_steps,
                          wandb=fake_wandb) is None
    assert 'SKIP exp_3: metrics.csv incomplete (last step 500 of 1000' in capsys.readouterr().out
    empty = _run_dir(tmp_path, name='exp_4', rows=ROWS[:1], summary=False)
    assert B.backfill_run(str(empty), project='P', require_col='val/loss', wandb=fake_wandb) is None
    assert fake_wandb.runs == []
    with pytest.raises(ValueError):
        B.backfill_run(str(short), project='P', mode='sideways', wandb=fake_wandb)


def test_backfill_notes_flag_the_git_state_of_backfilled_runs(tmp_path):
    md = B.backfill_notes({'description': 'D.'}, name='e', run_dir=UTIL, commit='HEAD', branch='main', host='h',
                          backfilled=True, dirty=None, shown_git_commit='f' * 40, legend='### Metrics\n')
    if 'Code + artefacts' not in md:
        pytest.skip('not a git checkout')
    assert '**Backfilled**' in md and "wandb's **Git state** on this run shows `ffffffff`" in md
    assert md.index("Git state") < md.index('### Metrics')
    live = B.backfill_notes({'description': 'D.'}, name='e', run_dir=str(tmp_path), commit=None, branch=None, host='h',
                            backfilled=False, dirty=None, shown_git_commit=None)
    assert 'Backfilled' not in live and 'Git state' not in live and live.startswith('**e**\n\nD.')
    assert live.endswith(T.NO_LEGEND_LINE)


class _Api:
    """wandb.Api stand-in: one run on the server; GraphQL upsertBucket stores notes, the notes query reads them."""

    def __init__(self):
        self.notes, self.paths, self.mutations = {}, [], 0
        self._service_api = types.SimpleNamespace(execute_graphql=self._gql)

    def run(self, path):
        self.paths.append(path)
        return types.SimpleNamespace(name=path.rsplit('/', 1)[1], storage_id='S1', tags=['backfilled'], metadata={},
                                     config={'commit': None, 'branch': 'research/x', 'host': 'pasta-box'})

    def _gql(self, query, variables, timeout=None):
        if 'upsertBucket' in query:
            self.mutations += 1
            self.notes[variables['id']] = variables['notes']
            return {'upsertBucket': {'bucket': {'id': variables['id']}}}
        return {'project': {'run': {'notes': self.notes.get('S1')}}}


def test_update_notes_gives_an_existing_run_its_legend_and_writes_only_notes(capsys, glossary, tmp_path):
    d = _run_dir(tmp_path)
    api = _Api()
    dry = B.update_notes(api, 'ent', 'P', {'exp_3': str(d)}, glossary=glossary, descriptions={'exp_3': 'Override.'},
                         dry_run=True)
    assert api.mutations == 0 and 'Override.' in dry['exp_3'] and glossary.legend_markdown().rstrip('\n') in dry['exp_3']
    assert '===== exp_3 (backfilled)' in capsys.readouterr().out
    assert B.update_notes(api, 'ent', 'P', {'exp_3': str(d)}, glossary=glossary) == {'exp_3': True}
    assert api.mutations == 1 and api.paths == ['ent/P/exp_3', 'ent/P/exp_3'] and 'Backfill test.' in api.notes['S1']
    assert '| `val/loss` |' in api.notes['S1']
