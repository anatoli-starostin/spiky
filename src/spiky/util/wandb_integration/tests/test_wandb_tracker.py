"""Tracker: the loop never waits on or breaks because of wandb, the drift check is never fatal, project / group / tags /
metrics / glossary are injected, and the notes / config helpers behave. CPU only, no server (a fake wandb module).

    python -m pytest src/spiky/util/wandb_integration/tests -q
"""
import os
import time

import pytest

from spiky.util.wandb_integration import tracker as T

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = os.path.abspath(os.path.join(HERE, '..', '..'))                         # src/spiky/util, committed on main


def _drain(t):
    t._q.put(None)
    t._th.join(5)


def _online_env(monkeypatch):
    monkeypatch.setenv('WANDB_BASE_URL', 'http://wandb.invalid:8080')
    monkeypatch.setenv('WANDB_MODE', 'offline')                                # no server probe


def test_drift_check_flags_unknown_keys_once(capsys, fakes, glossary):
    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline', glossary=glossary)
    t.train_step(1, {'train/loss': 3.0})
    t.eval_step(500, {'val/loss': 1.2, 'norm_L0': 1.0, 'brand_new': 2.0})
    t.eval_step(1000, {'val/loss': 1.1, 'norm_L0': 1.0, 'brand_new': 2.0, 'other_new': 1.0})
    _drain(t)
    assert T.UNDOC_TAG in run.tags and run.tags.count(T.UNDOC_TAG) == 1
    assert run.summary[T.UNDOC_SUMMARY] == 'brand_new, other_new'
    out = capsys.readouterr().out
    assert out.count('brand_new') == 1 and out.count('other_new') == 1
    assert len(run.logged) == 3


def test_drift_check_is_never_fatal(fakes, glossary):
    run = fakes.BrokenTagsRun()
    t = T.Tracker(run, wandb=None, mode='offline', glossary=glossary)
    t.eval_step(500, {'val/loss': 1.2, 'brand_new': 2.0})
    t.eval_step(1000, {'val/loss': 1.1})
    _drain(t)
    assert t.run is run and len(run.logged) == 2                  # still logging, not disabled


def test_documented_rows_raise_no_flag_and_extra_eval_metrics_merge(capsys, fakes, glossary):
    run, seen = fakes.Run(), []

    def per_layer(model):
        seen.append(model)
        return {'norm_L0': 1.0, 'norm_L1': 2.0}

    t = T.Tracker(run, wandb=None, mode='offline', glossary=glossary, extra_eval_metrics=per_layer)
    t.train_step(10, {'train/loss': 3.0})
    t.eval_step(500, {'val/loss': 1.2}, model='M')
    t.eval_step(600, {'val/loss': 1.1})                           # no model: no extra metrics
    _drain(t)
    assert T.UNDOC_TAG not in run.tags and T.UNDOC_SUMMARY not in run.summary
    assert 'glossary' not in capsys.readouterr().out
    assert run.logged[1] == (500, {'val/loss': 1.2, 'norm_L0': 1.0, 'norm_L1': 2.0})
    assert run.logged[2] == (600, {'val/loss': 1.1}) and seen == ['M']


def test_without_a_glossary_there_is_no_drift_check(capsys, fakes):
    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline')
    t.eval_step(1, {'anything/at_all': 1.0})
    _drain(t)
    assert run.logged == [(1, {'anything/at_all': 1.0})] and T.UNDOC_TAG not in run.tags
    assert capsys.readouterr().out == ''


def test_train_step_throttles_converts_and_times(fakes):
    class Tensorish:
        def __float__(self):
            return 0.5

    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline', log_every=5)
    for s in range(1, 12):
        t.train_step(s, {'train/loss': 1.0, 'grad_norm': Tensorish(), 'not_a_number': object()})
    _drain(t)
    assert [s for s, _ in run.logged] == [1, 5, 10]
    row = run.logged[1][1]
    assert row['grad_norm'] == 0.5 and 'not_a_number' not in row and row[T.TIMING_KEY] >= 0
    run2 = fakes.Run()
    t2 = T.Tracker(run2, wandb=None, mode='offline', timing_key=None)
    t2.train_step(1, {'train/loss': 1.0})
    _drain(t2)
    assert run2.logged == [(1, {'train/loss': 1.0})]


def test_an_extra_eval_metrics_error_disables_the_tracker_without_raising(capsys, fakes):
    def boom(model):
        raise RuntimeError('no such layer')

    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline', extra_eval_metrics=boom)
    t.eval_step(1, {'val/loss': 1.0}, model=object())
    assert not t.active and 'disabled after an error in eval_step' in capsys.readouterr().out
    t.train_step(1, {'train/loss': 1.0})
    t.eval_step(2, {'val/loss': 1.0})
    t.finish({'final_loss': 1.0})
    _drain(t)
    assert run.logged == [] and run.summary == {}


def test_a_full_queue_drops_rows_and_never_blocks(monkeypatch, fakes):
    monkeypatch.setattr(T, 'LOG_QUEUE_MAX', 3)
    run = fakes.BlockingRun()
    t = T.Tracker(run, wandb=None, mode='offline')
    t0 = time.time()
    for s in range(50):
        t.log({'x': float(s)}, s)
    assert time.time() - t0 < 0.5                                 # the loop never waited
    assert t.dropped >= 50 - 3 - 1
    run.release.set()
    _drain(t)
    assert len(run.logged) == 50 - t.dropped


def test_finish_gives_up_at_its_deadline(monkeypatch, capsys, fakes, fake_wandb):
    monkeypatch.setattr(T, 'FINISH_TIMEOUT', 0.2)
    monkeypatch.setattr(T, 'FINISH_GRACE', 0.3)
    fake_wandb.finish_delay = 30                                  # a wandb.finish() that hangs
    run = fakes.Run(dir='/tmp/x')
    t = T.Tracker(run, wandb=fake_wandb, mode='online')
    t.log({'val/loss': 1.0}, 1)
    t0 = time.time()
    t.finish({'final_loss': 1.0, 'nested': {'not': 'a scalar'}})
    assert time.time() - t0 < 3
    assert 'finish() gave up' in capsys.readouterr().out and not t.active
    assert run.logged == [(1, {'val/loss': 1.0})] and run.summary == {'final_loss': 1.0}


def test_start_is_off_without_server_project_or_when_disabled(monkeypatch, fake_wandb):
    assert T.Tracker.start({}, '/tmp', project='P').reason == 'no WANDB_BASE_URL'
    monkeypatch.setenv('WANDB_BASE_URL', 'http://wandb.invalid:8080')
    assert T.Tracker.start({}, '/tmp').reason == 'no project'
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    assert T.Tracker.start({}, '/tmp', project='P').reason == 'disabled'
    assert fake_wandb.runs == []


def test_start_wires_the_injected_project_group_tags_config_metrics_and_glossary(monkeypatch, capsys, fake_wandb,
                                                                                 glossary, tmp_path):
    _online_env(monkeypatch)
    monkeypatch.setenv('WANDB_ENTITY', 'ent')
    monkeypatch.setenv('WANDB_RUN_GROUP', 'family')
    exp = tmp_path / 'exp_7'
    exp.mkdir()
    cfg = {'lr': 3e-4, 'legacy': 10, 'form': 'x', 'description': 'Tests a thing.', '_arch_note': 'Long note.'}
    t = T.Tracker.start(cfg, str(exp), project='Throwaway', tags=['static'], extra_tags=lambda c: [f"form:{c['form']}"],
                        extra_eval_metrics=lambda m: {'norm_L0': 1.0}, glossary=glossary,
                        config_extra={'total_params': 123}, config_renames={'legacy': 'legacy_ignored'}, log_every=2)
    assert t.active and t.mode == 'offline' and t.log_every == 2
    run = fake_wandb.runs[0]
    kw = run.init
    assert (kw['project'], kw['entity'], kw['group'], kw['job_type']) == ('Throwaway', 'ent', 'family', 'train')
    assert kw['name'] == kw['id'] == 'exp_7' and kw['resume'] == 'allow' and kw['mode'] == 'offline'
    assert kw['tags'][0] == 'family' and 'static' in kw['tags'] and 'form:x' in kw['tags']
    c = kw['config']
    assert c['legacy_ignored'] == 10 and 'legacy' not in c and c['total_params'] == 123 and c['lr'] == 3e-4
    assert 'description' not in c and '_arch_note' not in c and {'branch', 'commit', 'commit_dirty', 'host'} <= set(c)
    assert kw['settings']['console'] == 'off' and kw['settings']['finish_timeout'] == T.FINISH_TIMEOUT
    art, aliases = run.artifacts[0]
    assert aliases == ['latest', f'glossary-{glossary.glossary_hash()}']
    assert art.files['glossary'].data == glossary.table_rows() and art.metadata['source'] == glossary.SOURCE
    assert 'Tests a thing.' in run.notes
    assert f'artifact [metric\\_glossary:glossary-{glossary.glossary_hash()}]' in run.notes
    assert 'at the top of the' not in run.notes                  # no panel link unless glossary_panel asks for one
    t.eval_step(2, {'val/loss': 1.0}, model=object())
    t.finish({'final_loss': 1.0})
    assert run.logged == [(2, {'val/loss': 1.0, 'norm_L0': 1.0})] and run.summary == {'final_loss': 1.0}
    assert fake_wandb.finished == 1 and 'undocumented' not in capsys.readouterr().out


def test_start_survives_a_failing_extra_tags_and_an_incomplete_glossary(monkeypatch, capsys, fake_wandb, tmp_path):
    _online_env(monkeypatch)

    def bad_tags(cfg):
        raise KeyError('form')

    t = T.Tracker.start({}, str(tmp_path), project='P', extra_tags=bad_tags, glossary=object())
    out = capsys.readouterr().out
    assert t.active and 'extra_tags failed' in out
    assert 'glossary ignored: it lacks undocumented, glossary_hash, table_rows' in out
    assert t.glossary is None and fake_wandb.runs[0].artifacts == [] and 'Metric glossary' not in fake_wandb.runs[0].notes
    t.finish()


def test_github_web_url():
    assert T.github_web_url('git@github-spikybot:owner/repo.git') == 'https://github.com/owner/repo'
    assert T.github_web_url('git@github.com:owner/repo') == 'https://github.com/owner/repo'
    assert T.github_web_url('https://github.com/owner/repo.git') == 'https://github.com/owner/repo'
    assert T.github_web_url('ssh://git@github.com/owner/repo.git') == 'https://github.com/owner/repo'
    assert T.github_web_url('git@gitlab.com:owner/repo.git') is None
    assert T.github_web_url('unknown') is None and T.github_web_url(None) is None


def test_host_and_config_normalisation():
    assert T.normalise_host('pasta-gpustar') == 'gpustar'
    assert T.normalise_host('gpustar') == 'gpustar' and T.normalise_host(None) is None
    cfg = {'eval_steps': 10, 'lr': 3e-4, '_arch_note': 'long', 'description': 'short'}
    c = T.wandb_config(cfg, {'host': 'gpustar'}, renames={'eval_steps': 'eval_steps_legacy_ignored'})
    assert c == {'eval_steps_legacy_ignored': 10, 'lr': 3e-4, 'host': 'gpustar'}
    assert T.wandb_config(cfg) == {'eval_steps': 10, 'lr': 3e-4}


def test_description_text():
    assert T.description_text({'description': 'Short.', '_arch_note': 'Long note.'}) == 'Short.'
    note = 'First sentence. Second one here. Third (short). Fourth. Fifth should not appear.'
    assert T.description_text({'_arch_note': note}) == 'First sentence. Second one here. Third (short). Fourth.'
    assert T.description_text({'_arch_note': 'A' * 3000}).endswith('...')
    assert T.description_text({}) == ''


def test_run_notes_links_and_flags():
    sha = 'a' * 40
    info = dict(web='https://github.com/o/r', sha=sha, branch='research/x', dirty=True,
                rel='experiments/f/runs/exp_1', committed=False)
    cfg = {'_arch_note': 'What it tests. Why. How it differs. The fourth. The fifth is not in the summary. *star* x_y'}
    ws = T.workspace_url('http://h/', 'e', 'p')
    assert ws == 'http://h/e/p/workspace' and T.workspace_url('', 'e', 'p') is None
    md = T.run_notes(cfg, exp_name='exp_1', info=info, host='gpustar', workspace_url=ws, panel_title='About these metrics',
                     artifact_url='http://h/art', artifact_label='metric_glossary:glossary-abc')
    head, _, tail = md.partition('\n---\n')
    assert head.startswith('**exp\\_1**')
    assert f'(https://github.com/o/r/tree/{sha}/experiments/f/runs/exp_1)' in head
    assert '(https://github.com/o/r/tree/research/x/experiments/f/runs/exp_1)' in head
    assert 'dirty' in head and 'not committed at launch' in head
    assert '`experiments/f/runs/exp_1` on `gpustar`' in head
    assert '"About these metrics" at the top of the [project workspace](http://h/e/p/workspace)' in head
    assert '[metric\\_glossary:glossary-abc](http://h/art)' in head
    assert 'fifth' not in head and 'The fifth' in tail
    assert '\\*star\\* x\\_y' in tail                              # _arch_note rendered literally
    assert 'project workspace' not in T.run_notes(cfg, exp_name='e', info=info, host='h', workspace_url=ws)


def test_run_notes_without_git():
    md = T.run_notes({'description': 'D.'}, exp_name='e', info=dict(), host='h')
    assert md.startswith('**e**') and 'github' not in md and '---' not in md and 'Metric glossary' not in md


def test_git_launch_info_on_this_checkout():
    info = T.git_launch_info(UTIL)
    if info['sha'] is None:
        pytest.skip('not a git checkout')
    assert len(info['sha']) == 40
    assert info['rel'] == os.path.relpath(UTIL, info['root']) and not info['rel'].startswith('..')
    assert info['committed'] is True
    outside = T.git_launch_info('/tmp')
    assert outside['rel'] is None and outside['committed'] is None
    assert T.git_launch_info('/tmp', code_dir=UTIL)['root'] == info['root']
