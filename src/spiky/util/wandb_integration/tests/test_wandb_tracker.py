"""Tracker: the loop never waits on or breaks because of wandb; one notes blob carries the description and the metric
legend; every logged key is checked against the legend in-process (at first log and at finish), loudly and never
fatally, also when tracking is off; project / group / tags / metrics are injected. CPU only, no server (a fake wandb).

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


def test_an_undescribed_key_is_reported_at_its_first_log_once(capsys, fakes, glossary):
    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline', glossary=glossary)
    t.train_step(1, {'train/loss': 3.0, 'brand_new': 1.0})
    assert 'UNDESCRIBED METRIC: brand_new' in capsys.readouterr().out          # surfaces at step 1, not at the end
    t.eval_step(500, {'val/loss': 1.2, 'norm_L0': 1.0, 'brand_new': 2.0, 'other_new': 1.0})
    t.eval_step(1000, {'val/loss': 1.1, 'brand_new': 2.0, 'other_new': 1.0})
    _drain(t)
    out = capsys.readouterr().out
    assert 'brand_new' not in out and out.count('other_new') == 1
    assert t.undocumented == ['brand_new', 'other_new'] and len(run.logged) == 3
    assert run.tags == ('a',) and run.summary == {} and run.artifacts == []   # no drift tag, no summary key, no artifact


def test_finish_prints_a_loud_banner_listing_every_undescribed_key(capsys, fakes, fake_wandb, glossary):
    run = fakes.Run(dir='/tmp/x')
    t = T.Tracker(run, wandb=fake_wandb, mode='offline', glossary=glossary)
    t.train_step(1, {'train/loss': 3.0, 'mystery/a': 1.0})
    t.finish({'final_loss': 1.0, 'mystery_summary': 2.0})
    out = capsys.readouterr().out
    assert '!' * 100 in out and 'UNDESCRIBED METRICS in this run (2): mystery/a, mystery_summary' in out
    assert run.summary == {'final_loss': 1.0, 'mystery_summary': 2.0} and fake_wandb.finished == 1   # still uploaded


def test_the_legend_check_runs_when_tracking_is_off(capsys, fake_wandb, glossary):
    t = T.Tracker.start({}, '/tmp', project='P', glossary=glossary)            # WANDB_BASE_URL unset -> OFF
    assert not t.active and t.reason == 'no WANDB_BASE_URL'
    for s in range(1, 12):
        t.train_step(s, {'train/loss': 1.0, 'not_described': 1.0})
    t.eval_step(10, {'val/loss': 1.0})
    t.finish({'final_loss': 1.0})
    out = capsys.readouterr().out
    assert out.count('UNDESCRIBED METRIC: not_described') == 1 and 'UNDESCRIBED METRICS in this run (1)' in out
    assert fake_wandb.runs == []


def test_without_a_glossary_every_key_but_wandbs_own_is_undescribed(capsys, fakes):
    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline')
    t.log({'loss': 1.0, '_step': 1, 'system/gpu.0.gpu': 3.0}, 1)
    _drain(t)
    assert t.undocumented == ['loss'] and len(run.logged) == 1


def test_the_legend_check_is_never_fatal(capsys, fakes):
    class Broken:
        def undocumented(self, keys):
            raise RuntimeError('glossary bug')

        def legend_markdown(self):
            return ''

    run = fakes.Run()
    t = T.Tracker(run, wandb=None, mode='offline', glossary=Broken())
    t.eval_step(1, {'val/loss': 1.0})
    t.eval_step(2, {'val/loss': 0.9})
    _drain(t)
    assert t.active and len(run.logged) == 2 and 'legend check skipped' in capsys.readouterr().out


def test_documented_rows_raise_no_flag_and_extra_eval_metrics_merge(capsys, fakes, glossary):
    run, seen = fakes.Run(), []

    def per_layer(model):
        seen.append(model)
        return {'norm_L0': 1.0, 'norm_L1': 2.0}

    t = T.Tracker(run, wandb=None, mode='offline', glossary=glossary, extra_eval_metrics=per_layer)
    t.train_step(10, {'train/loss': 3.0})
    t.eval_step(500, {'val/loss': 1.2}, model='M')
    t.eval_step(600, {'val/loss': 1.1})                           # no model: no extra metrics
    t.finish()
    _drain(t)
    assert t.undocumented == [] and 'UNDESCRIBED' not in capsys.readouterr().out
    assert run.logged[1] == (500, {'val/loss': 1.2, 'norm_L0': 1.0, 'norm_L1': 2.0})
    assert run.logged[2] == (600, {'val/loss': 1.1}) and seen == ['M']


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


def test_start_writes_one_notes_blob_with_the_legend_and_no_artifact(monkeypatch, capsys, fake_wandb, glossary,
                                                                      tmp_path):
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
    notes = run.notes
    assert notes.index('Tests a thing.') < notes.index(glossary.legend_markdown().rstrip('\n')) < notes.index('Long note.')
    assert '| `val/loss` | Validation loss. [nats/tok] |' in notes and T.NO_LEGEND_LINE not in notes
    assert run.artifacts == [] and not hasattr(T, 'log_glossary_artifact')
    t.eval_step(2, {'val/loss': 1.0}, model=object())
    t.finish({'final_loss': 1.0})
    assert run.logged == [(2, {'val/loss': 1.0, 'norm_L0': 1.0})] and run.summary == {'final_loss': 1.0}
    assert fake_wandb.finished == 1 and 'UNDESCRIBED' not in capsys.readouterr().out


def test_start_survives_a_failing_extra_tags_and_an_incomplete_glossary(monkeypatch, capsys, fake_wandb, tmp_path):
    _online_env(monkeypatch)

    def bad_tags(cfg):
        raise KeyError('form')

    t = T.Tracker.start({}, str(tmp_path), project='P', extra_tags=bad_tags, glossary=object())
    out = capsys.readouterr().out
    assert t.active and 'extra_tags failed' in out
    assert 'glossary ignored: it lacks undocumented, legend_markdown' in out and 'no glossary given' in out
    assert t.glossary is None and T.NO_LEGEND_LINE in fake_wandb.runs[0].notes
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


def test_run_notes_links_flags_and_legend():
    sha = 'a' * 40
    info = dict(web='https://github.com/o/r', sha=sha, branch='research/x', dirty=True,
                rel='experiments/f/runs/exp_1', committed=False)
    cfg = {'_arch_note': 'What it tests. Why. How it differs. The fourth. The fifth is not in the summary. *star* x_y'}
    legend = '### Metrics\n\n| key | what it measures [unit] |\n|---|---|\n| `loss` | The loss. [nats/tok] |\n'
    md = T.run_notes(cfg, exp_name='exp_1', info=info, host='gpustar', legend=legend)
    head, _, tail = md.partition('\n---\n')
    assert head.startswith('**exp\\_1**')
    assert f'(https://github.com/o/r/tree/{sha}/experiments/f/runs/exp_1)' in head
    assert '(https://github.com/o/r/tree/research/x/experiments/f/runs/exp_1)' in head
    assert 'dirty' in head and 'not committed at launch' in head
    assert '`experiments/f/runs/exp_1` on `gpustar`' in head
    assert tail.lstrip('\n').startswith('### Metrics') and 'fifth' not in head and 'The fifth' in md
    assert md.index('| `loss` |') < md.index('**Architecture note**') and '\\*star\\* x\\_y' in md
    assert 'workspace' not in md and 'artifact' not in md


def test_run_notes_without_git_or_legend():
    md = T.run_notes({'description': 'D.'}, exp_name='e', info=dict(), host='h')
    assert md.startswith('**e**') and 'github' not in md and '---' not in md and md.endswith(T.NO_LEGEND_LINE)


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
