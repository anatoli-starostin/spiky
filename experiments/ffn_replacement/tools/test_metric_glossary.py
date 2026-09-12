"""CPU-only, no server: the metric glossary covers everything we log, the workspace panel is built and placed
correctly (idempotent, other sections untouched), and the tracker's glossary / notes helpers behave (drift check
never fatal, links built from the git remote, legacy config renamed).

    python -m pytest experiments/ffn_replacement/tools/test_metric_glossary.py -q
"""
import copy
import csv
import glob
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metric_glossary as MG                                                     # noqa: E402
import wandb_glossary as WG                                                        # noqa: E402
import wandb_tracking as WT                                                        # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def _metrics_csv_headers():
    out = {}
    for p in glob.glob(os.path.join(REPO, '**', 'metrics.csv'), recursive=True):
        if os.sep + '.venv' + os.sep in p:
            continue
        with open(p, newline='') as f:
            for h in next(csv.reader(f), []):
                out.setdefault(h, os.path.relpath(p, REPO))
    return out


def test_every_metrics_csv_header_in_the_repo_is_documented():
    headers = _metrics_csv_headers()
    assert headers, 'no metrics.csv found in the repo'
    missing = {h: p for h, p in headers.items() if MG.entry_key(h) is None}
    assert not missing, f'metrics.csv columns with no glossary entry (add them to metric_glossary.py): {missing}'


class _Mod:
    confidence_form = 'learned_margin'

    def learned_confidence_values(self):
        return {'g': 0.0, 'beta': 2.0, 'gamma': 1.0}


class _Model:
    def modules(self):
        return [object(), _Mod(), _Mod()]


def test_tracker_fixed_keys_are_documented():
    keys = list(WT.TRAIN_KEYS) + list(WT.EVAL_KEYS) + list(WT.SUMMARY_KEYS)
    keys += list(WT.learned_confidence_by_layer(_Model()))
    assert 'lm_gamma_L1' in keys
    assert not [k for k in keys if MG.entry_key(k) is None]


def test_patterns_match_layer_indices_only():
    assert MG.entry_key('ln2_norm_L0') == 'ln2_norm_L{i}'
    assert MG.entry_key('lut_tv_L11') == 'lut_tv_L{i}'
    assert MG.entry_key('lut_tv') == 'lut_tv'
    for k in ('ln2_norm_Lx', 'ln2_norm_L', 'xln2_norm_L0', 'ln2_norm_L0_extra'):
        assert MG.entry_key(k) is None, k
    assert MG.is_documented('_step') and MG.is_documented('system/gpu.0.gpu')
    assert MG.undocumented(['val_bpb', 'brand/new', 'lm_beta_L3']) == ['brand/new']


def test_every_entry_is_complete():
    for k, v in MG.METRICS.items():
        assert v['section'] in MG.SECTIONS, k
        assert v['unit'], k
        assert len(v['desc']) > 40, k
        assert 10 < len(v['short']) <= 240 and '\n' not in v['short'], k      # one legend-sized sentence
        assert len(v['short']) < len(v['desc']) or len(v['desc']) < 120, k
    assert set(MG.GOTCHAS) <= set(MG.CONFIG_NOTES)
    assert MG.stale(set(_metrics_csv_headers()) | set(WT.TRAIN_KEYS) | set(WT.EVAL_KEYS) | set(WT.SUMMARY_KEYS)
                    | {'lm_g_L0', 'lm_beta_L0', 'lm_gamma_L0'}) == []


def test_hash_is_stable_and_content_sensitive(monkeypatch):
    h = MG.glossary_hash()
    assert h == MG.glossary_hash() and len(h) == 12
    monkeypatch.setitem(MG.METRICS['val_bpb'], 'short', 'changed')
    assert MG.glossary_hash() != h


def test_panel_markdown_lists_every_logged_key_grouped():
    md = MG.panel_markdown('tools/metric_glossary.py')
    assert md.startswith(f'### {MG.PANEL_TITLE}')
    assert md == MG.panel_markdown('tools/metric_glossary.py')                 # content-only: verify is exact across commits
    for k, v in MG.METRICS.items():
        if v['section'] in MG.PANEL_SECTIONS:
            assert f'| `{k}` |' in md, k
    assert '| `step` |' not in md                                          # metrics.csv-only, not a wandb key
    titles = [md.index(f'**{t}**') for t in MG.PANEL_SECTIONS.values()]
    assert titles == sorted(titles)
    assert md.index('**Config gotchas**') > titles[-1]
    for k in MG.GOTCHAS:
        assert f'- `{k}`:' in md
    for line in md.splitlines():                                          # escaping never breaks the table:
        if line.startswith('| `'):
            assert len(re.split(r'(?<!\\)\|', line)) == 5, line               # '' key unit sentence '': no stray pipes
    assert len(md) < 12000


def test_table_rows_still_cover_everything():
    rows = MG.table_rows()
    assert len(rows) == len(MG.METRICS) + len(MG.CONFIG_NOTES)
    assert all(len(r) == 3 for r in rows)


def _spec(extra_sections=()):
    return {'section': {'panelBankConfig': {
        'sections': [{'__id__': 'a1', 'name': 'Charts', 'isPanelsAuto': True, 'panels': []},
                     {'__id__': 'b2', 'name': 'train', 'isPanelsAuto': True, 'panels': []}] + list(extra_sections)
        + [{'__id__': 'h3', 'name': 'Hidden Panels', 'isPanelsAuto': False, 'panels': []}],
        'panelConfigOverrides': {'val_bpb': {'config': {'metrics': ['val_bpb'], 'legendTemplate': 'x'}}},
        'settings': {'searchQuery': ''}}, 'runSets': [{'id': 'r'}]}}


def test_apply_section_is_first_idempotent_and_leaves_the_rest_alone():
    before = _spec()
    sec = WG.glossary_section('# v1')
    once = WG.apply_section(before, sec)
    twice = WG.apply_section(once, WG.glossary_section('# v2'))
    for spec, text in ((once, '# v1'), (twice, '# v2')):
        secs = spec['section']['panelBankConfig']['sections']
        assert secs[0]['__id__'] == WG.SECTION_ID and secs[0]['isPanelsAuto'] is False and secs[0]['pinned'] is True
        assert [s['__id__'] for s in secs[1:]] == ['a1', 'b2', 'h3']          # others untouched, in order
        assert sum(s['name'] == MG.PANEL_TITLE for s in secs) == 1
        assert secs[0]['panels'][0]['viewType'] == 'Markdown Panel' and secs[0]['panels'][0]['config']['value'] == text
        assert spec['section']['panelBankConfig']['panelConfigOverrides'] == before['section']['panelBankConfig']['panelConfigOverrides']
        assert spec['section']['runSets'] == before['section']['runSets']
    assert before == _spec()                                                  # input not mutated


def test_apply_section_replaces_a_renamed_or_moved_copy():
    moved = copy.deepcopy(WG.glossary_section('# old'))
    moved['__id__'] = 'regenerated-by-the-ui'                                 # matched by name too
    spec = WG.apply_section(_spec([moved]), WG.glossary_section('# new'))
    names = [s['name'] for s in spec['section']['panelBankConfig']['sections']]
    assert names == [MG.PANEL_TITLE, 'Charts', 'train', 'Hidden Panels']


def test_section_state():
    md = '# current'
    assert WG.section_state(_spec(), md) == (False, 'the section is absent')
    ok, why = WG.section_state(WG.apply_section(_spec(), WG.glossary_section(md)), md)
    assert ok and why == 'present, first, current'
    assert not WG.section_state(WG.apply_section(_spec(), WG.glossary_section('# old')), md)[0]
    late = _spec([WG.glossary_section(md)])
    assert WG.section_state(late, md) == (False, 'the section is at position 2, not first')
    dup = WG.apply_section(_spec(), WG.glossary_section(md))
    dup['section']['panelBankConfig']['sections'].append(WG.glossary_section(md))
    assert WG.section_state(dup, md)[1] == '2 copies of the section'
    assert WG.section_state({}, md)[0] is False


class _FakeRun:
    def __init__(self):
        self._tags, self.summary, self.logged = ('a',), {}, []

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, v):
        self._tags = tuple(v)

    def log(self, row, step=None):
        self.logged.append((step, dict(row)))


class _BrokenTagsRun(_FakeRun):
    @property
    def tags(self):
        return ()

    @tags.setter
    def tags(self, v):
        raise RuntimeError('server said no')


def _drain(t):
    t._q.put(None)
    t._th.join(5)


def test_drift_check_flags_unknown_keys_once(capsys):
    run = _FakeRun()
    t = WT.Tracker(run, wandb=None, mode='offline')
    t.train_step(1, 3.0, 3.0, 1e-4, grad_norm=0.5)
    t.eval_step(500, 1.2, 3.0, {'ln2_norm_L0': 1.0, 'brand_new': 2.0})
    t.eval_step(1000, 1.1, 3.0, {'ln2_norm_L0': 1.0, 'brand_new': 2.0, 'other_new': 1.0})
    _drain(t)
    assert WT.UNDOC_TAG in run.tags and run.tags.count(WT.UNDOC_TAG) == 1
    assert run.summary[WT.UNDOC_SUMMARY] == 'brand_new, other_new'
    out = capsys.readouterr().out
    assert out.count('brand_new') == 1 and out.count('other_new') == 1
    assert len(run.logged) == 3 and run.logged[0][1]['train/grad_norm'] == 0.5


def test_drift_check_is_never_fatal():
    run = _BrokenTagsRun()
    t = WT.Tracker(run, wandb=None, mode='offline')
    t.eval_step(500, 1.2, 3.0, {'brand_new': 2.0})
    t.eval_step(1000, 1.1, 3.0, {})
    _drain(t)
    assert t.run is run and len(run.logged) == 2                  # still logging, not disabled


def test_documented_rows_raise_no_flag(capsys):
    run = _FakeRun()
    t = WT.Tracker(run, wandb=None, mode='offline')
    t.train_step(10, 3.0, 3.0, 1e-4)
    t.eval_step(500, 1.2, 3.0, {'lut_tv': 1.0, 'lut_tv_L0': 1.0}, _Model())
    _drain(t)
    assert WT.UNDOC_TAG not in run.tags and WT.UNDOC_SUMMARY not in run.summary
    assert 'glossary' not in capsys.readouterr().out
    assert 'train/grad_norm' not in run.logged[0][1]


def test_github_web_url():
    assert WT.github_web_url('git@github-spikybot:owner/repo.git') == 'https://github.com/owner/repo'
    assert WT.github_web_url('git@github.com:owner/repo') == 'https://github.com/owner/repo'
    assert WT.github_web_url('https://github.com/owner/repo.git') == 'https://github.com/owner/repo'
    assert WT.github_web_url('ssh://git@github.com/owner/repo.git') == 'https://github.com/owner/repo'
    assert WT.github_web_url('git@gitlab.com:owner/repo.git') is None
    assert WT.github_web_url('unknown') is None and WT.github_web_url(None) is None


def test_host_and_config_normalisation():
    assert WT.normalise_host('pasta-gpustar') == 'gpustar'
    assert WT.normalise_host('gpustar') == 'gpustar' and WT.normalise_host(None) is None
    cfg = {'eval_steps': 10, 'lr': 3e-4, '_arch_note': 'long', 'description': 'short'}
    c = WT.wandb_config(cfg, host='gpustar')
    assert c == {'eval_steps_legacy_ignored': 10, 'lr': 3e-4, 'host': 'gpustar'}


def test_description_text():
    assert WT.description_text({'description': 'Short.', '_arch_note': 'Long note.'}) == 'Short.'
    note = 'First sentence. Second one here. Third (short). Fourth. Fifth should not appear.'
    assert WT.description_text({'_arch_note': note}) == 'First sentence. Second one here. Third (short). Fourth.'
    assert WT.description_text({'_arch_note': 'A' * 3000}).endswith('...')
    assert WT.description_text({}) == ''


def test_run_notes_links_and_flags():
    sha = 'a' * 40
    info = dict(web='https://github.com/o/r', sha=sha, branch='research/x', dirty=True,
                rel='experiments/f/runs/exp_1', committed=False)
    cfg = {'_arch_note': 'What it tests. Why. How it differs. The fourth. The fifth is not in the summary. *star* x_y'}
    ws = WT.workspace_url('http://h/', 'e', 'p')
    assert ws == 'http://h/e/p/workspace' and WT.workspace_url('', 'e', 'p') is None
    md = WT.run_notes(cfg, exp_name='exp_1', info=info, host='gpustar', workspace_url=ws,
                      artifact_url='http://h/art', artifact_label='metric_glossary:glossary-abc')
    head, _, tail = md.partition('\n---\n')
    assert head.startswith('**exp\\_1**')
    assert f'(https://github.com/o/r/tree/{sha}/experiments/f/runs/exp_1)' in head
    assert '(https://github.com/o/r/tree/research/x/experiments/f/runs/exp_1)' in head
    assert 'dirty' in head and 'not committed at launch' in head
    assert '`experiments/f/runs/exp_1` on `gpustar`' in head
    assert '"About these metrics" at the top of the [project workspace](http://h/e/p/workspace)' in head
    assert '/reports/' not in md
    assert '[metric\\_glossary:glossary-abc](http://h/art)' in head
    assert 'fifth' not in head and 'The fifth' in tail
    assert '\\*star\\* x\\_y' in tail                              # _arch_note rendered literally


def test_run_notes_without_git():
    md = WT.run_notes({'description': 'D.'}, exp_name='e', info=dict(), host='h')
    assert md.startswith('**e**') and 'github' not in md and '---' not in md and 'Metric glossary' not in md


def test_git_launch_info_on_this_checkout():
    info = WT.git_launch_info(HERE)
    if info['sha'] is None:
        pytest.skip('not a git checkout')
    assert len(info['sha']) == 40
    assert info['rel'] == os.path.relpath(HERE, info['root']) and not info['rel'].startswith('..')
    assert info['committed'] is True
    outside = WT.git_launch_info('/tmp')
    assert outside['rel'] is None and outside['committed'] is None
