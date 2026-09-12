"""audit scoped to a W&B group: the filter reaches the API, undocumented and stale are computed over that group only, an
unscoped audit says so, the tracker's drift key is never stale; publish --no-readme leaves the project description
alone. CPU only, no server (a fake wandb.Api)."""
import json
import sys
import types

import pytest

from spiky.util.wandb_integration import glossary as G
from spiky.util.wandb_integration import workspace as W


class _Run:
    def __init__(self, name, group, keys):
        self.name, self.group, self.summary_metrics = name, group, {k: 1.0 for k in keys}


RUNS = [_Run('ffn_1', 'ffn', ['val_bpb', 'ln2_norm_L0', 'lut_tv']),
        _Run('smoke_1', 'smoke', ['val_bpb', 'train/loss'])]


class _Api:
    calls = []

    def __init__(self, timeout=None):
        pass

    def runs(self, path, filters=None, per_page=50):
        _Api.calls.append((path, filters))
        return [r for r in RUNS if not filters or r.group == filters.get('group')]


@pytest.fixture
def fake_api(monkeypatch):
    _Api.calls = []
    monkeypatch.setitem(sys.modules, 'wandb', types.SimpleNamespace(Api=_Api))
    monkeypatch.setenv('WANDB_BASE_URL', 'http://wandb.invalid')
    monkeypatch.setenv('WANDB_ENTITY', 'ent')
    return _Api


DRIFT = {'glossary/undocumented': dict(unit='-', desc='Logged keys with no entry (drift check).')}
FFN = G.DictGlossary({'val_bpb': dict(unit='bits/byte', desc='The fixed 100 x 48 eval.'),
                      'ln2_norm_L{i}': dict(unit='L2', desc='ln2 gain norm of block i.'),
                      'lut_tv': dict(unit='tv', desc='Mean cell TV.'), **DRIFT})
SMOKE = G.DictGlossary({'val_bpb': dict(unit='bits/byte', desc='The batch-coupled 10 x 48 eval.'),
                        'train/loss': dict(unit='nats', desc='Step loss.'), **DRIFT})


def test_a_scoped_audit_passes_the_group_filter_and_reads_only_that_group(fake_api, capsys):
    assert W.audit(SMOKE, 'P', group='smoke') == 0
    out = capsys.readouterr().out
    assert fake_api.calls == [('ent/P', {'group': 'smoke'})]
    assert "scope: P, group 'smoke' (1 runs)" in out
    assert "server runs in group 'smoke': 2 keys, 0 undocumented" in out
    assert 'stale entries (match nothing seen in scope): none' in out


def test_each_glossary_is_clean_in_its_own_group(fake_api, capsys):
    assert W.audit(FFN, 'P', group='ffn') == 0 and W.audit(SMOKE, 'P', group='smoke') == 0
    out = capsys.readouterr().out
    assert out.count('0 undocumented') >= 2 and out.count('stale entries (match nothing seen in scope): none') == 2


def test_an_unscoped_audit_says_so_and_flags_other_groups_keys(fake_api, capsys):
    assert W.audit(SMOKE, 'P') == 1
    out = capsys.readouterr().out
    assert fake_api.calls == [('ent/P', None)]
    assert 'ALL 2 runs -- UNSCOPED' in out and 'server runs (all groups): 4 keys, 2 undocumented' in out
    assert 'ln2_norm_L0' in out and 'lut_tv' in out


def test_stale_is_computed_over_the_scope_only(fake_api, capsys):
    W.audit(FFN, 'P', group='smoke')
    assert "stale entries (match nothing seen in scope): ['ln2_norm_L{i}', 'lut_tv']" in capsys.readouterr().out


def test_the_drift_key_counts_as_always_emitted(fake_api, capsys):
    W.audit(SMOKE, 'P', group='smoke')
    out = capsys.readouterr().out
    stale_line = [line for line in out.splitlines() if line.startswith('stale entries')][0]
    assert 'glossary/undocumented' not in stale_line
    assert "not yet in any run's data: ['glossary/undocumented']" in out


def test_main_group_flag_beats_the_wrapper_default(fake_api, monkeypatch):
    seen = {}
    monkeypatch.setattr(W, 'audit', lambda glossary, project, **kw: seen.update(kw) or 0)
    assert W.main(['audit', '--project', 'P'], glossary=SMOKE, audit_group='smoke') == 0 and seen['group'] == 'smoke'
    assert W.main(['audit', '--project', 'P', '--group', 'ffn'], glossary=SMOKE, audit_group='smoke') == 0
    assert seen['group'] == 'ffn'
    assert W.main(['audit', '--project', 'P'], glossary=SMOKE) == 0 and seen['group'] is None


def _fake_publish_server(monkeypatch):
    spec = {'section': {'panelBankConfig': {'sections': [{'__id__': 'a1', 'name': 'Charts', 'isPanelsAuto': True,
                                                          'panels': []}]}}}
    view = {'id': 'V1', 'name': 'nw-pdescribed-v', 'updatedAt': '2026-09-13T00:00:00Z', 'updatedBy': {'username': 'u'},
            'specObject': spec}
    store, queries = {'spec': spec}, []

    def gql(api, query, variables, timeout=None):
        queries.append(query)
        if 'upsertView' in query:
            store['spec'] = json.loads(variables['i']['spec'])
            return {'upsertView': {'inserted': False, 'view': {'id': 'V1', 'name': view['name']}}}
        if 'upsertModel' in query:
            return {'upsertModel': {'project': {'id': 'p', 'name': 'P', 'description': variables['i']['description']}}}
        return {'view': dict(view, specObject=store['spec'])}

    monkeypatch.setattr(W, 'gql', gql)
    monkeypatch.setattr(W, '_connect', lambda user=None: (object(), 'http://h', 'ent', 'u'))
    monkeypatch.setattr(W, '_target', lambda shared, api, base, entity, project, user:
                        (view, 'http://h/ent/P?nw=pdescribed', 'shared saved view'))
    return queries


def test_publish_no_readme_leaves_the_project_description_alone(monkeypatch, capsys):
    queries = _fake_publish_server(monkeypatch)
    g = G.DictGlossary({'k': dict(unit='u', desc='d')})
    assert W.publish(g, 'P', shared_title='P described', readme=False) == 0
    assert not any('upsertModel' in q for q in queries) and any('upsertView' in q for q in queries)
    assert 'project P description left unchanged (--no-readme)' in capsys.readouterr().out
    assert W.publish(g, 'P', shared_title='P described') == 0
    assert any('upsertModel' in q for q in queries)                          # the default still sets it


def test_main_no_readme_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(W, 'publish', lambda glossary, project, **kw: seen.update(kw) or 0)
    g = G.DictGlossary({'k': dict(unit='u', desc='d')})
    assert W.main(['publish', '--project', 'P', '--shared', 'T', '--no-readme'], glossary=g) == 0 and seen['readme'] is False
    assert W.main(['publish', '--project', 'P', '--shared', 'T'], glossary=g) == 0 and seen['readme'] is True
