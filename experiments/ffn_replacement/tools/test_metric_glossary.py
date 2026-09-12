"""CPU-only, no server: the Spiky metric glossary (metric_glossary.py) covers everything we log -- every metrics.csv
header in the repo and the keys wandb_tracking.py emits -- and its entries, hash, panel text and artifact rows are
well-formed. The generic tracker / workspace / backfill tests live with the package in
src/spiky/util/wandb_integration/tests.

    python -m pytest experiments/ffn_replacement/tools/test_metric_glossary.py -q
"""
import csv
import glob
import os
import re

import metric_glossary as MG
import wandb_tracking as WT

HERE = os.path.dirname(os.path.abspath(__file__))
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
    assert MG.panel_markdown() == MG.panel_markdown(MG.SOURCE)                  # the default the package's publish uses
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
