"""CPU-only, no server: the Spiky metric legend (metric_glossary.py) covers everything we log -- every metrics.csv
header in the repo and the keys wandb_tracking.py emits -- its entries are well-formed, and legend_markdown(), which the
tracker writes into every run's notes, lists every logged key. The generic tracker / backfill tests live with the
package in src/spiky/util/wandb_integration/tests.

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
    assert not missing, f'metrics.csv columns with no legend entry (add them to metric_glossary.py): {missing}'


class _Mod:
    confidence_form = 'learned_margin'

    def learned_confidence_values(self):
        return {'g': 0.0, 'beta': 2.0, 'gamma': 1.0}


class _Model:
    def modules(self):
        return [object(), _Mod(), _Mod()]


def _tracker_keys():
    return (list(WT.TRAIN_KEYS) + list(WT.EVAL_KEYS) + list(WT.SUMMARY_KEYS)
            + list(WT.learned_confidence_by_layer(_Model())))


def test_tracker_fixed_keys_are_documented():
    keys = _tracker_keys()
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
        assert set(v) == {'section', 'unit', 'desc'}, k
        assert v['section'] in MG.SECTIONS, k
        assert v['unit'] and len(v['unit']) <= 12, k                 # short: printed in brackets after the description
        assert len(v['desc']) > 40, k
    assert set(MG.LEGEND_SECTIONS) <= set(MG.SECTIONS)


def test_no_entry_is_stale():
    seen = set(_metrics_csv_headers()) | set(_tracker_keys())
    hit = {MG.entry_key(k) for k in seen}
    assert sorted(k for k in MG.METRICS if k not in hit) == []


def test_legend_lists_every_logged_key_grouped():
    md = MG.legend_markdown('tools/metric_glossary.py')
    assert md.startswith(f'### {MG.LEGEND_TITLE}\n')
    assert md == MG.legend_markdown('tools/metric_glossary.py')                 # content only: identical across commits
    assert MG.legend_markdown() == MG.legend_markdown(MG.SOURCE)                 # what the tracker writes
    for k, v in MG.METRICS.items():
        if v['section'] in MG.LEGEND_SECTIONS:
            assert f'| `{k}` |' in md, k
    assert '| `step` |' not in md                                           # metrics.csv-only, not a wandb key
    titles = [md.index(f'**{t}**') for t in MG.LEGEND_SECTIONS.values()]
    assert titles == sorted(titles)
    assert md.index(f'**{MG.CONFIG_NOTES_TITLE}**') > titles[-1]
    for k in MG.CONFIG_NOTES:
        assert f'- `{k}`:' in md
    for line in md.splitlines():                                           # escaping never breaks the table:
        if line.startswith('| `'):
            assert len(re.split(r'(?<!\\)\|', line)) == 4, line                # '' key description [unit] '': no stray pipes
    for gone in ('About these metrics', 'Artifacts tab', 'wandb_glossary', 'publish', '/workspace'):
        assert gone not in md, gone


def test_the_package_takes_this_module_as_the_run_legend():
    from spiky.util.wandb_integration import glossary as G
    from spiky.util.wandb_integration import tracker as T
    assert G.missing(MG) == []
    notes = T.run_notes({'description': 'D.'}, exp_name='e', info={}, host='h', legend=MG.legend_markdown())
    assert MG.legend_markdown().rstrip('\n') in notes and T.NO_LEGEND_LINE not in notes
