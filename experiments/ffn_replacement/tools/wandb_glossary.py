"""Publish and audit the metric glossary (tools/metric_glossary.py) on the self-hosted wandb.

    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py publish [--project Spiky] [--dry-run]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py audit   [--project Spiky]

publish  Upserts ONE report titled "Metric glossary": found by title and updated in place, created only if absent
         (with several, the first is updated and the others are reported, never deleted). Then sets the project
         description to a short README linking it. Raw GraphQL upsertView / upsertModel, because the
         wandb-workspaces reports API is not installed here. --dry-run prints the report markdown and README only.
audit    READ-ONLY. Undocumented keys -- logged keys with no glossary entry -- from the project's runs (their summary
         holds every logged key's last value) and from runs_corrected/*/metrics.csv headers; and stale entries,
         which match nothing seen on the server, in those headers or among the tracker's own keys.
         Exit status 1 when anything is undocumented.

No secrets: auth from ~/.netrc; server and entity from the environment.
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metric_glossary as MG                                                     # noqa: E402
from wandb_tracking import (PROJECT, REPORT_TITLE, EVAL_KEYS, SUMMARY_KEYS, TRAIN_KEYS, _git, find_glossary_reports,  # noqa: E402
                            gql, report_url)

SOURCE = 'experiments/ffn_replacement/tools/metric_glossary.py'
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')

_UPSERT_VIEW = 'mutation($i: UpsertViewInput!){ upsertView(input: $i){ inserted view { id name displayName type } } }'
_UPSERT_MODEL = 'mutation($i: UpsertModelInput!){ upsertModel(input: $i){ project { id name description } } }'


def _env():
    base, entity = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/'), os.environ.get('WANDB_ENTITY')
    if not base or not entity:
        sys.exit('WANDB_BASE_URL and WANDB_ENTITY must be set (never publish to wandb.ai)')
    return base, entity


def _arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def readme(report):
    return (f'# Spiky\n\nOne project for all spiky experiments; a run\'s `group` is its experiment family '
            f'(`experiments/<family>/`).\n\n'
            f'- **[{REPORT_TITLE}]({report})** -- what every logged key measures, and the config keys that are not what '
            f'they look like (glossary `{MG.glossary_hash()}`, generated from `{SOURCE}`).\n'
            f'- Each run\'s **notes** (Overview tab) say what the run tests and link to its code on GitHub at the launch '
            f'commit, its run folder and this glossary.\n')


def publish(argv):
    project, dry = _arg(argv, '--project', PROJECT), '--dry-run' in argv
    commit = _git(['rev-parse', '--short', 'HEAD'], HERE)
    body = MG.report_markdown(SOURCE, None if commit == 'unknown' else commit)
    spec = {'version': 5, 'panelSettings': {}, 'width': 'readable', 'authors': [], 'discussionThreads': [], 'ref': {},
            'blocks': [{'type': 'markdown-block', 'content': body, 'children': [{'text': ''}]}]}
    description = f'What every logged key measures. Glossary {MG.glossary_hash()}.'
    if dry:
        print(body)
        print(readme('<report url>'))
        return 0
    base, entity = _env()
    import wandb
    api = wandb.Api(timeout=60)
    viewer = (gql(api, '{ viewer { username name } }', {}) or {}).get('viewer') or {}
    if viewer.get('username'):                       # the UI records the author in the spec; do the same
        spec['authors'] = [{'name': viewer.get('name') or '', 'username': viewer['username']}]
    found = find_glossary_reports(api, entity, project)
    # type "runs" = a PUBLISHED report, listed in the project's Reports tab. "runs/draft" is an unpublished draft
    # (what the UI's "Create report" makes first) and a draft with parentId is an edit-in-progress of a published
    # report; neither is what we want. Set it explicitly on create AND update.
    view = {'displayName': REPORT_TITLE, 'description': description, 'spec': json.dumps(spec), 'type': 'runs'}
    if found:
        view['id'] = found[0]['id']
    else:
        view.update(entityName=entity, projectName=project, name='metric-glossary')
    res = gql(api, _UPSERT_VIEW, {'i': view})['upsertView']
    v = res['view']
    url = report_url(base, entity, project, v)
    print(f'report {"created" if res["inserted"] else "updated"}: {url} (glossary {MG.glossary_hash()}, commit {commit})')
    if len(found) > 1:
        print(f'WARNING: {len(found)} reports titled {REPORT_TITLE!r}; updated {found[0]["id"]}, left '
              f'{[f["id"] for f in found[1:]]} untouched -- delete the extras in the UI')
    p = gql(api, _UPSERT_MODEL, {'i': {'entityName': entity, 'name': project, 'description': readme(url)}})['upsertModel']
    print(f'project {p["project"]["name"]} description set ({len(p["project"]["description"])} chars)')
    ok, why = listed_in_reports_tab(api, entity, project, v['id'], title=REPORT_TITLE)
    if not ok:
        bar = '!' * 100
        print(f'{bar}\nWARNING: report {v["id"]} is NOT listed in the {project} Reports tab: {why}.\n'
              f'It still opens at {url}, but nobody will find it by navigation. Fix before relying on it.\n{bar}')
        return 2
    print(f'verified: listed in the {project} Reports tab ({why})')
    return 0


# The project Reports tab's own query (operation ReportTable in the web UI): published reports are
# allViews(viewType: "runs"), drafts allViews(viewType: "runs/draft"); a published report's pending edits are its children.
_REPORT_TABLE = ('query($e: String, $p: String!, $t: String){ project(name: $p, entityName: $e){ '
                 'reportDrafts: allViews(viewType: "runs/draft", first: 1000, displayNameContains: $t){ edges { node { id type parentId } } } '
                 'reports: allViews(viewType: "runs", first: 100, displayNameContains: $t){ edges { node { id type parentId displayName } } } } }')


def listed_in_reports_tab(api, entity, project, view_id, title=None):
    """(ok, reason): is `view_id` among the PUBLISHED reports the project's Reports tab lists? `title` narrows the
    listing server-side (displayNameContains); None reads it unfiltered (first 100 published, 1000 drafts)."""
    data = gql(api, _REPORT_TABLE, {'e': entity, 'p': project, 't': title}) or {}
    proj = data.get('project') or {}
    pub = {e['node']['id']: e['node'] for e in (proj.get('reports') or {}).get('edges', [])}
    drafts = {e['node']['id']: e['node'] for e in (proj.get('reportDrafts') or {}).get('edges', [])}
    if view_id in pub and pub[view_id]['type'] == 'runs' and not pub[view_id]['parentId']:
        extra = [d for d in drafts.values() if not d['parentId']]
        where = f'titled {title!r}' if title else 'in the project'
        return True, 'published' + (f'; note: {len(extra)} unpublished draft(s) {where} also exist' if extra else '')
    if view_id in drafts:
        return False, f'it is only a draft (type {drafts[view_id]["type"]}, parentId {drafts[view_id]["parentId"]})'
    return False, 'absent from both the published and the draft listing'


def audit(argv):
    project = _arg(argv, '--project', PROJECT)
    base, entity = _env()
    import wandb
    api = wandb.Api(timeout=60)
    server = {}
    for r in api.runs(f'{entity}/{project}', per_page=100):
        for k in r.summary_metrics.keys():
            server.setdefault(k, r.name)
    csv_keys = {}
    for p in sorted(glob.glob(os.path.join(RC, '*', 'metrics.csv'))):
        with open(p, newline='') as f:
            for h in next(csv.reader(f), []):
                csv_keys.setdefault(h, os.path.basename(os.path.dirname(p)))
    tracker = set(TRAIN_KEYS) | set(EVAL_KEYS) | set(SUMMARY_KEYS)
    bad = 0
    for label, keys in (('server runs', server), ('runs_corrected metrics.csv', csv_keys),
                        ('tracker keys', {k: 'wandb_tracking.py' for k in tracker})):
        und = MG.undocumented(keys)
        bad += len(und)
        print(f'{label}: {len(keys)} keys, {len(und)} undocumented'
              + ''.join(f'\n   {k}  (e.g. {keys[k]})' for k in und))
    stale_all = MG.stale(set(server) | set(csv_keys) | tracker)
    print(f'stale entries (match nothing seen anywhere): {stale_all or "none"}')
    not_in_data = [k for k in MG.stale(set(server) | set(csv_keys)) if k not in stale_all]
    print(f'documented, emitted by the tracker, not yet in any run\'s data: {not_in_data or "none"}')
    print(f'glossary {MG.glossary_hash()}: {len(MG.METRICS)} entries, {len(MG.CONFIG_NOTES)} config notes')
    return 1 if bad else 0


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd not in ('publish', 'audit'):
        sys.exit(__doc__)
    sys.exit((publish if cmd == 'publish' else audit)(sys.argv[2:]))
