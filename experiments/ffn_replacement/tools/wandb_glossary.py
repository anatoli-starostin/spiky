"""Publish, verify and audit the metric glossary (tools/metric_glossary.py) on the self-hosted wandb.

    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py publish [--project Spiky] [--shared TITLE | --user NAME] [--dry-run] [--if-idle MIN]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py verify  [--project Spiky] [--shared TITLE | --user NAME]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py audit   [--project Spiky]

publish  Writes an "About these metrics" Markdown Panel (metric_glossary.panel_markdown: one short sentence per key +
         config gotchas) into its own pinned section at the TOP of a workspace view. Idempotent: the section and panel
         carry fixed ids and are replaced in place, never duplicated. Nothing else in the spec is touched -- in
         particular no panelConfigOverrides (an override REPLACES an auto panel's whole config). Afterwards the spec
         is re-read: a loud warning and exit 2 if the section is not first or not identical.
         Target view:
         --shared TITLE  (recommended) a SHARED saved view named TITLE (view name nw-<id>-v, id = TITLE lower-cased,
                         letters and digits only), opened at <server>/<entity>/<project>?nw=<id>. Created from the
                         user's current personal workspace layout; later runs only replace our section, keeping any
                         edits saved into the view. The web client does NOT auto-save saved views ("Changes are not
                         auto-saved ... Save view"), so an open browser tab cannot silently remove the panel; only an
                         explicit "Save view" from a tab loaded before the write can.
         default         the user's personal workspace (nw-nwuser<username>-w), what they see on opening the project.
                         CLOBBER RISK: the client auto-saves the whole personal spec, so a tab opened BEFORE the
                         write removes the section on its next change. publish prints when the workspace was last
                         saved and by whom; --if-idle MIN refuses to write if that was less than MIN minutes ago.
         It also points the project description at the panel.
verify   READ-ONLY. Exit 2 unless the section is present, first, and matches the current glossary.
audit    READ-ONLY. Undocumented keys -- logged keys with no glossary entry -- from the project's runs (their summary
         holds every logged key's last value) and from runs_corrected/*/metrics.csv headers; and stale entries,
         which match nothing seen on the server, in those headers or among the tracker's own keys.
         Exit status 1 when anything is undocumented.

No secrets: auth from ~/.netrc; server and entity from the environment.
"""
import copy
import csv
import datetime
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metric_glossary as MG                                                     # noqa: E402
from wandb_tracking import EVAL_KEYS, PROJECT, SUMMARY_KEYS, TRAIN_KEYS, gql       # noqa: E402

SOURCE = 'experiments/ffn_replacement/tools/metric_glossary.py'
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')
SECTION_ID, PANEL_ID = 'metric-glossary-section', 'metric-glossary-panel'
# One full-width column; tall enough to show the four tables without scrolling the page (the panel scrolls inside).
FLOW_CONFIG = {'columnsPerPage': 1, 'rowsPerPage': 1, 'boxHeight': 560}

_VIEWS = ('query($e: String!, $p: String!){ project(entityName: $e, name: $p){ allViews(viewType: "project-view", first: 100){ '
          'edges { node { id name displayName updatedAt user { username } updatedBy { username } } } } } }')
_VIEW = 'query($id: ID!){ view(id: $id){ id name displayName type updatedAt updatedBy { username } specObject } }'
_UPSERT_VIEW = 'mutation($i: UpsertViewInput!){ upsertView(input: $i){ inserted view { id name updatedAt } } }'
_UPSERT_MODEL = 'mutation($i: UpsertModelInput!){ upsertModel(input: $i){ project { id name description } } }'


# ---- pure helpers (unit-tested, no server) ------------------------------------------------------------------------
def glossary_section(markdown):
    """The explicit, non-auto section holding the one Markdown Panel. pinned=True: the web client keeps pinned
    sections ahead of unpinned ones when sections are (un)pinned, and shows it as pinned."""
    return {'__id__': SECTION_ID, 'name': MG.PANEL_TITLE, 'isOpen': True, 'isPanelsAuto': False, 'sorted': 0,
            'pinned': True, 'flowConfig': dict(FLOW_CONFIG),
            'panels': [{'__id__': PANEL_ID, 'viewType': 'Markdown Panel', 'config': {'value': markdown}}]}


def _is_ours(section):
    return section.get('__id__') == SECTION_ID or section.get('name') == MG.PANEL_TITLE


def apply_section(spec, section):
    """A copy of the workspace spec with our section first and every other section untouched and in order.
    Any previous copy of our section (matched by id or by name) is removed, so re-running never duplicates."""
    out = copy.deepcopy(spec)
    pbc = out['section']['panelBankConfig']
    pbc['sections'] = [section] + [s for s in pbc.get('sections', []) if not _is_ours(s)]
    return out


def section_state(spec, markdown):
    """(ok, reason) for our section in a workspace spec."""
    secs = ((spec or {}).get('section') or {}).get('panelBankConfig', {}).get('sections', [])
    ours = [i for i, s in enumerate(secs) if _is_ours(s)]
    if not ours:
        return False, 'the section is absent'
    if len(ours) > 1:
        return False, f'{len(ours)} copies of the section'
    s = secs[ours[0]]
    if ours[0] != 0:
        return False, f'the section is at position {ours[0]}, not first'
    values = [(p.get('config') or {}).get('value') for p in s.get('panels', []) if p.get('viewType') == 'Markdown Panel']
    if not values:
        return False, 'the section has no Markdown Panel'
    if values[0] != markdown:
        return False, 'the panel text differs from the current glossary (stale or edited)'
    return True, 'present, first, current'


def shared_nw_id(title):
    """The named-workspace id of a shared saved view: the title lower-cased, letters and digits only."""
    nwid = re.sub(r'[^a-z0-9]', '', (title or '').lower())
    if not nwid:
        raise ValueError(f'cannot derive a view id from {title!r}')
    return nwid[:40]


def shared_view_name(title):
    return f'nw-{shared_nw_id(title)}-v'


# ---- server helpers ------------------------------------------------------------------------------------------------
def _env():
    base, entity = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/'), os.environ.get('WANDB_ENTITY')
    if not base or not entity:
        sys.exit('WANDB_BASE_URL and WANDB_ENTITY must be set (never publish to wandb.ai)')
    return base, entity


def _arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _views(api, entity, project):
    edges = ((gql(api, _VIEWS, {'e': entity, 'p': project}) or {}).get('project') or {}).get('allViews', {}).get('edges', [])
    return [e['node'] for e in edges]


def personal_workspace(api, entity, project, username):
    """The user's personal workspace view (created by the web UI on their first visit), or None."""
    nodes = _views(api, entity, project)
    exact = [n for n in nodes if n['name'] == f'nw-nwuser{username}-w']
    loose = [n for n in nodes if (n.get('user') or {}).get('username') == username
             and n['name'].startswith('nw-nwuser') and n['name'].endswith('-w')]
    return (exact or loose or [None])[0]


def _age_minutes(ts):
    t = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
    return (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds() / 60


def _markdown():
    return MG.panel_markdown(SOURCE)


def _connect(argv):
    project = _arg(argv, '--project', PROJECT)
    base, entity = _env()
    import wandb
    api = wandb.Api(timeout=60)
    user = _arg(argv, '--user', None) or ((gql(api, '{ viewer { username } }', {}) or {}).get('viewer') or {}).get('username')
    return api, base, entity, project, user


def _personal_or_exit(api, base, entity, project, user):
    ws = personal_workspace(api, entity, project, user)
    if ws is None:
        print(f'no personal workspace for {user!r} in {entity}/{project}: open {base}/{entity}/{project}/workspace '
              f'once in the browser as that user (wandb creates it on the first visit), then re-run')
        sys.exit(3)
    return gql(api, _VIEW, {'id': ws['id']})['view']


def _target(argv, api, base, entity, project, user):
    """(view or None, url, description) of the view publish/verify works on."""
    title = _arg(argv, '--shared', None)
    if title is None:
        view = _personal_or_exit(api, base, entity, project, user)
        return view, f'{base}/{entity}/{project}/workspace', f"{user}'s personal workspace"
    name = shared_view_name(title)
    hit = [n for n in _views(api, entity, project) if n['name'] == name]
    view = gql(api, _VIEW, {'id': hit[0]['id']})['view'] if hit else None
    return view, f'{base}/{entity}/{project}?nw={shared_nw_id(title)}', f'shared saved view {title!r} ({name})'


def publish(argv):
    md = _markdown()
    if '--dry-run' in argv:
        print(md)
        print(f'({len(md)} chars; section {MG.PANEL_TITLE!r}, id {SECTION_ID}, flowConfig {FLOW_CONFIG})')
        return 0
    api, base, entity, project, user = _connect(argv)
    title = _arg(argv, '--shared', None)
    view, url, what = _target(argv, api, base, entity, project, user)
    if title is not None:
        if view is None:                                  # create from the user's current layout
            base_spec = _personal_or_exit(api, base, entity, project, user)['specObject']
            inp = {'entityName': entity, 'projectName': project, 'name': shared_view_name(title), 'displayName': title,
                   'type': 'project-view', 'description': f'{project} workspace with the {MG.PANEL_TITLE} panel pinned on top'}
            print(f'{what}: creating it from {user}\'s personal workspace layout')
        else:
            base_spec = view['specObject']
            inp = {'id': view['id']}
            print(f'{what} ({view["id"]}) last saved {_age_minutes(view["updatedAt"]):.1f} min ago by '
                  f'{(view.get("updatedBy") or {}).get("username")}; saved views are not auto-saved by the web client')
    else:
        age = _age_minutes(view['updatedAt'])
        by = (view.get('updatedBy') or {}).get('username')
        print(f'{what} {view["name"]} ({view["id"]}) last saved {age:.1f} min ago by {by} ({view["updatedAt"]})')
        idle = _arg(argv, '--if-idle', None)
        if idle is not None and age < float(idle):
            print(f'NOT WRITTEN: saved less than {idle} min ago -- a browser tab may be open and would overwrite the '
                  f'change on its next save. Close the tabs (or wait) and re-run.')
            return 4
        if age < 15:
            print('!' * 100 + f'\nWARNING: the workspace was saved {age:.1f} min ago -- a browser tab may be open. If one is,'
                  f' its next save will silently remove this section; run `verify` later, or use --shared.\n' + '!' * 100)
        base_spec = view['specObject']
        inp = {'id': view['id']}
    new = apply_section(base_spec, glossary_section(md))
    others_before = [s.get('__id__') for s in base_spec['section']['panelBankConfig']['sections'] if not _is_ours(s)]
    had = any(_is_ours(s) for s in base_spec['section']['panelBankConfig']['sections']) and view is not None
    inp['spec'] = json.dumps(new)
    res = gql(api, _UPSERT_VIEW, {'i': inp})['upsertView']
    after = gql(api, _VIEW, {'id': res['view']['id']})['view']
    ok, why = section_state(after['specObject'], md)
    others_after = [s.get('__id__') for s in after['specObject']['section']['panelBankConfig']['sections'] if not _is_ours(s)]
    same_overrides = (base_spec['section']['panelBankConfig'].get('panelConfigOverrides')
                      == after['specObject']['section']['panelBankConfig'].get('panelConfigOverrides'))
    print(f'section {"replaced" if had else "created"} in {what}: {len(md)} chars, glossary {MG.glossary_hash()}; '
          f'other sections unchanged and in order: {others_before == others_after}; '
          f'panelConfigOverrides unchanged: {same_overrides}')
    p = gql(api, _UPSERT_MODEL, {'i': {'entityName': entity, 'name': project, 'description': readme(url, title)}})['upsertModel']
    print(f'project {p["project"]["name"]} description set ({len(p["project"]["description"])} chars)')
    if not ok or others_before != others_after:
        bar = '!' * 100
        print(f'{bar}\nWARNING: after the write the glossary section is NOT right in {what}: {why}'
              + ('' if others_before == others_after else '; the other sections changed') + f'.\n{bar}')
        return 2
    print(f'verified: {why} in {what} -> {url}')
    return 0


def verify(argv):
    api, base, entity, project, user = _connect(argv)
    view, url, what = _target(argv, api, base, entity, project, user)
    if view is None:
        print(f'{what} does not exist -> run publish')
        return 2
    ok, why = section_state(view['specObject'], _markdown())
    print(f'{what} last saved {_age_minutes(view["updatedAt"]):.1f} min ago by '
          f'{(view.get("updatedBy") or {}).get("username")}: {why}  ({url})')
    if not ok:
        print('!' * 100 + f'\nWARNING: the glossary panel is not in place in {what} -- re-run publish'
              + ('' if _arg(argv, '--shared', None) else ' (with the browser tabs closed, or use --shared)') + '.\n' + '!' * 100)
    return 0 if ok else 2


def readme(url, shared_title=None):
    where = (f'the saved view "{shared_title}" -- the project workspace with the panel pinned on top'
             if shared_title else 'the panel at the top of the workspace')
    return (f"# Spiky\n\nOne project for all spiky experiments; a run's `group` is its experiment family "
            f'(`experiments/<family>/`).\n\n'
            f'- **[{MG.PANEL_TITLE}]({url})** -- {where}: what every logged key measures, plus config gotchas '
            f'(glossary `{MG.glossary_hash()}`, generated from `{SOURCE}`).\n'
            f"- Each run's **notes** (Overview tab) say what the run tests and link to its code on GitHub at the launch "
            f'commit and its run folder.\n')


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
    print(f'glossary {MG.glossary_hash()}: {len(MG.METRICS)} entries, {len(MG.CONFIG_NOTES)} config notes, '
          f'{len(MG.GOTCHAS)} gotchas')
    return 1 if bad else 0


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd not in ('publish', 'verify', 'audit'):
        sys.exit(__doc__)
    sys.exit({'publish': publish, 'verify': verify, 'audit': audit}[cmd](sys.argv[2:]))
