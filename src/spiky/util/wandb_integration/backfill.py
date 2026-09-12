"""Backfill finished runs into wandb from their recorded artefacts (metrics.csv, config.json, optional summary.json),
and rewrite the notes of runs already on the server -- e.g. to give existing runs their metric legend. Backfilled runs
are clearly marked: tag "backfilled", config backfilled=True, notes say so.

A library: the project supplies where its runs live and everything run-specific (branch, host, tags, derived config,
which column marks a complete row, what "complete" means, description overrides, the glossary).

    from spiky.util.wandb_integration import backfill as B
    B.backfill_run(run_dir, project='P', group='family', branch='research/x', host='gpustar', tags=['line:a'],
                   config_extra={'total_params': 123}, summary_keys=('final_val_loss',), require_col='val_loss',
                   is_complete=lambda rows, cfg: (rows[-1][0] == cfg['n_steps'], 'last step'), mode='offline',
                   glossary=GLOSSARY)
    B.update_notes(api, entity, 'P', {'exp_1': '/path/to/exp_1'}, glossary=GLOSSARY, descriptions={'exp_1': '...'},
                   dry_run=True)

backfill_run: same organisation as the live tracker (tracker.py / claude/wandb.md): name = id = cfg exp_name (else
the folder name), tags + config with branch / commit / host, and the same notes blob (description, links, the metric
legend). commit = the commit that recorded the run's artefacts (git log on its metrics.csv). One wandb row per
metrics.csv row that has `require_col` (every row if None), at step = the row's `step_col`, with every other non-empty
column as a float. `summary_keys` are copied from summary.json into run.summary. Refuses runs that is_complete(rows,
cfg) rejects.

update_notes rewrites ONLY the notes of runs that are already on the server, with the tracker's markdown notes and the
glossary's legend. It sends one upsertBucket(id, notes) per run: no config, tags, summary or history is re-sent and
nothing is re-logged. (Run.update() would re-send config, tags and summary.) Backfilled runs (tag "backfilled") link
to their artefacts commit and say that wandb's Git state shows the HEAD at backfill time; live runs link to their
launch commit. dry_run prints the notes and writes nothing.
"""
import csv
import json
import os
import socket

from spiky.util.wandb_integration.tracker import (_git, github_web_url, gql, normalise_host, run_notes, wandb_config,
                                                  with_committed_flag)

_SET_NOTES = 'mutation($id: String!, $notes: String){ upsertBucket(input: {id: $id, notes: $notes}){ bucket { id } } }'
_GET_NOTES = 'query($e: String!, $p: String!, $r: String!){ project(entityName: $e, name: $p){ run(name: $r){ notes } } }'


def read_metrics(run_dir, step_col='step', require_col=None):
    """[(step, {column: float})] from <run_dir>/metrics.csv: rows with a non-empty `require_col` (all rows if None),
    every other non-empty column converted to float."""
    out = []
    with open(os.path.join(run_dir, 'metrics.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if require_col is not None and not r.get(require_col):
                continue
            out.append((int(r[step_col]), {k: float(v) for k, v in r.items() if k != step_col and v not in (None, '')}))
    return out


def _legend(glossary):
    return glossary.legend_markdown() if glossary is not None else None


def backfill_notes(cfg, *, name, run_dir, commit, branch, host, backfilled, dirty, shown_git_commit, legend=None,
                   description=None):
    """Markdown notes for a run recorded in run_dir (links resolved in the checkout that holds run_dir), with its legend."""
    root = _git(['rev-parse', '--show-toplevel'], os.path.abspath(run_dir))
    root = None if root in ('', 'unknown') else root
    sha = _git(['rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}'], root) if commit and root else 'unknown'
    sha = None if sha in ('', 'unknown') else sha
    rel = os.path.relpath(os.path.abspath(run_dir), root) if root else None
    info = with_committed_flag(dict(root=root, sha=sha, branch=branch, dirty=dirty,
                                    rel=None if rel is None or rel.startswith('..') else rel,
                                    web=github_web_url(_git(['remote', 'get-url', 'origin'], root)) if root else None))
    extra = []
    if backfilled:
        extra.append('- **Backfilled** from the committed metrics.csv / summary.json after the run: only the series '
                     'recorded there, at the steps recorded there (no train/\\*, time/\\* or system series logged live).')
        if shown_git_commit and sha and shown_git_commit != sha:
            extra.append(f"- ⚠ wandb's **Git state** on this run shows `{shown_git_commit[:8]}`, the repository HEAD "
                         f"when the backfill ran, not this run's commit. Its artefacts commit is `{sha[:8]}` (linked "
                         f'above).')
    return run_notes(cfg, exp_name=name, info=info, host=host, legend=legend, description=description,
                     code_label='Code + artefacts (artefacts commit)' if backfilled else 'Code at launch',
                     extra_lines=extra)


def _base():
    base = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/')
    if not base:
        raise RuntimeError('WANDB_BASE_URL must be set (never backfill to wandb.ai)')
    return base


def backfill_run(run_dir, *, project, entity=None, group=None, branch=None, host=None, name=None, tags=(),
                 config_extra=None, config_renames=None, summary_keys=(), step_col='step', require_col=None,
                 is_complete=None, mode='offline', description=None, glossary=None, wandb=None):
    """Upload one finished run (see the module docstring). Returns the local wandb run folder, or None if refused."""
    if mode not in ('online', 'offline'):
        raise ValueError(f'mode must be online or offline, got {mode!r}')
    _base()
    entity = entity or os.environ.get('WANDB_ENTITY')
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    summ_path = os.path.join(run_dir, 'summary.json')
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    name = name or cfg.get('exp_name') or os.path.basename(os.path.abspath(run_dir))
    rows = read_metrics(run_dir, step_col, require_col)
    ok, why = (bool(rows), 'no rows') if is_complete is None or not rows else is_complete(rows, cfg)
    if not ok:
        print(f'SKIP {name}: metrics.csv incomplete ({why}; {len(rows)} rows, last step {rows[-1][0] if rows else None})')
        return None
    if wandb is None:
        import wandb
    rd = os.path.abspath(run_dir)
    commit = _git(['log', '-1', '--format=%h', '--', 'metrics.csv'], rd) or 'unknown'
    host = normalise_host(host or socket.gethostname())
    config = wandb_config(cfg, dict(branch=branch, commit=commit, commit_kind='artefacts commit', host=host,
                                    backfilled=True,
                                    backfill_source='metrics.csv + summary.json' if summ else 'metrics.csv',
                                    **(config_extra or {})), renames=config_renames)
    all_tags = [t for t in ['backfilled', group, branch, commit, host] + list(tags) if t and t != 'unknown']
    notes = backfill_notes(cfg, name=name, run_dir=rd, commit=commit, branch=branch, host=host, backfilled=True,
                           dirty=None, shown_git_commit=_git(['rev-parse', 'HEAD'], rd), legend=_legend(glossary),
                           description=description)
    run = wandb.init(project=project, entity=entity, group=group, job_type='train',
                     name=name, id=name, resume='allow', tags=all_tags, config=config, mode=mode,
                     dir=os.environ.get('WANDB_DIR', os.path.expanduser('~/.cache/wandb')), notes=notes,
                     settings=wandb.Settings(init_timeout=60, console='off', x_disable_stats=True))
    for step, row in rows:
        run.log(row, step=step)
    for k in summary_keys:
        if k in summ:
            run.summary[k] = summ[k]
    d = os.path.dirname(run.dir)
    wandb.finish()
    print(f'backfilled {name}: {len(rows)} rows, commit {commit}, host {host}, mode {mode} -> {d}')
    return d


def set_notes(api, entity, project, run, notes):
    """upsertBucket(id, notes) on an existing run (a wandb.Api run object); True if the notes read back identical."""
    gql(api, _SET_NOTES, {'id': run.storage_id, 'notes': notes})
    back = gql(api, _GET_NOTES, {'e': entity, 'p': project, 'r': run.name})['project']['run']['notes']
    return back == notes


def update_notes(api, entity, project, runs, *, glossary=None, descriptions=None, dry_run=False):
    """Rewrite the notes of the runs {name: run_dir} already in entity/project, legend included (module docstring)."""
    legend = _legend(glossary)
    results = {}
    for name, run_dir in runs.items():
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        r = api.run(f'{entity}/{project}/{name}')
        conf, backfilled = r.config, 'backfilled' in (r.tags or [])
        notes = backfill_notes(cfg, name=name, run_dir=run_dir, commit=conf.get('commit'), branch=conf.get('branch'),
                               host=normalise_host(conf.get('host')), backfilled=backfilled,
                               dirty=None if backfilled else conf.get('commit_dirty'),
                               shown_git_commit=((r.metadata or {}).get('git') or {}).get('commit'), legend=legend,
                               description=(descriptions or {}).get(name))
        if dry_run:
            print(f'===== {name} ({"backfilled" if backfilled else "live"})\n{notes}\n')
            results[name] = notes
            continue
        same = set_notes(api, entity, project, r, notes)
        print(f'notes set on {name}: {len(notes)} chars, read back identical: {same}')
        results[name] = same
    return results
