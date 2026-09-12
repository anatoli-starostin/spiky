"""Backfill finished ffn_replacement runs into wandb from their committed artefacts (metrics.csv,
config.json, summary.json). Clearly marked: tag "backfilled", config backfilled=True, notes say so.

    WANDB_BASE_URL=... [WANDB_ENTITY=...] python wandb_backfill.py <run_dir_name> ... [--mode offline|online]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_backfill.py --notes-only [--dry-run] <run_dir_name> ...

Same organisation as the live tracker (wandb_tracking.py / claude/wandb.md): project Spiky, group
ffn_replacement, job_type train, name = id = exp_name, tags + config with branch / commit / host.
commit = the commit that recorded the run's artefacts (git log on its metrics.csv); host from the run's
record where the run did not run on this machine. Logged per eval row at step = the row's step: val_bpb,
train_loss (ema) and every other metrics.csv column (ln norms, learned-confidence per-layer
lm_g / lm_beta / lm_gamma, tau). Step timing was never written to metrics.csv, so it is not backfilled.
The run's corrected-eval summary goes to run.summary. Refuses runs whose metrics.csv is incomplete.

--notes-only rewrites ONLY the notes of runs that are already on the server, with the tracker's markdown notes
(description + links; tools/metric_glossary.py for what the keys mean). It sends one upsertBucket(id, notes) per
run: no config, tags, summary or history is re-sent, nothing is re-logged and no artifact is attached (finished
runs link to the glossary report instead). Backfilled runs (tag "backfilled") link to their artefacts commit and
say that wandb's Git state shows the HEAD at backfill time; live runs link to their launch commit.
--dry-run prints the notes and writes nothing.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')
sys.path.insert(0, HERE)
from wandb_tracking import (GROUP, PROJECT, _git, github_web_url, glossary_report_link, gql,  # noqa: E402
                            normalise_host, run_notes, wandb_config, with_committed_flag)

BRANCH = 'research/ffn_replacement_fix'
# where each backfilled run actually trained (from its run record); default: this repo's usual box
HOST_OVERRIDE = {'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1': 'nebius-h100'}
LINE = {'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1': 'baseline_margin',
        'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1': 'baseline_blend_n2'}
# run folders are read-only once run, so a description that the _arch_note's opening would get wrong lives here
DESCRIPTION_OVERRIDE = {
    'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1': (
        'Top-2 blended read-out (lut_read_top_n 2) with a learnable per-layer blend temperature tau, initialised '
        'flat at 0.5; otherwise exp_g_0193 (margin, no z_norm). Trained on nebius-h100.'),
}
_SET_NOTES = 'mutation($id: String!, $notes: String){ upsertBucket(input: {id: $id, notes: $notes}){ bucket { id } } }'
_GET_NOTES = 'query($e: String!, $p: String!, $r: String!){ project(entityName: $e, name: $p){ run(name: $r){ notes } } }'


def notes_for(name, cfg, *, commit, branch, host, backfilled, dirty, shown_git_commit, report, report_label):
    """Markdown notes for a run already recorded in runs_corrected/<name>."""
    root = _git(['rev-parse', '--show-toplevel'], HERE)
    sha = _git(['rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}'], root) if commit else 'unknown'
    sha = None if sha in ('', 'unknown') else sha
    info = with_committed_flag(dict(root=root, sha=sha, branch=branch, dirty=dirty,
                                    rel=os.path.relpath(os.path.join(RC, name), root),
                                    web=github_web_url(_git(['remote', 'get-url', 'origin'], root))))
    extra = []
    if backfilled:
        extra.append('- **Backfilled** from the committed metrics.csv / summary.json after the run: no train/\\*, '
                     'time/\\* or system series, and train\\_loss is sampled at eval steps only.')
        if shown_git_commit and sha and shown_git_commit != sha:
            extra.append(f"- ⚠ wandb's **Git state** on this run shows `{shown_git_commit[:8]}`, the repository HEAD "
                         f"when the backfill ran, not this run's commit. Its artefacts commit is `{sha[:8]}` (linked "
                         f'above).')
    return run_notes(cfg, exp_name=name, info=info, host=host, report_url=report, report_label=report_label,
                     description=DESCRIPTION_OVERRIDE.get(name),
                     code_label='Code + artefacts (artefacts commit)' if backfilled else 'Code at launch',
                     extra_lines=extra)


def _env(require_entity=False):
    base, entity = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/'), os.environ.get('WANDB_ENTITY')
    if not base:
        sys.exit('WANDB_BASE_URL must be set (never backfill to wandb.ai)')
    if require_entity and not entity:
        sys.exit('WANDB_ENTITY must be set for --notes-only')
    return base, entity


def notes_only(names, dry):
    base, entity = _env(require_entity=True)
    import wandb
    api = wandb.Api(timeout=60)
    report, report_label = glossary_report_link(base, entity, PROJECT)
    print(f'glossary link: {report_label} -> {report}')
    for name in names:
        cfg = json.load(open(os.path.join(RC, name, 'config.json')))
        r = api.run(f'{entity}/{PROJECT}/{name}')
        conf, backfilled = r.config, 'backfilled' in (r.tags or [])
        notes = notes_for(name, cfg, commit=conf.get('commit'), branch=conf.get('branch'),
                          host=normalise_host(conf.get('host')), backfilled=backfilled,
                          dirty=None if backfilled else conf.get('commit_dirty'),
                          shown_git_commit=((r.metadata or {}).get('git') or {}).get('commit'),
                          report=report, report_label=report_label)
        if dry:
            print(f'===== {name} ({"backfilled" if backfilled else "live"})\n{notes}\n')
            continue
        gql(api, _SET_NOTES, {'id': r.storage_id, 'notes': notes})
        back = gql(api, _GET_NOTES, {'e': entity, 'p': PROJECT, 'r': name})['project']['run']['notes']
        print(f'notes set on {name}: {len(notes)} chars, read back identical: {back == notes}')


def main():
    argv = sys.argv[1:]
    if '--notes-only' in argv:
        return notes_only([a for a in argv if not a.startswith('--')], '--dry-run' in argv)
    mode = 'offline'
    if '--mode' in argv:
        i = argv.index('--mode')
        mode = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if mode not in ('online', 'offline'):
        sys.exit(f'--mode must be online or offline, got {mode!r}')
    args = [a for a in argv if not a.startswith('--')]
    base, entity = _env()
    import wandb
    repo = os.path.dirname(os.path.dirname(HERE))
    report, report_label = (glossary_report_link(base, entity, PROJECT, online=mode == 'online')
                            if entity else (None, None))
    for name in args:
        rd = os.path.join(RC, name)
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        summ = json.load(open(os.path.join(rd, 'summary.json')))
        rows = list(csv.DictReader(open(os.path.join(rd, 'metrics.csv'))))
        steps = [int(r['step']) for r in rows if r.get('val_bpb')]
        n = int(cfg['n_steps'])
        if not steps or steps[-1] != n or len(steps) != n // int(cfg['eval_every']):
            print(f'SKIP {name}: metrics.csv incomplete ({len(steps)} eval rows, last {steps[-1] if steps else None})')
            continue
        commit = _git(['log', '-1', '--format=%h', '--', os.path.join(rd, 'metrics.csv')], repo)
        host = normalise_host(HOST_OVERRIDE.get(name, 'gpustar'))
        form = cfg.get('lut_confidence_form', 'margin')
        dbs, ga = cfg['device_batch_size'], cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len'])
        config = wandb_config(cfg, branch=BRANCH, commit=commit, commit_kind='artefacts commit',
                              host=host, backfilled=True, backfill_source='metrics.csv + summary.json',
                              grad_accum=ga, batch_rows_per_step=dbs * ga, tokens_per_step=dbs * ga * cfg['seq_len'],
                              total_params=summ.get('total_params'))
        tags = ['backfilled', GROUP, BRANCH, commit, host, f'form:{form}',
                f"line:{LINE.get(name, 'confidence_form')}"]
        if cfg.get('lut_learned_margin_freeze_g'):
            tags.append('learned_margin_freeze_g')
        notes = notes_for(name, cfg, commit=commit, branch=BRANCH, host=host, backfilled=True, dirty=None,
                          shown_git_commit=_git(['rev-parse', 'HEAD'], repo), report=report, report_label=report_label)
        run = wandb.init(project=PROJECT, entity=entity, group=GROUP, job_type='train',
                         name=name, id=name, resume='allow', tags=tags, config=config, mode=mode,
                         dir=os.environ.get('WANDB_DIR', os.path.expanduser('~/.cache/wandb')), notes=notes,
                         settings=wandb.Settings(init_timeout=60, console='off', x_disable_stats=True))
        for r in rows:
            if not r.get('val_bpb'):
                continue
            row = {'val_bpb': float(r['val_bpb']), 'train_loss': float(r['train_loss'])}
            for k, v in r.items():
                if k not in ('step', 'val_bpb', 'train_loss') and v not in (None, ''):
                    row[k] = float(v)
            run.log(row, step=int(r['step']))
        for k in ('final_val_bpb', 'best_val_bpb', 'training_time_hours', 'total_params'):
            if k in summ:
                run.summary[k] = summ[k]
        d = os.path.dirname(run.dir)
        wandb.finish()
        print(f'backfilled {name}: {len(steps)} eval rows, commit {commit}, host {host}, mode {mode} -> {d}')


if __name__ == '__main__':
    main()
