"""Backfill finished ffn_replacement runs into wandb from their committed artefacts (metrics.csv,
config.json, summary.json). Clearly marked: tag "backfilled", config backfilled=True, notes say so.

    WANDB_BASE_URL=... [WANDB_ENTITY=...] python wandb_backfill.py <run_dir_name> ... [--mode offline|online]

Same organisation as the live tracker (wandb_tracking.py / claude/wandb.md): project Spiky, group
ffn_replacement, job_type train, name = id = exp_name, tags + config with branch / commit / host.
commit = the commit that recorded the run's artefacts (git log on its metrics.csv); host from the run's
record where the run did not run on this machine. Logged per eval row at step = the row's step: val_bpb,
train_loss (ema) and every other metrics.csv column (ln norms, learned-confidence per-layer
lm_g / lm_beta / lm_gamma, tau). Step timing was never written to metrics.csv, so it is not backfilled.
The run's corrected-eval summary goes to run.summary. Refuses runs whose metrics.csv is incomplete.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')
sys.path.insert(0, HERE)
from wandb_tracking import GROUP, PROJECT, _git                          # noqa: E402

# where each backfilled run actually trained (from its run record); default: this repo's usual box
HOST_OVERRIDE = {'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1': 'nebius-h100'}
LINE = {'exp_g_0193_B16k_light_margin_tph128_noznorm_seed1': 'baseline_margin',
        'exp_g_0195_B16k_light_margin_blend_n2_tau_learn0p5_seed1': 'baseline_blend_n2'}


def main():
    argv = sys.argv[1:]
    mode = 'offline'
    if '--mode' in argv:
        i = argv.index('--mode')
        mode = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if mode not in ('online', 'offline'):
        sys.exit(f'--mode must be online or offline, got {mode!r}')
    args = [a for a in argv if not a.startswith('--')]
    if not os.environ.get('WANDB_BASE_URL'):
        sys.exit('WANDB_BASE_URL must be set (never backfill to wandb.ai)')
    import wandb
    repo = os.path.dirname(os.path.dirname(HERE))
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
        host = HOST_OVERRIDE.get(name, 'gpustar')
        form = cfg.get('lut_confidence_form', 'margin')
        dbs, ga = cfg['device_batch_size'], cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len'])
        config = {k: v for k, v in cfg.items() if k != '_arch_note'}
        config.update(branch='research/ffn_replacement_fix', commit=commit, commit_kind='artefacts commit',
                      host=host, backfilled=True, backfill_source='metrics.csv + summary.json',
                      grad_accum=ga, batch_rows_per_step=dbs * ga, tokens_per_step=dbs * ga * cfg['seq_len'],
                      total_params=summ.get('total_params'))
        tags = ['backfilled', GROUP, 'research/ffn_replacement_fix', commit, host, f'form:{form}',
                f"line:{LINE.get(name, 'confidence_form')}"]
        if cfg.get('lut_learned_margin_freeze_g'):
            tags.append('learned_margin_freeze_g')
        run = wandb.init(project=PROJECT, entity=os.environ.get('WANDB_ENTITY'), group=GROUP, job_type='train',
                         name=name, id=name, resume='allow', tags=tags, config=config, mode=mode,
                         dir=os.environ.get('WANDB_DIR', os.path.expanduser('~/.cache/wandb')),
                         notes=('BACKFILLED from committed metrics.csv/summary.json (not a live run). '
                                + (cfg.get('_arch_note') or ''))[:2000],
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
