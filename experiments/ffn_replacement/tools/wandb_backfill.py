"""Backfill finished ffn_replacement runs into wandb from their committed artefacts (metrics.csv, config.json,
summary.json). Clearly marked: tag "backfilled", config backfilled=True, notes say so. A thin wrapper over
spiky.util.wandb_integration.backfill holding this project's data.

    WANDB_BASE_URL=... [WANDB_ENTITY=...] python wandb_backfill.py <run_dir_name> ... [--mode offline|online]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_backfill.py --notes-only [--dry-run] <run_dir_name> ...

Same organisation as the live tracker (wandb_tracking.py): project Spiky, group ffn_replacement, job_type train,
name = id = the run folder, tags + config with branch / commit / host and the LUT tags. commit = the commit that
recorded the run's artefacts (git log on its metrics.csv); host from HOST_OVERRIDE where the run did not train on
gpustar. Logged per eval row (rows with val_bpb) at step = the row's step: val_bpb, train_loss (ema) and every
other metrics.csv column. Step timing was never written to metrics.csv, so it is not backfilled. The run's summary
keys go to run.summary. Refuses runs whose metrics.csv is incomplete (complete()).

--notes-only rewrites ONLY the notes of runs already on the server (one upsertBucket per run; nothing else re-sent).
--dry-run prints the notes and writes nothing.
"""
import json
import os
import sys

import metric_glossary as MG
from spiky.util.wandb_integration import backfill as B
from wandb_tracking import CONFIG_RENAMES, GROUP, PROJECT, batch_config, lut_tags

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')
BRANCH = 'research/ffn_replacement_fix'
DEFAULT_HOST = 'gpustar'
SUMMARY_KEYS = ('final_val_bpb', 'best_val_bpb', 'training_time_hours', 'total_params')
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


def complete(rows, cfg):
    """A finished run: an eval row every eval_every steps, the last one at n_steps."""
    steps, n = [s for s, _ in rows], int(cfg['n_steps'])
    ok = bool(steps) and steps[-1] == n and len(steps) == n // int(cfg['eval_every'])
    return ok, f'{len(steps)} eval rows, last {steps[-1] if steps else None}'


def backfill(names, mode):
    for name in names:
        rd = os.path.join(RC, name)
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        summ = json.load(open(os.path.join(rd, 'summary.json')))
        ga = cfg['total_batch_size'] // (cfg['device_batch_size'] * cfg['seq_len'])
        B.backfill_run(rd, project=PROJECT, entity=os.environ.get('WANDB_ENTITY'), group=GROUP, branch=BRANCH,
                       host=HOST_OVERRIDE.get(name, DEFAULT_HOST), name=name,
                       tags=lut_tags(cfg) + [f"line:{LINE.get(name, 'confidence_form')}"],
                       config_extra=batch_config(cfg, ga, summ.get('total_params')), config_renames=CONFIG_RENAMES,
                       summary_keys=SUMMARY_KEYS, require_col='val_bpb', is_complete=complete, mode=mode,
                       description=DESCRIPTION_OVERRIDE.get(name), panel_title=MG.PANEL_TITLE)


def notes_only(names, dry):
    if not os.environ.get('WANDB_BASE_URL') or not os.environ.get('WANDB_ENTITY'):
        sys.exit('WANDB_BASE_URL and WANDB_ENTITY must be set for --notes-only (never write to wandb.ai)')
    import wandb
    api = wandb.Api(timeout=60)
    B.update_notes(api, os.environ['WANDB_ENTITY'], PROJECT, {n: os.path.join(RC, n) for n in names},
                   descriptions=DESCRIPTION_OVERRIDE, panel_title=MG.PANEL_TITLE, dry_run=dry)


def main():
    argv = sys.argv[1:]
    if not os.environ.get('WANDB_BASE_URL'):
        sys.exit('WANDB_BASE_URL must be set (never backfill to wandb.ai)')
    if '--notes-only' in argv:
        return notes_only([a for a in argv if not a.startswith('--')], '--dry-run' in argv)
    mode = 'offline'
    if '--mode' in argv:
        i = argv.index('--mode')
        mode = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if mode not in ('online', 'offline'):
        sys.exit(f'--mode must be online or offline, got {mode!r}')
    backfill([a for a in argv if not a.startswith('--')], mode)


if __name__ == '__main__':
    main()
