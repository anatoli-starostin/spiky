"""Publish, verify and audit the ffn_replacement metric glossary (metric_glossary.py) on the self-hosted wandb.
A thin wrapper over spiky.util.wandb_integration.workspace, which documents the commands, targets and exit codes.

    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py publish [--project Spiky] [--shared TITLE | --user NAME] [--dry-run] [--if-idle MIN]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py verify  [--project Spiky] [--shared TITLE | --user NAME]
    WANDB_BASE_URL=... WANDB_ENTITY=... python wandb_glossary.py audit   [--project Spiky]

The Spiky workspace panel lives in the shared saved view "Spiky — described" (publish/verify --shared 'Spiky — described').
Project data held here: project Spiky and its README intro, metric_glossary.py, the runs_corrected/*/metrics.csv
headers and the keys wandb_tracking.py emits (both audited).

No secrets: auth from ~/.netrc; server and entity from the environment.
"""
import glob
import os
import sys

import metric_glossary as MG
from spiky.util.wandb_integration import workspace as W
from wandb_tracking import EVAL_KEYS, PROJECT, SUMMARY_KEYS, TRAIN_KEYS

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(os.path.dirname(HERE), 'runs_corrected')
README_INTRO = ("One project for all spiky experiments; a run's `group` is its experiment family "
                '(`experiments/<family>/`).')


def main(argv):
    return W.main(argv, glossary=MG, project=PROJECT, readme_intro=README_INTRO,
                  audit_csvs=sorted(glob.glob(os.path.join(RC, '*', 'metrics.csv'))),
                  audit_extra_keys={'tracker keys (wandb_tracking.py)': TRAIN_KEYS + EVAL_KEYS + SUMMARY_KEYS})


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
