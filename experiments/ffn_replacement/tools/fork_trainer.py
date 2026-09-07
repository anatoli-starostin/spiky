"""fork_trainer(src, dst) — copy a trainer into a run folder, refusing legacy eval code.

WHY THIS EXISTS. `runs_corrected/` holds one folder per run, and the established way to set
up a new experiment is to fork a neighbour's config and trainer. For a long time 23 of those
folders contained the ORIGINAL batch-coupled trainer (val loader built at the training
`device_batch_size`, `eval_steps=10`), kept as the record of what actually ran. Forking one
of those by mistake would have silently reintroduced the exact bug this branch exists to fix
— and it would not have been visible in the results, only in a bpb that quietly wasn't
comparable to anything else. 22 of the 23 have since been corrected in place; this guard
makes the failure mode unreachable rather than merely unlikely.

WHAT IT CHECKS, AND WHEN. The check fires **only when a trainer is being forked for a new
run** — i.e. at the moment of copy, on the SOURCE file. It is not a repo-wide scan and it
does not care whether a legacy file exists somewhere on disk; a run folder may keep any
trainer it likes without this ever complaining. That distinction is deliberate:
`exp_n_0138_outcompress_only_H4_nap8_tph128` still carries its original uncorrected trainer
(deliberately — its number cannot be corrected and is being handled separately), and this
guard must not trip merely because that file is present. It trips only if someone tries to
make a NEW run out of it.

    from fork_trainer import fork_trainer
    fork_trainer(os.path.join(FR, 'train_fixed.py'), os.path.join(d, 'train.py'))
"""
import os
import shutil

_LEGACY = 'val_loader_factory'
_REQUIRED = 'fixed_eval'


def check_trainer(path: str) -> None:
    """Raise unless `path` is a trainer that uses the corrected, shared eval set."""
    with open(path, encoding='utf-8', errors='ignore') as f:
        src = f.read()
    if _REQUIRED not in src:
        raise RuntimeError(
            f"refusing to fork {path}: it does not import the shared corrected eval "
            f"(tools/fixed_eval.py). Forking a trainer that scores with its own val loader "
            f"re-couples the eval window to the training device_batch_size — the bug "
            f"documented in experiments/ffn_replacement/FIXED_EVAL.md. Fork "
            f"experiments/ffn_replacement/train_fixed.py, or a run whose train.py imports "
            f"fixed_eval.")
    if _LEGACY in src:
        raise RuntimeError(
            f"refusing to fork {path}: it still builds a val loader of its own "
            f"({_LEGACY!r}), so its eval window would depend on the training batch size. "
            f"See experiments/ffn_replacement/FIXED_EVAL.md.")


def fork_trainer(src: str, dst: str) -> str:
    """Guarded `shutil.copy`: check the source, copy, then re-check what landed."""
    check_trainer(src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copy(src, dst)
    check_trainer(dst)          # cheap, and catches a truncated or clobbered copy
    return dst
