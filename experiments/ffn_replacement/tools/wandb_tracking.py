"""The ffn_replacement layer over spiky.util.wandb_integration.tracker (conventions: claude/wandb.md on main).

Frozen run folders (runs_corrected/*/train.py) and the train_fixed.py template use it UNCHANGED, with the old calls:

    from wandb_tracking import Tracker
    tracker = Tracker.start(cfg, EXP_DIR, grad_accum=..., total_params=..., extra_tags=[...])   # after the model is built
    tracker.train_step(step, loss, ema, lr, grad_norm=None)  # once per optimiser step (logged every LOG_EVERY)
    tracker.eval_step(step, bpb, ema, extra, model)          # at each eval, after metrics.csv is written
    tracker.finish(summary)                                  # at the end, after summary/checkpoint are written

This file only binds this project onto the package:
  * project "Spiky" and group "ffn_replacement";
  * tags form:<lut_confidence_form> and learned_margin_freeze_g (lut_tags), after the caller's extra_tags;
  * the learned-margin scalars lm_g / lm_beta / lm_gamma per layer at each eval (learned_confidence_by_layer);
  * metric_glossary.py as the glossary; eval_steps sent as eval_steps_legacy_ignored; derived batch sizes in the
    config (batch_config);
  * the old positional rows -> the package's dict rows (train/loss, train/loss_ema, train/lr, train/grad_norm;
    val_bpb, train_loss + extra).
Everything else -- online when the server answers a 2 s probe, else offline; the bounded queue that keeps wandb off
the training thread; the finish() deadline; the notes, glossary artifact and drift check -- is the package's: see
its docstring.

OPTIONAL, NEVER FATAL: if the package cannot be imported (a checkout from before it was merged, or these tools used
without the spiky install), Tracker is a no-op that says so in one line and training runs exactly as without it.
"""
try:
    from spiky.util.wandb_integration import tracker as _T
    _IMPORT_ERROR = None
except Exception as _e:                                          # tracking is optional: never kill a run
    _T, _IMPORT_ERROR = None, _e

try:
    import metric_glossary as MG
except Exception:                                                # pragma: no cover
    MG = None

PROJECT = 'Spiky'
GROUP = 'ffn_replacement'
LOG_EVERY = 10
CONFIG_RENAMES = {'eval_steps': 'eval_steps_legacy_ignored'}

# the keys this layer emits (test_metric_glossary.py checks each has a glossary entry; wandb_glossary.py audits them)
TRAIN_KEYS = ('train/loss', 'train/loss_ema', 'train/lr', 'time/sec_per_step', 'train/grad_norm')
EVAL_KEYS = ('val_bpb', 'train_loss')
SUMMARY_KEYS = ('exp_name', 'best_val_bpb', 'final_val_bpb', 'total_params', 'training_time_hours',
                'glossary/undocumented')


def learned_confidence_by_layer(model):
    """{'lm_g_L0': .., 'lm_beta_L0': .., 'lm_gamma_L0': .., ...} for learned_margin layers, else {}."""
    out, i = {}, 0
    for mod in model.modules():
        get = getattr(mod, 'learned_confidence_values', None)
        if get is None or getattr(mod, 'confidence_form', None) != 'learned_margin':
            continue
        v = get()
        for k in ('g', 'beta', 'gamma'):
            out[f'lm_{k}_L{i}'] = float(v[k])
        i += 1
    return out


def lut_tags(cfg):
    """form:<lut_confidence_form> (default margin), plus learned_margin_freeze_g when set."""
    tags = [f"form:{cfg.get('lut_confidence_form', 'margin')}"]
    if cfg.get('lut_learned_margin_freeze_g'):
        tags.append('learned_margin_freeze_g')
    return tags


def batch_config(cfg, grad_accum, total_params):
    """Derived config values: grad_accum, total_params, batch_rows_per_step, tokens_per_step."""
    dbs, seq = cfg.get('device_batch_size'), cfg.get('seq_len')
    return dict(grad_accum=grad_accum, total_params=total_params,
                batch_rows_per_step=(dbs * grad_accum if dbs and grad_accum else None),
                tokens_per_step=(dbs * grad_accum * seq if dbs and grad_accum and seq else None))


class Tracker:
    """The old-signature tracker: wraps the package's Tracker, or does nothing when the package is unavailable."""

    def __init__(self, inner=None, reason=''):
        self._inner = inner
        self.reason = reason or (getattr(inner, 'reason', '') if inner is not None else '')

    @property
    def active(self):
        return self._inner is not None and self._inner.active

    @property
    def mode(self):
        return getattr(self._inner, 'mode', None)

    @property
    def dropped(self):
        return getattr(self._inner, 'dropped', 0)

    @classmethod
    def start(cls, cfg, exp_dir, grad_accum=None, total_params=None, job_type='train', extra_tags=()):
        if _T is None:
            print(f'[wandb] off: spiky.util.wandb_integration is not importable ({type(_IMPORT_ERROR).__name__}: '
                  f'{_IMPORT_ERROR}) -- training continues', flush=True)
            return cls(reason='package not importable')
        try:
            inner = _T.Tracker.start(cfg, exp_dir, project=PROJECT, group=GROUP, job_type=job_type,
                                     tags=list(extra_tags or ()), extra_tags=lut_tags,
                                     extra_eval_metrics=learned_confidence_by_layer, glossary=MG,
                                     config_extra=batch_config(cfg, grad_accum, total_params),
                                     config_renames=CONFIG_RENAMES, log_every=LOG_EVERY)
        except Exception as e:                                   # the package never raises here; belt and braces
            print(f'[wandb] off: start failed: {type(e).__name__}: {e} -- training continues', flush=True)
            return cls(reason='start failed')
        return cls(inner)

    def train_step(self, step, loss, ema, lr, grad_norm=None):
        if self._inner is None:
            return
        row = {'train/loss': loss, 'train/loss_ema': ema, 'train/lr': lr}
        if grad_norm is not None:
            row['train/grad_norm'] = grad_norm                   # a tensor: converted on logged steps only
        self._inner.train_step(step, row)

    def eval_step(self, step, bpb, ema, extra=None, model=None):
        if self._inner is None:
            return
        row = {'val_bpb': bpb, 'train_loss': ema}
        row.update(extra or {})
        self._inner.eval_step(step, row, model)

    def finish(self, summary=None):
        if self._inner is not None:
            self._inner.finish(summary)
