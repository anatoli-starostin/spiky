"""Optional, NON-FATAL wandb tracking for the ffn_replacement trainers (conventions: claude/wandb.md on main).

    from wandb_tracking import Tracker
    tracker = Tracker.start(cfg, EXP_DIR, grad_accum=..., total_params=...)   # after the model is built
    tracker.train_step(step, loss, ema, lr)          # once per optimiser step (logged every LOG_EVERY)
    tracker.eval_step(step, bpb, ema, extra, model)  # at each eval, after metrics.csv is written
    tracker.finish(summary)                          # at the end

GUARANTEES -- the tracker can never cost a run:
  * OFF unless WANDB_BASE_URL is set (so a run can never silently go to wandb.ai), unless
    WANDB_MODE=disabled, and unless `import wandb` works.
  * Every wandb call is wrapped; the first exception disables the tracker for the rest of the run
    with ONE warning line. Nothing is raised into the training loop; metrics.csv is written by the
    trainer exactly as before, whatever happens here.
  * It reads Python floats the trainer already has and, at evals, the learned confidence scalars via
    their read-only accessor. It never touches torch's RNG, the model, the optimiser or the data.
  * No secrets: authentication comes from ~/.netrc (`wandb login`); the server URL and entity come
    from the environment (WANDB_BASE_URL, WANDB_ENTITY) -- deployment values stay out of the repo.

MODE. OFFLINE BY DEFAULT: the run is written under $WANDB_DIR (default ~/.cache/wandb) and uploaded
afterwards with `wandb sync <dir>` (the path is printed at start and at finish) -- from the host, or
from the cage with `sbox --net tailnet -- wandb sync <dir>` (tailnet-only egress; bare `sbox` has no
network). Online only when WANDB_MODE=online is set AND a 2 s TCP probe of the server succeeds.

ORGANISATION (claude/wandb.md section 4): project "Spiky"; group = the experiments/<family>/ folder,
"ffn_replacement"; job_type "train"; name and id = exp_name (unique per run folder, so a crashed
run resumes instead of duplicating); tags = family, branch, short commit, host, confidence form;
config = branch, commit, host + the full config.json + derived batch sizes.
"""
import os
import socket
import subprocess
import time
from urllib.parse import urlparse

PROJECT = 'Spiky'
GROUP = 'ffn_replacement'
LOG_EVERY = 10


def _git(args, cwd):
    try:
        return subprocess.check_output(['git'] + args, cwd=cwd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except Exception:
        return 'unknown'


def _reachable(url, timeout=2.0):
    try:
        u = urlparse(url)
        with socket.create_connection((u.hostname, u.port or (443 if u.scheme == 'https' else 80)), timeout=timeout):
            return True
    except Exception:
        return False


def _writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.write_probe')
        with open(probe, 'w') as f:
            f.write('x')
        os.remove(probe)
        return True
    except Exception:
        return False


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


class Tracker:
    def __init__(self, run=None, wandb=None, reason=''):
        self.run, self._wandb, self.reason = run, wandb, reason
        self._t_last, self._s_last = time.time(), 0

    @property
    def active(self):
        return self.run is not None

    def _fail(self, where, exc):
        print(f'[wandb] disabled after an error in {where}: {type(exc).__name__}: {exc} '
              f'-- training continues, metrics.csv unaffected', flush=True)
        self.run = None

    @classmethod
    def start(cls, cfg, exp_dir, grad_accum=None, total_params=None, job_type='train', extra_tags=()):
        base = os.environ.get('WANDB_BASE_URL')
        mode = os.environ.get('WANDB_MODE')
        if not base:
            print('[wandb] off: WANDB_BASE_URL not set', flush=True)
            return cls(reason='no WANDB_BASE_URL')
        if mode == 'disabled':
            print('[wandb] off: WANDB_MODE=disabled', flush=True)
            return cls(reason='disabled')
        try:
            import wandb
        except Exception as e:                                   # pragma: no cover
            print(f'[wandb] off: import failed ({type(e).__name__})', flush=True)
            return cls(reason='import failed')
        try:
            root = os.path.expanduser('~/.cache/wandb')
            # inside the cage the default ~/.config and ~/.local are read-only: keep wandb's own
            # state in the writable cache when those defaults cannot be written (setdefault only)
            for var, default, sub in (('WANDB_CONFIG_DIR', '~/.config/wandb', 'config'),
                                      ('WANDB_DATA_DIR', '~/.local/share/wandb', 'data')):
                if var not in os.environ and not _writable_dir(os.path.expanduser(default)):
                    os.environ[var] = os.path.join(root, sub)
            run_dir = os.environ.get('WANDB_DIR', root)
            os.makedirs(run_dir, exist_ok=True)
            # OFFLINE BY DEFAULT. A forced-online wandb.init against an unreachable server was measured
            # to hang the trainer (>10 min, despite init_timeout) -- so online is used only when it is
            # explicitly requested AND the server answers a 2 s probe right now. Offline never touches
            # the network: sync afterwards with `wandb sync`. (An online run whose network drops
            # mid-run can still block in finish(); by then metrics.csv/summary/checkpoint are written.)
            if mode == 'online' and not _reachable(base):
                print(f'[wandb] WANDB_MODE=online requested but {base} is unreachable -> offline', flush=True)
                mode = 'offline'
            elif mode != 'online':
                mode = 'offline'
            repo = os.path.dirname(os.path.abspath(__file__))           # this checkout, wherever the run dir is
            branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'], repo)
            commit = _git(['rev-parse', '--short', 'HEAD'], repo)
            dirty = _git(['status', '--porcelain', '--untracked-files=no'], repo)
            host = socket.gethostname()
            name = cfg.get('exp_name') or os.path.basename(os.path.abspath(exp_dir))
            dbs, seq = cfg.get('device_batch_size'), cfg.get('seq_len')
            config = {k: v for k, v in cfg.items() if k != '_arch_note'}
            config.update(branch=branch, commit=commit, commit_dirty=bool(dirty and dirty != 'unknown'), host=host,
                          grad_accum=grad_accum, total_params=total_params,
                          batch_rows_per_step=(dbs * grad_accum if dbs and grad_accum else None),
                          tokens_per_step=(dbs * grad_accum * seq if dbs and grad_accum and seq else None))
            tags = [GROUP, branch, commit, host, f"form:{cfg.get('lut_confidence_form', 'margin')}"] + list(extra_tags)
            if cfg.get('lut_learned_margin_freeze_g'):
                tags.append('learned_margin_freeze_g')
            run = wandb.init(project=PROJECT, entity=os.environ.get('WANDB_ENTITY'), group=GROUP, job_type=job_type,
                             name=name, id=name, resume='allow', tags=[t for t in tags if t and t != 'unknown'],
                             notes=(cfg.get('_arch_note') or '')[:2000] or None, config=config, dir=run_dir,
                             mode=mode, settings=wandb.Settings(init_timeout=60, console='off'))
            print(f'[wandb] {mode}: run {name} -> {run.dir}'
                  + ('  (sync later: wandb sync ' + os.path.dirname(run.dir) + ')' if mode == 'offline' else ''),
                  flush=True)
            return cls(run, wandb)
        except Exception as e:
            print(f'[wandb] off: init failed: {type(e).__name__}: {e} -- training continues', flush=True)
            return cls(reason='init failed')

    def train_step(self, step, loss, ema, lr):
        if self.run is None or not (step % LOG_EVERY == 0 or step == 1):
            return
        try:
            now = time.time()
            dt = (now - self._t_last) / max(step - self._s_last, 1)
            self._t_last, self._s_last = now, step
            self.run.log({'train/loss': loss, 'train/loss_ema': ema, 'train/lr': lr, 'time/sec_per_step': dt},
                         step=step)
        except Exception as e:
            self._fail('train_step', e)

    def eval_step(self, step, bpb, ema, extra=None, model=None):
        if self.run is None:
            return
        try:
            row = {'val_bpb': bpb, 'train_loss': ema}
            row.update(extra or {})
            if model is not None:
                row.update(learned_confidence_by_layer(model))
            self.run.log(row, step=step)
        except Exception as e:
            self._fail('eval_step', e)

    def finish(self, summary=None):
        if self.run is None:
            return
        try:
            for k, v in (summary or {}).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    self.run.summary[k] = v
            d = os.path.dirname(self.run.dir)
            offline = self.run.settings.mode == 'offline' if hasattr(self.run, 'settings') else False
            self._wandb.finish()
            print(f'[wandb] finished' + (f' (offline; sync with: wandb sync {d})' if offline else ''), flush=True)
        except Exception as e:
            self._fail('finish', e)
        finally:
            self.run = None
