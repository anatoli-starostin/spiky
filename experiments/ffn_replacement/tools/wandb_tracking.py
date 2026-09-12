"""Optional, NON-FATAL wandb tracking for the ffn_replacement trainers (conventions: claude/wandb.md on main).

    from wandb_tracking import Tracker
    tracker = Tracker.start(cfg, EXP_DIR, grad_accum=..., total_params=...)   # after the model is built
    tracker.train_step(step, loss, ema, lr)          # once per optimiser step (logged every LOG_EVERY)
    tracker.eval_step(step, bpb, ema, extra, model)  # at each eval, after metrics.csv is written
    tracker.finish(summary)                          # at the end, after summary/checkpoint are written

MODE -- ONLINE BY DEFAULT when the server is reachable:
  * WANDB_BASE_URL unset -> tracker OFF (a run can never silently go to wandb.ai); WANDB_MODE=disabled -> OFF.
  * WANDB_MODE=offline   -> offline: written under $WANDB_DIR (default ~/.cache/wandb), uploaded later with
                            `wandb sync <dir>` (from the host, or `sbox --net tailnet -- wandb sync <dir>`).
  * otherwise            -> a 2 s TCP probe of the server: reachable -> ONLINE (live curves); unreachable ->
                            OFFLINE with one line saying so. A dead server costs ~2 s at start, never a hang
                            (a wandb.init against an unreachable server was measured to block for >11 min).
                            Bare `sbox` has no network, so runs launched there go offline automatically;
                            launch with `sbox --net tailnet -- ...` to log live.

GUARDS -- the tracker can never stall or break training:
  * The training loop never calls wandb. train_step/eval_step put a row on a bounded in-memory queue and
    return; a background thread does run.log(). If wandb stops accepting rows the queue fills and further
    rows are DROPPED (counted, reported at finish) -- the loop never waits.
  * Every wandb network path has capped retries/timeouts (BOUNDS), and finish() has a hard deadline
    (FINISH_TIMEOUT, default 60 s, env WANDB_TRACKER_FINISH_TIMEOUT): past it the tracker gives up on the
    upload, kills its own wandb-core helper processes and returns, so the process exits. The trainer calls
    finish() only after metrics.csv, summary.json and checkpoint.pt are written; the run's local record
    stays on disk for a later `wandb sync`.
  * Any tracker error disables the tracker for the rest of the run with ONE line. Nothing is raised into
    the training loop; metrics.csv is written by the trainer exactly as without the tracker.
  * It reads floats the trainer already has and, at evals, the learned confidence scalars via their
    read-only accessor: no RNG, no graph, no optimiser or data interaction.
  * No secrets: auth from ~/.netrc (`wandb login`); server URL / entity from WANDB_BASE_URL / WANDB_ENTITY.

ORGANISATION (claude/wandb.md section 4): project "Spiky"; group = the experiments/<family>/ folder,
"ffn_replacement"; job_type "train"; name and id = exp_name (unique per run folder, so a crashed
run resumes instead of duplicating); tags = family, branch, short commit, host, confidence form;
config = branch, commit, host + the full config.json + derived batch sizes.
"""
import os
import queue
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

PROJECT = 'Spiky'
GROUP = 'ffn_replacement'
LOG_EVERY = 10
LOG_QUEUE_MAX = 4096
FINISH_TIMEOUT = float(os.environ.get('WANDB_TRACKER_FINISH_TIMEOUT', '60'))
BOUNDS = dict(init_timeout=60, x_graphql_retry_max=5, x_graphql_timeout_seconds=20,
              x_graphql_retry_wait_min_seconds=2, x_graphql_retry_wait_max_seconds=10,
              x_file_stream_retry_max=15, x_file_stream_timeout_seconds=30,
              x_file_stream_retry_wait_min_seconds=2, x_file_stream_retry_wait_max_seconds=20,
              x_file_transfer_retry_max=5, x_file_transfer_timeout_seconds=60,
              finish_timeout=FINISH_TIMEOUT, finish_timeout_raises=False)


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


def _kill_own_wandb_helpers():
    """SIGKILL the wandb-core / wandb-xpu processes descended from THIS process (never anyone else's)."""
    me, killed = os.getpid(), []
    for pid in (int(x) for x in os.listdir('/proc') if x.isdigit()):
        try:
            exe = os.path.basename(open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')[0].decode(errors='ignore'))
            if not exe.startswith(('wandb-core', 'wandb-xpu')):
                continue
            p = pid
            for _ in range(8):
                p = int(open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()[1])
                if p == me:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                    break
                if p <= 1:
                    break
        except (OSError, ValueError):
            continue
    return killed


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
    def __init__(self, run=None, wandb=None, reason='', mode=None):
        self.run, self._wandb, self.reason, self.mode = run, wandb, reason, mode
        self._t_last, self._s_last = time.time(), 0
        self._q, self._th, self.dropped = None, None, 0
        if run is not None:
            self._q = queue.Queue(maxsize=LOG_QUEUE_MAX)
            self._th = threading.Thread(target=self._worker, name='wandb-log', daemon=True)
            self._th.start()

    @property
    def active(self):
        return self.run is not None

    def _fail(self, where, exc):
        if self.run is not None:
            print(f'[wandb] disabled after an error in {where}: {type(exc).__name__}: {exc} '
                  f'-- training continues, metrics.csv unaffected', flush=True)
        self.run = None

    def _worker(self):
        """The only place rows reach wandb. Runs off the training thread; exits on the None sentinel."""
        while True:
            item = self._q.get()
            if item is None:
                return
            run = self.run
            if run is None:
                continue                                          # disabled: just drain
            try:
                run.log(item[0], step=item[1])
            except Exception as e:
                self._fail('log', e)

    def _enqueue(self, row, step):
        if self.run is None or self._q is None:
            return
        try:
            self._q.put_nowait((row, step))
        except queue.Full:
            self.dropped += 1                                      # never wait on the tracker

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
            if mode != 'offline':
                if _reachable(base):
                    mode = 'online'
                else:
                    print(f'[wandb] {base} unreachable (2 s probe) -> offline', flush=True)
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
                             mode=mode, settings=wandb.Settings(console='off', **BOUNDS))
            print(f'[wandb] {mode}: run {name} -> {run.dir}'
                  + ('  (sync later: wandb sync ' + os.path.dirname(run.dir) + ')' if mode == 'offline' else ''),
                  flush=True)
            return cls(run, wandb, mode=mode)
        except Exception as e:
            print(f'[wandb] off: init failed: {type(e).__name__}: {e} -- training continues', flush=True)
            return cls(reason='init failed')

    def train_step(self, step, loss, ema, lr):
        if self.run is None or not (step % LOG_EVERY == 0 or step == 1):
            return
        now = time.time()
        dt = (now - self._t_last) / max(step - self._s_last, 1)
        self._t_last, self._s_last = now, step
        self._enqueue({'train/loss': loss, 'train/loss_ema': ema, 'train/lr': lr, 'time/sec_per_step': dt}, step)

    def eval_step(self, step, bpb, ema, extra=None, model=None):
        if self.run is None:
            return
        try:
            row = {'val_bpb': bpb, 'train_loss': ema}
            row.update(extra or {})
            if model is not None:
                row.update(learned_confidence_by_layer(model))
        except Exception as e:
            self._fail('eval_step', e)
            return
        self._enqueue(row, step)

    def finish(self, summary=None):
        run = self.run
        if run is None:
            return
        try:
            d = os.path.dirname(run.dir)
        except Exception:
            d = '?'
        done, errs, t0 = threading.Event(), [], time.time()

        def _finish():
            try:
                try:
                    self._q.put(None, timeout=5)                   # let queued rows go out (bounded)
                except queue.Full:
                    pass
                self._th.join(timeout=20)
                for k, v in (summary or {}).items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        run.summary[k] = v
                self._wandb.finish()                              # bounded by BOUNDS['finish_timeout']
            except Exception as e:
                errs.append(e)
            finally:
                done.set()

        threading.Thread(target=_finish, name='wandb-finish', daemon=True).start()
        deadline = FINISH_TIMEOUT + 45
        extra = f'; {self.dropped} log rows dropped' if self.dropped else ''
        if not done.wait(deadline):
            killed = _kill_own_wandb_helpers()
            print(f'[wandb] finish() gave up after {deadline:.0f}s (killed wandb helpers {killed}){extra}; '
                  f'local record intact: wandb sync {d}', flush=True)
        elif errs:
            print(f'[wandb] finish() error: {type(errs[0]).__name__}: {errs[0]}{extra}; local record: {d}', flush=True)
        elif self.mode == 'online' and time.time() - t0 >= FINISH_TIMEOUT - 1:
            print(f'[wandb] finish() hit its {FINISH_TIMEOUT:.0f}s timeout -- upload likely incomplete{extra}; '
                  f'local record intact: wandb sync {d}', flush=True)
        else:
            print(f'[wandb] finished in {time.time() - t0:.1f}s{extra}'
                  + (f' (offline; sync with: wandb sync {d})' if self.mode == 'offline' else ''), flush=True)
        self.run = None
