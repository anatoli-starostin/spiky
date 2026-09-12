"""Optional, NON-FATAL wandb tracking for the ffn_replacement trainers (conventions: claude/wandb.md on main).

    from wandb_tracking import Tracker
    tracker = Tracker.start(cfg, EXP_DIR, grad_accum=..., total_params=...)   # after the model is built
    tracker.train_step(step, loss, ema, lr, grad_norm=None)  # once per optimiser step (logged every LOG_EVERY)
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
  * The descriptions work below is best-effort on top of that: a failure prints one line and leaves the run
    exactly as it was initialised.
  * It reads floats the trainer already has and, at evals, the learned confidence scalars via their
    read-only accessor: no RNG, no graph, no optimiser or data interaction.
  * No secrets: auth from ~/.netrc (`wandb login`); server URL / entity from WANDB_BASE_URL / WANDB_ENTITY.

ORGANISATION (claude/wandb.md section 4): project "Spiky"; group = the experiments/<family>/ folder,
"ffn_replacement"; job_type "train"; name and id = exp_name (unique per run folder, so a crashed
run resumes instead of duplicating); tags = family, branch, short commit, host, confidence form;
config = branch, commit, host + the full config.json (minus _arch_note and description, which go into the notes;
the legacy eval_steps is sent as eval_steps_legacy_ignored) + derived batch sizes. host has a 'pasta-' prefix
removed, so runs launched through `sbox --net tailnet` share the machine's name.

DESCRIPTIONS (tools/metric_glossary.py holds what every key measures):
  * notes = run_notes(): bold exp_name, the config `description` (else the opening sentences of _arch_note),
    links to the run folder on GitHub at the launch commit and at the branch head, the folder and host, a pointer
    to the "About these metrics" panel at the top of the project workspace and this run's glossary artifact, then
    the full _arch_note.
  * a `metric_glossary` artifact (Table key | description | unit, alias glossary-<hash>) via log_artifact ONLY,
    never into run history (a Table in history renders as a broken panel on the self-hosted server).
    Identical glossaries dedup to one artifact version.
  * drift check: the first time a logged key has no glossary entry, one line is printed, the run gets the tag
    glossary:undocumented and summary glossary/undocumented lists the keys. Never fatal.
"""
import os
import queue
import re
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

try:
    import metric_glossary as MG
except Exception:                                                # pragma: no cover
    MG = None

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

# the keys this tracker emits itself (test_metric_glossary.py checks each has a glossary entry)
TRAIN_KEYS = ('train/loss', 'train/loss_ema', 'train/lr', 'time/sec_per_step', 'train/grad_norm')
EVAL_KEYS = ('val_bpb', 'train_loss')
SUMMARY_KEYS = ('exp_name', 'best_val_bpb', 'final_val_bpb', 'total_params', 'training_time_hours',
                'glossary/undocumented')
UNDOC_TAG = 'glossary:undocumented'
UNDOC_SUMMARY = 'glossary/undocumented'
GLOSSARY_ARTIFACT, GLOSSARY_ARTIFACT_TYPE = 'metric_glossary', 'glossary'
NOTES_CONFIG_KEYS = ('_arch_note', 'description')               # go into the notes, not the config


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


# ---- run description helpers (pure; used by Tracker.start and wandb_backfill.py) ------------------------------
def normalise_host(name):
    """'pasta-gpustar' (the hostname inside `sbox --net tailnet`'s network namespace) -> 'gpustar'."""
    return name[len('pasta-'):] if isinstance(name, str) and name.startswith('pasta-') else name


_SSH_REMOTE = re.compile(r'^(?:ssh://)?[^@/\s]+@([^:/\s]+)[:/](.+?)(?:\.git)?/?$')
_HTTP_REMOTE = re.compile(r'^https?://(?:[^@/\s]+@)?([^/\s]+)/(.+?)(?:\.git)?/?$')


def github_web_url(remote):
    """https://github.com/<owner>/<repo> for a GitHub remote -- including ssh host aliases such as
    git@github-<alias>:<owner>/<repo>.git, where the alias only selects a key in ~/.ssh/config -- else None."""
    for rx in (_SSH_REMOTE, _HTTP_REMOTE):
        m = rx.match((remote or '').strip())
        if m:
            host, path = m.group(1).split(':')[0], m.group(2)
            ok = (host == 'github.com' or host.startswith('github')) and path.count('/') == 1
            return f'https://github.com/{path}' if ok else None
    return None


def git_launch_info(exp_dir):
    """Where the run's code lives, from the checkout this file is in. Keys: root, sha (full), branch, dirty
    (tracked files only), rel (run folder relative to the repo root, None if outside it), committed (the folder
    exists in the tree at sha; None if unknown), web (GitHub URL or None). Any of them may be None."""
    info = dict(root=None, sha=None, branch=None, dirty=None, rel=None, committed=None, web=None)
    root = _git(['rev-parse', '--show-toplevel'], os.path.dirname(os.path.abspath(__file__)))
    if root in ('', 'unknown'):
        return info
    sha = _git(['rev-parse', 'HEAD'], root)
    branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'], root)
    dirty = _git(['status', '--porcelain', '--untracked-files=no'], root)
    rel = os.path.relpath(os.path.abspath(exp_dir), root)
    info.update(root=root, sha=None if sha == 'unknown' else sha,
                branch=None if branch in ('unknown', 'HEAD') else branch,
                dirty=None if dirty == 'unknown' else bool(dirty),
                rel=None if rel.startswith('..') else rel,
                web=github_web_url(_git(['remote', 'get-url', 'origin'], root)))
    return with_committed_flag(info)


def with_committed_flag(info):
    """Set info['committed'] from `git ls-tree <sha> -- <rel>` (False = the folder is not in that commit)."""
    if info.get('root') and info.get('sha') and info.get('rel'):
        out = _git(['ls-tree', info['sha'], '--', info['rel']], info['root'])
        info['committed'] = None if out == 'unknown' else bool(out)
    return info


_MD_SPECIAL = re.compile(r'([\\`*_\[\]|])')           # not < > #: wandb can show those escapes literally
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\["`])')


def md_escape(text):
    """Escape markdown syntax so free text (config notes) renders literally."""
    return _MD_SPECIAL.sub(r'\\\1', text or '')


def description_text(cfg, override=None, max_chars=500):
    """The run's short description: `override`, else config `description`, else the opening sentences of the first
    paragraph of _arch_note -- at least one, at most four, stopping before max_chars."""
    d = (override or cfg.get('description') or '').strip()
    if d:
        return d
    para = (cfg.get('_arch_note') or '').strip().split('\n\n')[0].strip()
    out = []
    for s in _SENTENCE_END.split(para):
        if out and (len(out) == 4 or len(' '.join(out + [s])) > max_chars):
            break
        out.append(s)
    text = ' '.join(out)
    if len(text) > 2 * max_chars:
        text = text[:2 * max_chars].rsplit(' ', 1)[0] + ' ...'
    return text


def workspace_url(base, entity, project):
    """The project workspace, where the "About these metrics" panel sits at the top (None without base/entity)."""
    return f'{base.rstrip("/")}/{entity}/{project}/workspace' if base and entity and project else None


def run_notes(cfg, *, exp_name, info, host, workspace_url=None, artifact_url=None, artifact_label=None,
              description=None, code_label='Code at launch', extra_lines=()):
    """Markdown notes for a run (rendered in the run's Overview tab)."""
    lines = [f'**{md_escape(exp_name)}**', '', md_escape(description_text(cfg, description)), '']
    web, sha, rel, branch = info.get('web'), info.get('sha'), info.get('rel'), info.get('branch')
    flags = []
    if info.get('dirty'):
        flags.append('⚠ dirty: tracked files differed from this commit at launch')
    if info.get('committed') is False:
        flags.append('⚠ folder not committed at launch: this link will 404')
    flag_txt = (' — ' + '; '.join(flags)) if flags else ''
    folder = os.path.basename(rel) if rel else None
    if web and sha and rel:
        lines.append(f'- **{code_label}:** [{md_escape(folder)} @ {sha[:8]}]({web}/tree/{sha}/{rel}){flag_txt}')
    elif sha:
        lines.append(f'- **{code_label}:** commit `{sha}`{flag_txt}')
    if web and branch and rel:
        lines.append(f'- **Artefacts (branch head):** [{md_escape(branch)}]({web}/tree/{branch}/{rel})')
    if rel:
        lines.append(f'- **Run folder:** `{rel}` on `{host}`')
    glossary = []
    if workspace_url:
        glossary.append(f'"About these metrics" at the top of the [project workspace]({workspace_url})')
    if artifact_label:                                   # wandb renders no link whose text is `code`: plain text
        glossary.append(f'artifact [{md_escape(artifact_label)}]({artifact_url})' if artifact_url
                        else f'artifact `{artifact_label}`')
    if glossary:
        lines.append('- **Metric glossary:** ' + ' · '.join(glossary))
    lines += list(extra_lines)
    note = (cfg.get('_arch_note') or '').strip()
    if note:
        lines += ['', '---', '', '**Architecture note** (config `_arch_note`):', '', md_escape(note)]
    return '\n'.join(lines)


def wandb_config(cfg, **extra):
    """The config sent to wandb: config.json minus the notes keys, legacy eval_steps renamed, plus `extra`."""
    config = {k: v for k, v in cfg.items() if k not in NOTES_CONFIG_KEYS}
    if 'eval_steps' in config:
        config['eval_steps_legacy_ignored'] = config.pop('eval_steps')
    config.update(extra)
    return config


def gql(api, query, variables, timeout=None):
    """Raw GraphQL through the public API's wandb-core connection (wandb >= 0.28; no public helper exists)."""
    return api._service_api.execute_graphql(query, variables, timeout=timeout)


def log_glossary_artifact(run, wandb):
    """log_artifact ONLY (never run.log): Table key | description | unit, aliases latest + glossary-<hash>."""
    h = MG.glossary_hash()
    table = wandb.Table(columns=['key', 'description', 'unit'], data=MG.table_rows())
    art = wandb.Artifact(GLOSSARY_ARTIFACT, type=GLOSSARY_ARTIFACT_TYPE,
                         description=f'Metric glossary {h}: what every logged key measures (tools/metric_glossary.py).',
                         metadata={'glossary_hash': h, 'source': 'experiments/ffn_replacement/tools/metric_glossary.py'})
    art.add(table, 'glossary')
    run.log_artifact(art, aliases=['latest', f'glossary-{h}'])
    return f'glossary-{h}'


class Tracker:
    def __init__(self, run=None, wandb=None, reason='', mode=None):
        self.run, self._wandb, self.reason, self.mode = run, wandb, reason, mode
        self._t_last, self._s_last = time.time(), 0
        self._q, self._th, self.dropped = None, None, 0
        self._seen, self.undocumented = set(), []
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

    def _check_keys(self, keys):
        """Glossary drift check, never fatal: logged keys with no entry -> one line, tag, summary list."""
        new = [k for k in keys if k not in self._seen]
        if not new or MG is None:
            return
        self._seen.update(new)
        bad = MG.undocumented(new)
        if not bad:
            return
        self.undocumented.extend(bad)
        print(f'[wandb] glossary: no entry for {", ".join(bad)} -- add to tools/metric_glossary.py '
              f'(run tagged {UNDOC_TAG})', flush=True)
        run = self.run
        try:
            if run is not None:
                if UNDOC_TAG not in tuple(run.tags or ()):
                    run.tags = tuple(run.tags or ()) + (UNDOC_TAG,)
                run.summary[UNDOC_SUMMARY] = ', '.join(sorted(self.undocumented))
        except Exception as e:
            print(f'[wandb] glossary drift check could not tag the run: {type(e).__name__}: {e}', flush=True)

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
                self._check_keys(item[0].keys())
            except Exception:
                pass
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

    def _describe(self, cfg, name, info, host):
        """Glossary artifact + markdown notes, after init. Best-effort: a failure prints one line."""
        run, wandb = self.run, self._wandb
        base = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/')
        entity = os.environ.get('WANDB_ENTITY') or getattr(run, 'entity', None)
        project = getattr(run, 'project', None) or PROJECT
        label = art_url = None
        try:
            if MG is None:
                raise RuntimeError('tools/metric_glossary.py could not be imported')
            label = log_glossary_artifact(run, wandb)
            if base and entity:
                art_url = f'{base}/{entity}/{project}/artifacts/{GLOSSARY_ARTIFACT_TYPE}/{GLOSSARY_ARTIFACT}/{label}'
        except Exception as e:
            print(f'[wandb] glossary artifact not logged: {type(e).__name__}: {e}', flush=True)
        try:
            run.notes = run_notes(cfg, exp_name=name, info=info, host=host,
                                  workspace_url=workspace_url(base, entity, project),
                                  artifact_url=art_url, artifact_label=label and f'{GLOSSARY_ARTIFACT}:{label}')
        except Exception as e:
            print(f'[wandb] markdown notes not set ({type(e).__name__}: {e}); notes stay the _arch_note', flush=True)

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
            host = normalise_host(socket.gethostname())
            name = cfg.get('exp_name') or os.path.basename(os.path.abspath(exp_dir))
            dbs, seq = cfg.get('device_batch_size'), cfg.get('seq_len')
            config = wandb_config(cfg, branch=branch, commit=commit, commit_dirty=bool(dirty and dirty != 'unknown'),
                                  host=host, grad_accum=grad_accum, total_params=total_params,
                                  batch_rows_per_step=(dbs * grad_accum if dbs and grad_accum else None),
                                  tokens_per_step=(dbs * grad_accum * seq if dbs and grad_accum and seq else None))
            tags = [GROUP, branch, commit, host, f"form:{cfg.get('lut_confidence_form', 'margin')}"] + list(extra_tags)
            if cfg.get('lut_learned_margin_freeze_g'):
                tags.append('learned_margin_freeze_g')
            run = wandb.init(project=PROJECT, entity=os.environ.get('WANDB_ENTITY'), group=GROUP, job_type=job_type,
                             name=name, id=name, resume='allow', tags=[t for t in tags if t and t != 'unknown'],
                             notes=(cfg.get('_arch_note') or None), config=config, dir=run_dir,
                             mode=mode, settings=wandb.Settings(console='off', **BOUNDS))
            print(f'[wandb] {mode}: run {name} -> {run.dir}'
                  + ('  (sync later: wandb sync ' + os.path.dirname(run.dir) + ')' if mode == 'offline' else ''),
                  flush=True)
            tracker = cls(run, wandb, mode=mode)
        except Exception as e:
            print(f'[wandb] off: init failed: {type(e).__name__}: {e} -- training continues', flush=True)
            return cls(reason='init failed')
        try:
            tracker._describe(cfg, name, git_launch_info(exp_dir), host)
        except Exception as e:
            print(f'[wandb] run description skipped: {type(e).__name__}: {e}', flush=True)
        return tracker

    def train_step(self, step, loss, ema, lr, grad_norm=None):
        if self.run is None or not (step % LOG_EVERY == 0 or step == 1):
            return
        now = time.time()
        dt = (now - self._t_last) / max(step - self._s_last, 1)
        self._t_last, self._s_last = now, step
        row = {'train/loss': loss, 'train/loss_ema': ema, 'train/lr': lr, 'time/sec_per_step': dt}
        if grad_norm is not None:
            try:
                row['train/grad_norm'] = float(grad_norm)             # a tensor: one device sync per logged step
            except Exception:
                pass
        self._enqueue(row, step)

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
                scalars = {k: v for k, v in (summary or {}).items()
                           if isinstance(v, (int, float, str, bool)) or v is None}
                try:
                    self._check_keys(scalars.keys())
                except Exception:
                    pass
                for k, v in scalars.items():
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
