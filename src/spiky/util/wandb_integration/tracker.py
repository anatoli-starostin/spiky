"""Optional, NON-FATAL wandb tracking for training loops (conventions: claude/wandb.md).

    from spiky.util.wandb_integration.tracker import Tracker
    tracker = Tracker.start(cfg, exp_dir, project=..., group=..., glossary=None,          # after the model is built
                            extra_tags=None, extra_eval_metrics=None, config_extra=None, config_renames=None,
                            glossary_panel=None)
    tracker.train_step(step, {'train/loss': loss, 'train/lr': lr})   # once per optimiser step (logged every log_every)
    tracker.eval_step(step, {'val/loss': val}, model)                 # at each eval, after the local record is written
    tracker.finish(summary)                                           # at the end, after summary/checkpoint are written

WHAT IS INJECTED -- the tracker holds no project knowledge:
  * project / entity / group: arguments, defaulting to WANDB_PROJECT / WANDB_ENTITY / WANDB_RUN_GROUP. No project
    -> tracker OFF.
  * which keys are logged: the caller's row dicts. The tracker adds only TIMING_KEY (time/sec_per_step) to train rows.
  * tags: `tags` plus extra_tags(cfg) -> iterable. Per-eval metrics read off the model: extra_eval_metrics(model) -> dict.
  * config: the run config minus NOTES_CONFIG_KEYS, with config_renames {old: new} applied, plus config_extra.
  * glossary (optional): a glossary.DictGlossary, or any object with the protocol in glossary.py. It drives the drift
    check and the per-run glossary artifact.
  * glossary_panel (optional, explicit): where the project's glossary panel is published, for a link in the notes --
    the title of the shared saved view (as `workspace publish --shared TITLE`), or PERSONAL_WORKSPACE. None: no link.
    It is independent of the glossary: a publishable glossary adds no link unless this says where the panel is.

MODE -- ONLINE BY DEFAULT when the server is reachable:
  * WANDB_BASE_URL unset -> tracker OFF (a run can never silently go to wandb.ai); WANDB_MODE=disabled -> OFF.
  * WANDB_MODE=offline   -> offline: written under $WANDB_DIR (default ~/.cache/wandb), uploaded later with
                            `wandb sync <dir>` (from the host, or `sbox --net tailnet -- wandb sync <dir>`).
  * otherwise            -> a 2 s TCP probe of the server: reachable -> ONLINE (live curves); unreachable ->
                            OFFLINE with one line saying so. A dead server costs ~2 s at start, never a hang
                            (a wandb.init against an unreachable server was measured to block for >11 min).
                            Bare `sbox` has no network, so runs launched there go offline automatically;
                            launch with `sbox --net tailnet -- ...` to log live.
  * `wandb` is imported lazily in start(); if it is not installed the tracker is OFF with one line.

GUARDS -- the tracker can never stall or break training:
  * The training loop never calls wandb. train_step/eval_step put a row on a bounded in-memory queue and
    return; a background thread does run.log(). If wandb stops accepting rows the queue fills and further
    rows are DROPPED (counted, reported at finish) -- the loop never waits.
  * Every wandb network path has capped retries/timeouts (BOUNDS), and finish() has a hard deadline
    (FINISH_TIMEOUT, default 60 s, env WANDB_TRACKER_FINISH_TIMEOUT, plus FINISH_GRACE): past it the tracker gives
    up on the upload, kills its own wandb-core helper processes and returns, so the process exits. Call finish()
    only after the run's local record (metrics file, summary, checkpoint) is written: it stays on disk for a later
    `wandb sync`.
  * Any tracker error disables the tracker for the rest of the run with ONE line. Nothing is raised into
    the training loop; the trainer's own metrics file is written exactly as without the tracker.
  * The descriptions work below is best-effort on top of that: a failure prints one line and leaves the run
    exactly as it was initialised.
  * No secrets: auth from ~/.netrc (`wandb login`); server URL / entity from WANDB_BASE_URL / WANDB_ENTITY.
  * KNOWN LIMIT: rows logged while the server is unreachable in the middle of an ONLINE run can be missing
    server-side after it comes back; the trainer's local metrics file stays complete.

OPTIONAL MEANS OPTIONAL -- what this package can and cannot guarantee:
  * Once imported, nothing here raises into the loop: Tracker.start never raises and returns an inactive tracker
    (active False, every call a no-op) when tracking is off for any reason, wandb missing included.
  * NullTracker is the same do-nothing surface for code paths that deliberately start no run (DDP ranks other than 0,
    dry runs, tests).
  * What NO code in this package can cover is the package itself failing to import (absent -- e.g. an editable
    install pointing at a checkout that predates it -- or broken): nothing of it runs then. That guard belongs to the
    consumer, and its fallback must be built from builtins. IMPORT_GUARD below is the pattern, as one block.

ORGANISATION (claude/wandb.md section 4): name and id = cfg exp_name, else the run folder name (unique per run folder,
so a crashed run resumes instead of duplicating); tags = group, branch, short commit, host + injected tags;
config = the run config (minus NOTES_CONFIG_KEYS, which go into the notes) + branch, commit, commit_dirty, host
+ config_extra. host has a 'pasta-' prefix removed, so runs launched through `sbox --net tailnet` share the machine's
name. branch / commit describe the checkout that holds the run folder (else the working directory's; code_dir overrides).

DESCRIPTIONS:
  * notes = run_notes(): bold exp_name, the config `description` (else the opening sentences of _arch_note),
    links to the run folder on GitHub at the launch commit and at the branch head (from `git remote get-url origin`),
    the folder and host; with a glossary, a pointer to this run's glossary artifact; with glossary_panel, a link to the
    published panel (the shared saved view, or the personal workspace); then the full _arch_note.
  * with a glossary: a `metric_glossary` artifact (Table key | description | unit, alias glossary-<hash>) via
    log_artifact ONLY, never into run history (a Table in history renders as a broken panel on the self-hosted
    server). Identical glossaries dedup to one artifact version.
  * with a glossary, the drift check: the first time a logged key has no glossary entry, one line is printed, the
    run gets the tag glossary:undocumented and summary glossary/undocumented lists the keys. Never fatal.
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

from spiky.util.wandb_integration import glossary as G

LOG_EVERY = 10
LOG_QUEUE_MAX = 4096
TIMING_KEY = 'time/sec_per_step'
FINISH_TIMEOUT = float(os.environ.get('WANDB_TRACKER_FINISH_TIMEOUT', '60'))
FINISH_GRACE = 45                                   # finish() waits FINISH_TIMEOUT + this before giving up
BOUNDS = dict(init_timeout=60, x_graphql_retry_max=5, x_graphql_timeout_seconds=20,
              x_graphql_retry_wait_min_seconds=2, x_graphql_retry_wait_max_seconds=10,
              x_file_stream_retry_max=15, x_file_stream_timeout_seconds=30,
              x_file_stream_retry_wait_min_seconds=2, x_file_stream_retry_wait_max_seconds=20,
              x_file_transfer_retry_max=5, x_file_transfer_timeout_seconds=60,
              finish_timeout=FINISH_TIMEOUT, finish_timeout_raises=False)

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


# ---- run description helpers (pure; used by Tracker.start and backfill.py) ------------------------------------
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


def code_root(exp_dir, code_dir=None):
    """The top of the git checkout that describes the run's code: code_dir's if given, else the one holding exp_dir,
    else the working directory's. None outside any checkout."""
    for d in ([code_dir] if code_dir else [exp_dir, os.getcwd()]):
        root = _git(['rev-parse', '--show-toplevel'], os.path.abspath(d)) if d and os.path.isdir(d) else 'unknown'
        if root not in ('', 'unknown'):
            return root
    return None


def git_launch_info(exp_dir, code_dir=None):
    """Where the run's code lives (see code_root). Keys: root, sha (full), branch, dirty (tracked files only), rel (run
    folder relative to the repo root, None if outside it), committed (the folder exists in the tree at sha; None if
    unknown), web (GitHub URL or None). Any of them may be None."""
    info = dict(root=None, sha=None, branch=None, dirty=None, rel=None, committed=None, web=None)
    root = code_root(exp_dir, code_dir)
    if root is None:
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
    """The project workspace, where a published glossary panel sits at the top (None without base/entity/project)."""
    return f'{base.rstrip("/")}/{entity}/{project}/workspace' if base and entity and project else None


class _PersonalWorkspace:
    def __repr__(self):
        return 'PERSONAL_WORKSPACE'


# glossary_panel value: the panel was published into each user's personal workspace (`publish` without --shared)
PERSONAL_WORKSPACE = _PersonalWorkspace()


def shared_nw_id(title):
    """The named-workspace id of a shared saved view: the title lower-cased, letters and digits only."""
    nwid = re.sub(r'[^a-z0-9]', '', (title or '').lower())
    if not nwid:
        raise ValueError(f'cannot derive a view id from {title!r}')
    return nwid[:40]


def shared_view_name(title):
    return f'nw-{shared_nw_id(title)}-v'


def saved_view_url(base, entity, project, title):
    """The shared saved view titled `title`, opened at <server>/<entity>/<project>?nw=<id> (None without base/entity/
    project)."""
    return f'{base.rstrip("/")}/{entity}/{project}?nw={shared_nw_id(title)}' if base and entity and project else None


def panel_link(base, entity, project, glossary_panel):
    """(url, link text) of the published glossary panel, for the run notes, or (None, None).
    glossary_panel: None -> no link; PERSONAL_WORKSPACE -> the project workspace; a string -> the shared saved view
    with that title (as `workspace publish --shared TITLE`). ValueError for anything else."""
    if glossary_panel is None:
        return None, None
    if glossary_panel is PERSONAL_WORKSPACE:
        url = workspace_url(base, entity, project)
        return (url, 'project workspace') if url else (None, None)
    if isinstance(glossary_panel, str) and glossary_panel.strip():
        url = saved_view_url(base, entity, project, glossary_panel)
        return (url, f'saved view "{md_escape(glossary_panel)}"') if url else (None, None)
    raise ValueError(f'glossary_panel must be None, PERSONAL_WORKSPACE or a saved view title, got {glossary_panel!r}')


def run_notes(cfg, *, exp_name, info, host, workspace_url=None, panel_title=None, artifact_url=None,
              artifact_label=None, description=None, code_label='Code at launch', extra_lines=(), panel_where=None):
    """Markdown notes for a run (rendered in the run's Overview tab). workspace_url + panel_title (+ panel_where, the
    link text, default "project workspace") make the glossary-panel link; see panel_link."""
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
    if workspace_url and panel_title:
        glossary.append(f'"{panel_title}" at the top of the [{panel_where or "project workspace"}]({workspace_url})')
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


def wandb_config(cfg, extra=None, renames=None):
    """The config sent to wandb: the run config minus NOTES_CONFIG_KEYS, keys renamed per `renames` {old: new}
    (e.g. a legacy key that must not be mistaken for its successor), plus `extra`."""
    config = {k: v for k, v in cfg.items() if k not in NOTES_CONFIG_KEYS}
    for old, new in (renames or {}).items():
        if old in config:
            config[new] = config.pop(old)
    config.update(extra or {})
    return config


def gql(api, query, variables, timeout=None):
    """Raw GraphQL through the public API's wandb-core connection (wandb >= 0.28; no public helper exists)."""
    return api._service_api.execute_graphql(query, variables, timeout=timeout)


def log_glossary_artifact(run, wandb, glossary):
    """log_artifact ONLY (never run.log): Table key | description | unit, aliases latest + glossary-<hash>."""
    h = glossary.glossary_hash()
    source = getattr(glossary, 'SOURCE', None)
    table = wandb.Table(columns=['key', 'description', 'unit'], data=glossary.table_rows())
    art = wandb.Artifact(GLOSSARY_ARTIFACT, type=GLOSSARY_ARTIFACT_TYPE,
                         description=f'Metric glossary {h}: what every logged key measures'
                                     + (f' ({source}).' if source else '.'),
                         metadata={'glossary_hash': h, 'source': source})
    art.add(table, 'glossary')
    run.log_artifact(art, aliases=['latest', f'glossary-{h}'])
    return f'glossary-{h}'


class Tracker:
    def __init__(self, run=None, wandb=None, reason='', mode=None, glossary=None, extra_eval_metrics=None,
                 log_every=LOG_EVERY, timing_key=TIMING_KEY):
        self.run, self._wandb, self.reason, self.mode = run, wandb, reason, mode
        self.glossary, self._extra_eval_metrics = glossary, extra_eval_metrics
        self.log_every, self.timing_key = max(int(log_every), 1), timing_key
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
                  f'-- training continues, local metrics unaffected', flush=True)
        self.run = None

    def _check_keys(self, keys):
        """Glossary drift check, never fatal: logged keys with no entry -> one line, tag, summary list."""
        new = [k for k in keys if k not in self._seen]
        if not new or self.glossary is None:
            return
        self._seen.update(new)
        bad = self.glossary.undocumented(new)
        if not bad:
            return
        self.undocumented.extend(bad)
        where = getattr(self.glossary, 'SOURCE', None) or 'the glossary'
        print(f'[wandb] glossary: no entry for {", ".join(bad)} -- add to {where} (run tagged {UNDOC_TAG})', flush=True)
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

    def _describe(self, cfg, name, info, host, project, entity, glossary_panel=None):
        """Glossary artifact + markdown notes, after init. Best-effort: a failure prints one line."""
        run, wandb, g = self.run, self._wandb, self.glossary
        base = (os.environ.get('WANDB_BASE_URL') or '').rstrip('/')
        entity = entity or getattr(run, 'entity', None)
        project = getattr(run, 'project', None) or project
        label = art_url = None
        if g is not None:
            try:
                label = log_glossary_artifact(run, wandb, g)
                if base and entity:
                    art_url = f'{base}/{entity}/{project}/artifacts/{GLOSSARY_ARTIFACT_TYPE}/{GLOSSARY_ARTIFACT}/{label}'
            except Exception as e:
                print(f'[wandb] glossary artifact not logged: {type(e).__name__}: {e}', flush=True)
        try:
            panel_url, where = panel_link(base, entity, project, glossary_panel)
        except Exception as e:
            print(f'[wandb] glossary_panel ignored ({type(e).__name__}: {e}); no panel link in the notes', flush=True)
            panel_url = where = None
        try:
            run.notes = run_notes(cfg, exp_name=name, info=info, host=host, workspace_url=panel_url,
                                  panel_title=(getattr(g, 'PANEL_TITLE', None) or G.DEFAULT_TITLE) if panel_url else None,
                                  panel_where=where,
                                  artifact_url=art_url, artifact_label=label and f'{GLOSSARY_ARTIFACT}:{label}')
        except Exception as e:
            print(f'[wandb] markdown notes not set ({type(e).__name__}: {e}); notes stay the _arch_note', flush=True)

    @classmethod
    def start(cls, cfg, exp_dir, *, project=None, entity=None, group=None, job_type='train', name=None, tags=(),
              extra_tags=None, extra_eval_metrics=None, glossary=None, config_extra=None, config_renames=None,
              code_dir=None, log_every=LOG_EVERY, timing_key=TIMING_KEY, glossary_panel=None):
        """Start a run, or return an inactive tracker (with one line saying why). Never raises.

        project / entity / group   default WANDB_PROJECT / WANDB_ENTITY / WANDB_RUN_GROUP; no project -> OFF
        name                       default cfg exp_name, else the run folder name; also the run id (resume='allow')
        tags                       static tags, after group / branch / commit / host
        extra_tags(cfg)            optional callable -> more tags (a failure prints one line; the run starts without them)
        extra_eval_metrics(model)  optional callable -> {key: number}, merged into eval rows that pass a model
        glossary                   optional: a glossary.DictGlossary or any object with the protocol in glossary.py
        glossary_panel             optional, explicit: the shared saved view's title, or PERSONAL_WORKSPACE, where the
                                   glossary panel is published -> a link in the notes. None (default): no link.
        config_extra / renames     merged into / renamed in the config (see wandb_config)
        code_dir                   the checkout that describes the code (see code_root)
        """
        base = os.environ.get('WANDB_BASE_URL')
        mode = os.environ.get('WANDB_MODE')
        if not base:
            print('[wandb] off: WANDB_BASE_URL not set', flush=True)
            return cls(reason='no WANDB_BASE_URL')
        if mode == 'disabled':
            print('[wandb] off: WANDB_MODE=disabled', flush=True)
            return cls(reason='disabled')
        project = project or os.environ.get('WANDB_PROJECT')
        if not project:
            print('[wandb] off: no project (pass project= or set WANDB_PROJECT)', flush=True)
            return cls(reason='no project')
        entity = entity or os.environ.get('WANDB_ENTITY')
        group = group or os.environ.get('WANDB_RUN_GROUP')
        if glossary is not None and G.missing(glossary, G.TRACKER_NEEDS):
            print(f'[wandb] glossary ignored: it lacks {", ".join(G.missing(glossary, G.TRACKER_NEEDS))}', flush=True)
            glossary = None
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
            repo = code_root(exp_dir, code_dir) or os.path.abspath(exp_dir)
            branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'], repo)
            commit = _git(['rev-parse', '--short', 'HEAD'], repo)
            dirty = _git(['status', '--porcelain', '--untracked-files=no'], repo)
            host = normalise_host(socket.gethostname())
            name = name or cfg.get('exp_name') or os.path.basename(os.path.abspath(exp_dir))
            config = wandb_config(cfg, dict(branch=branch, commit=commit,
                                            commit_dirty=bool(dirty and dirty != 'unknown'), host=host,
                                            **(config_extra or {})), renames=config_renames)
            all_tags = [group, branch, commit, host] + list(tags or ())
            if extra_tags is not None:
                try:
                    all_tags += list(extra_tags(cfg) or ())
                except Exception as e:
                    print(f'[wandb] extra_tags failed ({type(e).__name__}: {e}); starting without them', flush=True)
            run = wandb.init(project=project, entity=entity, group=group, job_type=job_type,
                             name=name, id=name, resume='allow', tags=[t for t in all_tags if t and t != 'unknown'],
                             notes=(cfg.get('_arch_note') or None), config=config, dir=run_dir,
                             mode=mode, settings=wandb.Settings(console='off', **BOUNDS))
            print(f'[wandb] {mode}: run {name} -> {run.dir}'
                  + ('  (sync later: wandb sync ' + os.path.dirname(run.dir) + ')' if mode == 'offline' else ''),
                  flush=True)
            tracker = cls(run, wandb, mode=mode, glossary=glossary, extra_eval_metrics=extra_eval_metrics,
                          log_every=log_every, timing_key=timing_key)
        except Exception as e:
            print(f'[wandb] off: init failed: {type(e).__name__}: {e} -- training continues', flush=True)
            return cls(reason='init failed')
        try:
            tracker._describe(cfg, name, git_launch_info(exp_dir, code_dir), host, project, entity, glossary_panel)
        except Exception as e:
            print(f'[wandb] run description skipped: {type(e).__name__}: {e}', flush=True)
        return tracker

    def train_step(self, step, row):
        """Call once per optimiser step; rows are logged at step 1 and every log_every-th step, other calls return at
        once. Values that are not plain numbers (e.g. a 0-d tensor) are converted with float() on logged steps only --
        one device sync per logged step; a value that cannot be converted is left out. Adds timing_key: wall seconds
        per step since the previous logged row."""
        if self.run is None or not (step % self.log_every == 0 or step == 1):
            return
        now = time.time()
        dt = (now - self._t_last) / max(step - self._s_last, 1)
        self._t_last, self._s_last = now, step
        out = {}
        for k, v in (row or {}).items():
            if isinstance(v, (int, float)):
                out[k] = v
                continue
            try:
                out[k] = float(v)
            except Exception:
                pass
        if self.timing_key:
            out[self.timing_key] = dt
        self._enqueue(out, step)

    def eval_step(self, step, row=None, model=None):
        """Log `row` at `step`, plus extra_eval_metrics(model) when a model is passed."""
        if self.run is None:
            return
        try:
            out = dict(row or {})
            if model is not None and self._extra_eval_metrics is not None:
                out.update(self._extra_eval_metrics(model))
        except Exception as e:
            self._fail('eval_step', e)
            return
        self._enqueue(out, step)

    def log(self, row, step):
        """Log any row at `step`, unthrottled (never blocks)."""
        if self.run is None:
            return
        self._enqueue(dict(row), step)

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
        deadline = FINISH_TIMEOUT + FINISH_GRACE
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


class NullTracker:
    """Tracks nothing: Tracker's public surface (active, mode, reason, dropped, undocumented, train_step, eval_step,
    log, finish), every call a no-op. For code paths that deliberately start no run -- DDP ranks other than 0, dry
    runs, tests. A consumer that can import this package never needs it for safety (Tracker.start never raises and
    returns an inactive tracker when tracking is off); for this package failing to import, see IMPORT_GUARD."""
    active, mode, dropped = False, None, 0

    def __init__(self, reason='not started'):
        self.reason, self.undocumented = reason, []

    def train_step(self, step, row=None):
        pass

    def eval_step(self, step, row=None, model=None):
        pass

    def log(self, row, step=None):
        pass

    def finish(self, summary=None):
        pass


# The consumer-side guard for the one failure no code in this package can absorb: this package failing to import
# (absent -- e.g. an editable install pointing at a checkout that predates it -- or broken). Tracker.start never raises,
# so in practice only an import reaches the except branch; the fallback is built from builtins because nothing of this
# package is importable there. It supports .active and calls (every method call is a no-op), nothing else. Anything
# that itself imports this package -- a DictGlossary module included -- must be imported inside the try as well.
IMPORT_GUARD = '''\
try:
    from spiky.util.wandb_integration.tracker import Tracker
    from my_glossary import GLOSSARY  # a DictGlossary imports this package too: keep it inside the guard
    tracker = Tracker.start(cfg, exp_dir, project=PROJECT, group=GROUP, glossary=GLOSSARY)
except Exception as e:  # the package itself could not be imported: Tracker.start never raises
    print(f'[wandb] off: {type(e).__name__}: {e} -- training continues')
    tracker = type('NoTracker', (), {'active': False, '__getattr__': lambda self, name: lambda *a, **k: None})()
'''
