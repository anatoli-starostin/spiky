# Experiment tracking with wandb (self-hosted)

How to log spiky experiments to a **self-hosted Weights & Biases** server and
keep runs organized across many branches, machines, and experiment families.
This is tooling/operational guidance; the science lives in the other `claude/`
docs.

> **Deployment-specific values stay OUT of this repo.** Your actual server URL,
> entity/account name, host names, and (above all) your API key are per-deployment
> and per-host — keep them in your own private/per-host notes, not here (consistent
> with the `claude/README.md` rule that this folder is kept free of machine- and
> account-specific details). Below, `<...>` and `$WANDB_BASE_URL` are placeholders
> you fill in for your own deployment.

> **Never commit secrets.** The wandb API key is a password. It must never be in
> this repo — this doc explains how to obtain it, never its value.

## 1. The server

- Run a **self-hosted W&B server** (the `wandb/local` container) somewhere always-on
  and reach it in a browser at your server URL, e.g. `http://<your-wandb-host>:<port>`.
  Export that as `WANDB_BASE_URL` so clients target it instead of wandb.ai.
- Pick a **single entity** (account/namespace), `<YOUR_ENTITY>`, so everyone shares
  one view of runs and projects.
- Keep it private to your network (e.g. a tailnet/VPN) rather than the public
  internet.

## 2. One-time client setup on a training host

Do this once per machine (GPU box), in the environment your training runs in:

```sh
pip install wandb                                  # into the experiment's env
export WANDB_BASE_URL=<your-wandb-server-url>       # e.g. http://<your-wandb-host>:<port>
wandb login --host "$WANDB_BASE_URL" <YOUR_API_KEY>
```

- **Get `<YOUR_API_KEY>` from the browser**: open `<WANDB_BASE_URL>/authorize`
  while logged in as `<YOUR_ENTITY>` and copy the key. Treat it like a password:
  never hardcode it, never commit it, never paste it into code or config. After
  `wandb login` it is stored in `~/.netrc` and used automatically.
- **Gotcha — pass the key as a positional arg** to `wandb login` (as above). On
  some client versions, setting only the `WANDB_API_KEY` env var was not enough
  ("No API key configured"); the positional form is reliable.
- **Gotcha — `localhost` vs IPv4 on the server host.** If your server host
  resolves `localhost` to IPv6 (`::1`) but the server only forwards IPv4, use
  `http://127.0.0.1:<port>` when talking to it *from the server box itself*.
  Remote clients using the host's name/IP are unaffected.
- Keep `export WANDB_BASE_URL=...` in the run environment (e.g. your launch
  script) so every run targets your server rather than wandb.ai.

## 3. Minimal integration in a train.py

```python
import wandb, subprocess

sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
run = wandb.init(
    project="Spiky",                      # your project name (one project for all — see below)
    group="hyperplane_ffn",               # experiment family / folder
    job_type="train",                     # train | eval | sweep
    name="exp006_nap6_tph256",            # concise, unique run name
    tags=["hyperplane_ffn", "lut", sha],  # flexible labels (branch, machine, ...)
    notes=notes_markdown,                 # what the run tests, its code, every metric it logs (section 5)
    config=dict(branch="<git-branch>", commit=sha, host="<this-host>",
                lr=lr, batch=batch, seq_len=seq_len, n_tables=tph, bits=nap,
                d_model=n_embd, depth=depth),  # the reproducibility record
)
for step in range(1, n_steps + 1):
    ...
    wandb.log({"train_loss": loss, "val_bpb": bpb,
               "grad_norm": gnorm, "lr": cur_lr}, step=step)
wandb.finish()   # always finish (or use `with wandb.init(...) as run:`)
```

- wandb **auto-captures system metrics** (GPU util/mem, CPU, disk) and
  **auto-plots every scalar** you log — no chart setup needed. Histograms,
  images, tables, etc. are logged only if you explicitly call them.
- Log a consistent `step=`; make `val_bpb` (our headline metric) one of the
  scalars so runs are directly comparable.

### In this repo: `spiky.util.wandb_integration`

Trainers here should not hand-roll the above: the shared package on `main`,
`src/spiky/util/wandb_integration/`, does it with the conventions of sections 4–5 built in.

- `tracker.py` — `Tracker`: optional, never-fatal logging from a training loop. It writes the run's notes
  (section 5) and checks every logged key against the metric legend.
- `glossary.py` — `DictGlossary`: the metric legend as data in code.
- `backfill.py` — upload a finished run from its `metrics.csv` + `config.json` (`backfill_run`), and
  notes-only rewrites of runs already on the server (`update_notes`), e.g. to give existing runs their legend.
- `tests/` — CPU tests, no server.

**No install step.** It imports from an existing editable install (`pip install -e .`): the editable
finder maps `spiky.util` and resolves its immediate children, so no `setup.py` entry and no reinstall
are needed. Like the rest of `src/spiky`, it has no `__init__.py`. `wandb` itself is not in
`requirements.txt`; without it the tracker switches itself off.

```python
from spiky.util.wandb_integration.glossary import DictGlossary
from spiky.util.wandb_integration.tracker import Tracker

GLOSSARY = DictGlossary({                                                  # what every logged key measures
    "train/loss":    dict(unit="nats/tok", section="train", desc="Cross-entropy of the step, ..."),
    "val_bpb":       dict(unit="bits/byte", section="eval", desc="Bits per byte on the validation window, ..."),
    "ln2_norm_L{i}": dict(unit="L2", section="per layer", desc="L2 norm of block i's ln2 gain."),  # {i}: any layer
}, sections={"train": "Training", "eval": "Evaluation", "per layer": "Per layer"}, source="<path of this file>")

tracker = Tracker.start(cfg, exp_dir, project="Spiky", group="<family>",  # default WANDB_PROJECT / WANDB_RUN_GROUP
                        tags=["<static tag>"],
                        extra_tags=lambda cfg: [f"form:{cfg['form']}"],      # more tags from the config
                        extra_eval_metrics=lambda model: {},                 # merged into eval rows that pass a model
                        glossary=GLOSSARY,                                   # its legend goes into the notes
                        config_extra={"total_params": n_params},
                        config_renames={"old_key": "old_key_legacy"})
tracker.train_step(step, {"train/loss": loss, "train/lr": lr})  # logged at step 1 and every log_every (10)
tracker.eval_step(step, {"val_bpb": bpb}, model)                 # + extra_eval_metrics(model)
tracker.finish(summary)                                          # after the local record is written
```

- **Everything project-specific is injected:** project, entity and group (defaults `WANDB_PROJECT`,
  `WANDB_ENTITY`, `WANDB_RUN_GROUP`), tags, the logged keys (rows are plain dicts; the tracker adds only
  `time/sec_per_step`), per-eval model metrics, config renames and extras, and the glossary. With
  `WANDB_BASE_URL` unset or no project, the tracker is off.
- **It never stalls or breaks training:** rows go onto a bounded queue that a background thread logs (a
  full queue drops rows, counted, and never waits); a 2 s probe of the server before `wandb.init` falls
  back to offline (`wandb sync` later); every network path has capped retries; `finish()` gives up after
  a deadline (60 s plus a grace period); any tracker error disables it with one printed line.
- **The notes are the whole mechanism.** At start the tracker writes one markdown blob into the run's notes:
  the description (config `description`, else the opening sentences of `_arch_note`), the run folder on GitHub
  at the launch commit and at the branch head, folder + host, then `GLOSSARY.legend_markdown()` (every metric
  the run logs, one table per section: key | what it measures [unit]), then the full `_arch_note`. Nothing else
  describes the run: no panels, saved views, artifacts or audit command.
- **`DictGlossary` is the legend as data.** Per key: `unit` and `desc` (the definition, written from the code
  that computes the value), optional `section`; a key containing `{i}` describes that key for any layer index.
  Keep units short (`nats/tok`, `bits/byte`, `L2`, `s`): they are printed in brackets after the description.
  Optional `sections` sets the legend's order and headings (sections left out still count as described, e.g.
  columns that exist only in a local `metrics.csv`), `notes` adds a bullet list at the end, `source` names the
  file. **Escape hatch:** any object with `undocumented(keys)` and `legend_markdown()` works, for what data
  cannot express.
- **Enforcement: a loud warning, never a failure.** In-process, no API calls, and also when tracking is off:
  the first time a row carries a key the legend does not describe, the tracker prints
  `[wandb] UNDESCRIBED METRIC: <key> ...` (at the first logged step, so a smoke run shows it within seconds);
  `finish()` prints a banner listing every undescribed key the run logged, summary keys included, and
  `tracker.undocumented` holds them. A run started without a glossary says so at start. It warns rather than
  fails because the cost is asymmetric: a killed or refused run loses GPU hours that cannot be recovered, while a
  missing description is fixed afterwards with a notes-only rewrite (`backfill.update_notes`). Failing would
  also break the never-fatal guarantee below.
- **Optional means optional — the import guard.** Once the package is imported nothing in it raises:
  `Tracker.start` never raises and returns an inactive tracker when tracking is off, and `NullTracker` is the
  same do-nothing surface for code that starts no run (DDP ranks other than 0, dry runs, tests). What no code in
  the package can cover is the package itself failing to import (absent — e.g. an editable install pointing at
  an older checkout — or broken). That guard is the consumer's, with a fallback built from builtins
  (`tracker.IMPORT_GUARD`); anything that imports the package — a `DictGlossary` module too — goes inside it:

  ```python
  try:
      from spiky.util.wandb_integration.tracker import Tracker
      from my_glossary import GLOSSARY  # a DictGlossary imports this package too: keep it inside the guard
      tracker = Tracker.start(cfg, exp_dir, project=PROJECT, group=GROUP, glossary=GLOSSARY)
  except Exception as e:  # the package itself could not be imported: Tracker.start never raises
      print(f'[wandb] off: {type(e).__name__}: {e} -- training continues')
      tracker = type('NoTracker', (), {'active': False, '__getattr__': lambda self, name: lambda *a, **k: None})()
  ```
- **Worked example — how a project wires it up:** `experiments/ffn_replacement/tools/` on
  `research/ffn_replacement_fix`. `wandb_tracking.py` is a thin shim: it pre-binds project and group,
  the LUT tags, `learned_confidence_by_layer` as `extra_eval_metrics`, `metric_glossary` and an
  `eval_steps` rename, and keeps the old positional calls (`train_step(step, loss, ema, lr, grad_norm)`,
  `eval_step(step, bpb, ema, extra, model)`) so frozen run folders run unchanged; if the package cannot be
  imported it is a no-op. `metric_glossary.py` is the legend content and `wandb_backfill.py` a thin wrapper
  for notes rewrites. When the package API changes, change the shim — never a frozen run folder.
- **Why `wandb_integration`, never `wandb`:** pytest's default import mode (prepend) puts
  `src/spiky/util` itself at `sys.path[0]` when it collects `util/test_utils.py`. A `util/wandb/` package
  with an `__init__.py` would then shadow the real library for the whole session (`import wandb` gets the
  folder); without one it happens to work, until someone adds it.

## 4. Organization conventions (the important part)

Everything goes into **one project**, and you slice it with the run fields:

| field      | use it for                                | examples                                  |
|------------|-------------------------------------------|-------------------------------------------|
| `project`  | **one project for all runs**              | `"Spiky"` (your project name)             |
| `group`    | experiment **family / folder**            | `"hyperplane_ffn"`, `"lutgpt"`, `"walker2d"` |
| `job_type` | the kind of run                           | `"train"`, `"eval"`, `"sweep"`            |
| `name`     | concise **unique** run name               | `"exp006_nap6_tph256"`, `"vanilla_baseline"` |
| `tags`     | flexible cross-cutting labels             | git branch, short commit, machine, dataset, key knob |
| `config`   | the **reproducibility record**            | branch, commit sha, host, all hyperparams |
| `notes`    | **what the run tests, where its code is, what every metric means** (markdown) | 2–4 sentences; links to the run folder at the launch commit and at the branch head; folder + host; the legend of every logged metric |

Rules of thumb:

- **One project, `group` is the folder.** Don't spin up a new project per idea —
  a new `group` (matching the `experiments/<family>/` folder) keeps everything
  comparable in one place, and the UI collapses runs by group.
- **`config` is reproducibility.** Always include `branch`, `commit` (short sha),
  and `host`/machine, plus every hyperparameter that defines the run. This is what
  lets you tell two runs apart six months later and re-launch one.
- **`tags` for the git branch and machine.** One idea = one branch (see
  [experiment-methodology.md](experiment-methodology.md)); tag the run with that
  branch and the host so cross-machine runs of the same family stay legible.
- **`name` is human-facing and unique** — the exp id plus a short variant tag.
- **`notes` say what the run is.** Markdown, shown in the run's Overview tab (see section 5).
- **Every logged key is described in the run's notes** (section 5). A metric's name is not its definition.

## 5. Describing runs and metrics

> **Binding rule: every run describes its experiment AND every metric it logs, in its notes.**
> One markdown blob in the run's `notes` field, written at run start. That is the entire
> mechanism: no glossary artifact, no workspace panel, no shared view, no publish step, no audit tool.

**Where.** `wandb.init(notes=...)` (or `run.notes = ...` after init) renders as full markdown in the
run's Overview tab — headings, bold/italic, code, lists, tables, clickable links — and a legend of
several dozen metrics is fine. The runs table shows only one truncated line of it: open the run.

**What the blob holds, in order:**

- the run name, then 2–4 sentences on what the run tests;
- a link to the run folder on GitHub **at the full launch commit**
  (`https://github.com/<owner>/<repo>/tree/<sha>/<run folder>`), flagged when the working
  tree was dirty or the folder was not yet committed at that sha (the link would 404);
- a link to the same folder at the branch head (where the artefacts land later);
- the folder path and host;
- **the metric legend:** every key the run logs (or a per-layer pattern such as `ln2_norm_L{i}`) →
  what it measures, **written from the code that computes it** (reduction, which tokens count,
  cadence, running mean vs instantaneous, the exact eval protocol, normalisation), with a short
  unit; grouped by section, plus notes for a key that means something different elsewhere;
- the longer design note, if any, below a rule.

Derive the GitHub URL from `git remote get-url origin` (an ssh host alias such as
`git@github-<alias>:<owner>/<repo>.git` maps to `https://github.com/<owner>/<repo>`), and the
server URL and entity from the environment — never hard-code either.

**Keep the legend as data in code** next to the trainer — a `DictGlossary` (section 3) — and add a
key's entry in the same change that starts logging it. A CPU test in the project should check every
`metrics.csv` column and every key the trainer emits against it (e.g. `metric_glossary.py` and
`test_metric_glossary.py` in `experiments/ffn_replacement/tools/` on `research/ffn_replacement_fix`).

**Enforcement is in-process, loud and never fatal** (section 3): an undescribed key is printed when it
is first logged and again in a banner at `finish()`. **Existing runs** get the blob through a
notes-only rewrite (`backfill.update_notes`, GraphQL `upsertBucket(id, notes)`), which re-sends
nothing else.

Tried and rejected: a separate glossary **report** (too far from the runs); **panel legend
templates** (section 6: they cannot hold a description next to long run names); and a Markdown
panel published into a shared saved view, a per-run glossary artifact, and an audit CLI. That last
set was built and then dropped: it gave several copies to keep in sync, workspace state that is per
user and overwritten by open tabs, and nothing a reader of one run cannot get from its notes.

## 6. Gotchas (self-hosted server)

**Workspaces**

- **Workspace state is per user.** Opening a project creates (on the first visit) a personal
  workspace for that user: a view of type `"project-view"` named `nw-nwuser<username>-w`
  ("<User>'s workspace"). Other users get their own and never see yours; shared *saved views*
  ("Save as new view") are a separate thing. Anything written into a workspace spec — panels,
  sections, legends — is visible only to that user.
- **A stale open tab silently overwrites programmatic workspace writes.** The web client saves
  the whole spec it holds (last write wins): a tab that loaded the workspace before your API
  write drops the change on its next UI action, with no error. Check the view's `updatedAt`
  before writing (recent = probably an open tab), re-read after writing, and verify again later.
  The client saves aggressively — even a panel-search query typed into the box is persisted, and
  merely opening a run page re-saves the personal workspace. This happened for real: a panel written
  into a personal workspace was gone eight minutes later, removed by the user's own tab.
- **Panel legend templates are no place for descriptions.** Syntax: literal prose is kept;
  `${run:displayName}`, `${metricName}`, `${config:<key>}`, `${summary:<key>}`; the `[[ ... ]]`
  part (`${x}`, `${y}`, …) shows only on hover. But each run is **one legend line, and the whole
  line is middle-truncated** to the panel width — about 50 characters *including the run name*
  in a default panel, so with long run names nothing of a description survives; the prose
  **repeats on every run's line**; the **hover tooltip shows less, not more** (it truncates
  harder); single-run charts show no legend at all. Chart titles are one truncated line too.

**Runs, metrics, reports**

- **`define_metric` has no description.** Its parameters are `step_metric`, `step_sync`,
  `hidden`, `summary`, `goal`, `overwrite`; passing `description=` raises `TypeError`.
- **No panel or section descriptions** in the workspace UI. Sections are auto-named by
  key prefix (`train/…`, `time/…`); unprefixed keys land in "Charts".
- **Don't log a `wandb.Table` into run history.** On some self-hosted versions the
  auto-created workspace "Tables" panel errors ("Oops, something went wrong"). Put the table
  in an artifact via `log_artifact` only; it renders under Artifacts → the artifact → Files.
- **A mid-run disconnect loses rows server-side.** In an online run, rows logged after the
  connection to the server drops are lost on the server, while the trainer's local `metrics.csv`
  stays complete. Treat the local file as the record, and backfill from it if the server copy matters.
- **Auto git capture renders as commands, not links.** The run Overview shows
  `git clone <remote>` and `git checkout -b <run> <sha>`, with an ssh host alias shown verbatim.
  Overriding `settings.git_remote_url` records the remote but **drops the commit**. Put
  clickable links in the notes instead. Runs uploaded after the fact record the
  uploader's HEAD, not the run's commit — say so in their notes.
- **Editing notes after the fact.** The public `Run.update()` re-sends config, tags and
  summary. For a notes-only edit use GraphQL `upsertBucket(input: {id, notes})` with the run's
  storage id (`backfill.set_notes` / `update_notes`); everything else stays untouched. The public
  `wandb.Api()` caches `Run` objects, so verify with a fresh query.
- **Reports, if you use them.** Without `wandb-workspaces` a report can be created with raw
  GraphQL `upsertView` (`type: "runs"`, a JSON `spec`), found by title and updated by `id`.
  `type` decides listing: `"runs"` = published (project Reports tab, Home), `"runs/draft"` = a
  private draft, a draft with `parentId` = an unsaved edit; the profile page's Reports card
  shows only *showcased* reports. Deleting a report leaves dead links wherever it was linked.
- **The client may be newer than the server's advertised maximum**
  (`serverInfo.cliVersionInfo.max_cli_version`). It can work, but check this first when
  something fails oddly after a client upgrade.
- **Host inside a network namespace.** A sandbox that gives the process its own network
  namespace can change `socket.gethostname()` (e.g. add a prefix), which splits host tags
  between sandboxed and plain runs; normalise the name before logging it.

## 7. Viewing and comparing

- Open `<WANDB_BASE_URL>/<YOUR_ENTITY>/<project>` and use the **group** selector to
  collapse families; the runs **table** + **parallel-coordinates**/scatter panels
  compare hyperparams vs. metrics; **filter by `tags`** (branch, machine).
- If a run seems "missing", check you're logged into the browser as `<YOUR_ENTITY>`
  and looking at the right project (a run's project is set by `project=`).

## 8. Do / don't

- **Do** keep the API key out of the repo (it lives only in `~/.netrc`); use one
  project; always `wandb.finish()`; log `val_bpb`; describe the experiment and every logged
  metric in the run notes, adding the legend entry in the same change that starts logging a key.
- **Don't** add other places that describe runs or metrics (panels, views, artifacts): the notes are
  the one place.
- **Do** log from trainers through `spiky.util.wandb_integration` (section 3) rather than a raw
  `wandb.init`, so the conventions and the never-fatal behaviour come with it.
- **Don't** hardcode the key, create per-experiment projects, or point runs at
  `wandb.ai` (always set `WANDB_BASE_URL`).
- Prefer resuming a crashed run with its id (`wandb.init(id=..., resume="allow")`)
  over creating a duplicate.

Related: a TensorBoard viewer is a lighter-weight alternative; wandb is the
primary shared dashboard because it aggregates many runs/branches into one
comparable project.
