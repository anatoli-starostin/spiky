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
    notes=notes_markdown,                 # what the run tests + links to its code (section 5)
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
| `notes`    | **what the run tests and where its code is** (markdown) | 2–4 sentences; links to the run folder at the launch commit and at the branch head; folder + host; the metric glossary |

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
- **Every logged key has a glossary entry** (section 5). A metric's name is not its definition.

## 5. Describing runs and metrics

**Run notes.** `wandb.init(notes=...)` (or `run.notes = ...` after init) renders as full
markdown in the run's Overview tab — headings, bold/italic, code, lists, tables,
clickable links — and several thousand characters are fine. Put in it:

- the run name, then 2–4 sentences on what the run tests;
- a link to the run folder on GitHub **at the full launch commit**
  (`https://github.com/<owner>/<repo>/tree/<sha>/<run folder>`), flagged when the working
  tree was dirty or the folder was not yet committed at that sha (the link would 404);
- a link to the same folder at the branch head (where the artefacts land later);
- the folder path and host, and a link to the metric glossary;
- the longer design note, if any, below a rule.

Derive the GitHub URL from `git remote get-url origin` (an ssh host alias such as
`git@github-<alias>:<owner>/<repo>.git` maps to `https://github.com/<owner>/<repo>`), and the
server URL and entity from the environment — never hard-code either.

**Metric glossary.** Keep one dict in code next to the tracker: logged key (or a per-layer
pattern such as `ln2_norm_L{i}`) → unit + a description **written from the code that
computes it** — reduction, which tokens count, cadence, running mean vs instantaneous,
the exact eval protocol, normalisation. Surface it three ways:

- a project **report titled "Metric glossary"** generated from the dict (and a short project
  README, the project description, linking it);
- a per-run **glossary artifact** (a `wandb.Table` of key | description | unit, logged with
  `log_artifact` only — see the gotchas); identical content dedups to one artifact version,
  so versions track glossary revisions;
- a link to both in every run's notes.

Make drift detectable, never fatal: the tracker flags a logged key with no entry (one
printed line, a tag, a summary field listing the keys); a read-only audit lists undocumented
keys on the server and stale entries; a CPU unit test checks every `metrics.csv` column and
every key the tracker emits. First implementation: `experiments/ffn_replacement/tools/`
(`metric_glossary.py`, `wandb_glossary.py`, `wandb_tracking.py`) on
`research/ffn_replacement_fix`.

## 6. Gotchas (self-hosted server)

- **`define_metric` has no description.** Its parameters are `step_metric`, `step_sync`,
  `hidden`, `summary`, `goal`, `overwrite`; passing `description=` raises `TypeError`.
  Metric descriptions have to live elsewhere (section 5).
- **No panel or section descriptions** in the workspace UI. Sections are auto-named by
  key prefix (`train/…`, `time/…`); unprefixed keys land in "Charts".
- **Reports without `wandb-workspaces`.** The Python reports API needs
  `pip install wandb[workspaces]`. Without it, create a report with raw GraphQL
  `upsertView` (`type: "runs"`, a JSON `spec` whose blocks include a `markdown-block`), find
  it again by title (`project.allViews(viewType: "runs")`) and update it by `id`, so
  re-publishing never duplicates. Report URL: `<server>/<entity>/<project>/reports/<Title-slug>--<viewId>`.
  The project description (the README on the project Overview) is set with
  `upsertModel(input: {entityName, name, description})`, which changes no other project field.
- **Don't log a `wandb.Table` into run history.** On some self-hosted versions the
  auto-created workspace "Tables" panel errors ("Oops, something went wrong"). Put the table
  in an artifact via `log_artifact` only; it renders under Artifacts → the artifact → Files.
- **Auto git capture renders as commands, not links.** The run Overview shows
  `git clone <remote>` and `git checkout -b <run> <sha>`, with an ssh host alias shown verbatim.
  Overriding `settings.git_remote_url` records the remote but **drops the commit**. Put
  clickable links in the notes instead. Runs uploaded after the fact record the
  uploader's HEAD, not the run's commit — say so in their notes.
- **Editing notes after the fact.** The public `Run.update()` re-sends config, tags and
  summary. For a notes-only edit use GraphQL `upsertBucket(input: {id, notes})` with the run's
  storage id; everything else stays untouched. The public `wandb.Api()` caches `Run` objects,
  so verify with a fresh query.
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
  project; always `wandb.finish()`; log `val_bpb`; write run notes; add a glossary entry
  in the same change that starts logging a new key.
- **Don't** hardcode the key, create per-experiment projects, or point runs at
  `wandb.ai` (always set `WANDB_BASE_URL`).
- Prefer resuming a crashed run with its id (`wandb.init(id=..., resume="allow")`)
  over creating a duplicate.

Related: a TensorBoard viewer is a lighter-weight alternative; wandb is the
primary shared dashboard because it aggregates many runs/branches into one
comparable project.
