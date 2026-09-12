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
- Optionally set `run.notes` (or `wandb.init(notes=...)`) to a one-line hypothesis.

## 5. Viewing and comparing

- Open `<WANDB_BASE_URL>/<YOUR_ENTITY>/<project>` and use the **group** selector to
  collapse families; the runs **table** + **parallel-coordinates**/scatter panels
  compare hyperparams vs. metrics; **filter by `tags`** (branch, machine).
- If a run seems "missing", check you're logged into the browser as `<YOUR_ENTITY>`
  and looking at the right project (a run's project is set by `project=`).

## 6. Do / don't

- **Do** keep the API key out of the repo (it lives only in `~/.netrc`); use one
  project; always `wandb.finish()`; log `val_bpb`.
- **Don't** hardcode the key, create per-experiment projects, or point runs at
  `wandb.ai` (always set `WANDB_BASE_URL`).
- Prefer resuming a crashed run with its id (`wandb.init(id=..., resume="allow")`)
  over creating a duplicate.

Related: a TensorBoard viewer is a lighter-weight alternative; wandb is the
primary shared dashboard because it aggregates many runs/branches into one
comparable project.
