# Experiment tracking with wandb (self-hosted)

How to log spiky experiments to our **self-hosted Weights & Biases** and keep
runs organized across many branches, machines, and experiment families. This is
tooling/operational guidance; the science lives in the other `claude/` docs.

> **Never commit secrets.** The wandb API key is a password. It is *not* in this
> repo and must never be — this doc explains how to obtain it, never its value.

## 1. The server

- Self-hosted **wandb-local** runs on **nucstar** (the always-on box) and is
  viewable in a browser at **http://nucstar:8080** over the tailnet.
- There is a **single entity** (account/namespace): **`astarostin`**. All runs
  and projects live under it, so everyone shares one view.
- It is a private deployment on our tailnet — reachable from gpustar and
  nebius-h100 (and any tailnet host), not from the public internet.

## 2. One-time client setup on a GPU box (gpustar / nebius-h100)

Do this once per machine, in the environment your training runs in:

```sh
pip install wandb                              # into the experiment's env
export WANDB_BASE_URL=http://nucstar:8080      # point the client at our server
wandb login --host http://nucstar:8080 <YOUR_API_KEY>
```

- **Get `<YOUR_API_KEY>` from the browser**: open http://nucstar:8080/authorize
  while logged in as `astarostin` and copy the key. Treat it like a password:
  never hardcode it, never commit it, never paste it into code or config. After
  `wandb login` it is stored in `~/.netrc` and used automatically.
- **Gotcha — key must be a positional arg.** On the client version we tested
  (wandb 0.30.0), setting only `WANDB_API_KEY` gave "No API key configured"; pass
  the key as the positional argument to `wandb login` as shown above.
- **Gotcha — on nucstar itself use `127.0.0.1`, not `localhost`.** `localhost`
  resolves to IPv6 `::1` while the server's port is forwarded on IPv4 only, so
  local calls on nucstar should use `http://127.0.0.1:8080`. Remote tailnet
  clients (gpustar, nebius) use the `nucstar` name normally.
- Keep `export WANDB_BASE_URL=http://nucstar:8080` in the run environment (e.g.
  your launch script) so every run targets our server rather than wandb.ai.

## 3. Minimal integration in a train.py

```python
import wandb, subprocess

sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
run = wandb.init(
    project="Spiky",                      # ONE project for everything (see below)
    group="hyperplane_ffn",               # experiment family / folder
    job_type="train",                     # train | eval | sweep
    name="exp006_nap6_tph256",            # concise, unique run name
    tags=["hyperplane_ffn", "lut", sha],  # flexible labels (branch, machine, ...)
    config=dict(branch="docs/wandb-guidelines", commit=sha, host="gpustar",
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
| `project`  | **always `"Spiky"`** — one project for all | `"Spiky"`                                 |
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

- Open **http://nucstar:8080/astarostin/Spiky** and use the **group** selector to
  collapse families; the runs **table** + **parallel-coordinates**/scatter panels
  compare hyperparams vs. metrics; **filter by `tags`** (branch, machine).
- If a run seems "missing", check you're logged into the browser as `astarostin`
  and looking at project **Spiky** (a run's project is set by `project=`).
- On nucstar itself, use `http://127.0.0.1:8080` (IPv6/IPv4 note in §2).

## 6. Do / don't

- **Do** keep the API key out of the repo (it lives only in `~/.netrc`); use one
  project `"Spiky"`; always `wandb.finish()`; log `val_bpb`.
- **Don't** hardcode the key, create per-experiment projects, or point runs at
  `wandb.ai` (always set `WANDB_BASE_URL`).
- Prefer resuming a crashed run with its id (`wandb.init(id=..., resume="allow")`)
  over creating a duplicate.

Related: nucstar also has a TensorBoard viewer as a lighter-weight alternative;
wandb is the primary shared dashboard because it aggregates many runs/branches
into one comparable project.
