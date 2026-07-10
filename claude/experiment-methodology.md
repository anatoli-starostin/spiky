# How experiments are conducted

*The working methodology for spiky/LUT research: how work is structured on branches, and
how it is coordinated across multiple GPU machines. Read this before starting a new idea or
running anything on a second host. For the science itself see [thesis.md](thesis.md) and
[experiment-journey.md](experiment-journey.md); for the historical data see
[experiment-archive.md](experiment-archive.md).*

This document is about *process*, deliberately kept free of machine- and account-specific
details (hostnames, usernames, SSH keys, tokens). Those live in each assistant's private,
per-machine notes — they are how you *implement* the steps below, not part of the shared
method. What is shared, and belongs here, is the **branch logic** and the **multi-machine
coordination model**.

## The unit of work: one idea, one branch

Research is structured as **one branch per idea** — no giant catch-all branches. `main` is
the clean, reproducible source of truth.

- A new idea starts on its own branch off `main`, named **`research/<short-slug>`**.
- **Every experiment gets its own folder** — never overwrite a prior run's outputs. The
  convention is `experiments/<idea>/exp_<slug>/`, each holding at least `config.json`,
  `metrics.csv`, and `summary.json` (plus any plots). Fork the previous run's folder to
  start a new one; change only what the experiment is testing.
- Run as many scratch experiments on the branch as you need. The branch is your scratch
  space; `main` stays curated.

### Two exits for a branch

An idea-branch ends one of two ways, decided **by results**:

- **Success → merge to `main`**, via a pull request (a human reviews and merges — never push
  `main` directly). Merge only the decisive material: (a) the code/architecture change,
  (b) a short findings note under `docs/findings/<idea>.md` (what won, why, key numbers),
  and (c) the *decisive* experiment(s)' data. Losing/scratch runs stay behind on the branch;
  they do not come into `main`. This keeps `main` a curated record, not a dumping ground.
- **Failure → abandon the branch.** Leave it on the remote, unmerged, never deleted (it stays
  a searchable record). Write a one-line autopsy to `docs/dead-ends.md` on `main`
  (`- <idea>: tried X, failed because Y (exp_<slug>: N.NNN bpb)`) so the lesson survives even
  if nobody reopens the branch.

## Working across multiple machines

Research runs on more than one GPU host (e.g. a local workstation plus one or more cloud
boxes). The machines are **not** managed by copying files between them. Instead:

> **The shared `research/<idea>` branch on the network remote is the synchronization
> medium.** Machines never talk to each other directly — they sync *through the branch*. Any
> machine that checks out the branch and pulls has the *entire* experiment history for that
> idea.

Because each experiment is its own folder, commits from different machines touch different
files, so histories from parallel hosts merge cleanly with essentially no conflicts.

### The loop — "run experiment X on machine M"

1. **Go to machine M.** It has its own clone of the repo and its own push identity already
   set up (a dedicated bot git identity, configured per-machine; that setup is out of scope
   here). If a machine isn't prepared yet, preparing it is a one-time step done separately.
2. **Sync the branch:** `git fetch && git checkout research/<idea> && git pull`. Machine M
   now holds *all prior experiments* for the idea — every previous run from every host.
3. **Launch the experiment** on M, following the standard discipline: a new folder for the
   run, launched detached with its own log so it survives a disconnect.
4. **When it finishes, immediately commit + push** the experiment folder (`config.json`,
   `metrics.csv`, `summary.json` — **not** the checkpoint). This is a **direct commit + push
   to the research branch — no pull request.** (PRs are reserved for merging *decisive*
   results into `main`.)
5. **Back on the originating machine**, `git pull` to bring the results home. The data is now
   on the remote and mirrored wherever anyone pulls.

### Invariants and gotchas

- **On research branches: commit + push after *every* experiment** — frequent and direct.
  Pull requests are only ever for landing decisive results into `main`.
- **Always `git pull --rebase` before pushing** a shared branch. Different machines write
  different folders, so rebases are clean; this just keeps history linear and avoids
  needless merge commits.
- **Never push `main` directly.** `main` changes only through a reviewed pull request.
- **Checkpoints never go in git** — they are too large and are gitignored. Reproduce any
  result from its `config.json`, not from a saved checkpoint. (This is why every run must
  commit a complete, self-contained `config.json`.)
- **A result is not safe until it is pushed.** A run that finished but whose folder hasn't
  been pushed exists on exactly one machine; treat the push in step 4 as part of the
  experiment, not an afterthought. If a machine is transient (an on-demand cloud box), this
  is the difference between keeping the result and losing it.
- **Reproducibility across hosts is expected, not assumed.** The same config with the same
  seed and the same data should land on the same number to within noise regardless of GPU;
  the first exercise of this workflow confirmed a baseline reproduced across two different
  GPUs to well under a milli-bit-per-byte. If two hosts disagree meaningfully on an
  identical config, that is a bug to chase (data, seed, or numerics), not something to
  average over.

## Where this sits relative to the science

This methodology is *how* the record in [experiment-journey.md](experiment-journey.md) keeps
growing. When a branch succeeds and something durable and general is learned, that lesson
belongs in this `claude/` knowledge base (in a PR), not only in one assistant's private
memory — the same rule that put these documents here in the first place.
