# `claude/` — scientific knowledge base for the spiky project

This folder is the **shared, version-controlled memory** for AI assistants (Claude and
others) working on **spiky**. Historically this knowledge lived only in each assistant's
private per-machine memory; as the project spreads across multiple machines it belongs
here, in the repo, so any assistant on any host inherits the same understanding.

**Audience:** a capable coding/research assistant who has just been pointed at this repo
and needs to understand *what spiky is trying to do*, *what has already been learned*, and
*what not to re-try*. It is written to be read top-to-bottom on first contact.

## What's here (read in this order)

1. **[thesis.md](thesis.md)** — the *why*. The scientific thesis behind spiky/LUTGPT:
   differentiable lookup tables as a permutation-coded, (nearly) matmul-free transformer.
   Start here.
2. **[experiment-journey.md](experiment-journey.md)** — the *what we tried*. The arc of
   ~250 experiments that produced LUTGPT: the eras, the durable mechanistic lessons (the
   load-bearing knowledge), the dead-ends that are already falsified, and the gotchas.
   Read this before designing a new experiment so you don't repeat settled work.
3. **[experiment-archive.md](experiment-archive.md)** — *where the data is*. How to find
   the full historical experimental record (configs / metrics / summaries for ~1,366 runs)
   and how to verify any specific experiment number or figure.
4. **[experiment-methodology.md](experiment-methodology.md)** — *how work is run*. The
   process: one branch per idea, and how experiments are coordinated across multiple GPU
   machines (the shared research branch as the sync medium). Read this before starting a new
   idea or running anything on a second host. Includes the **agree → commit → go** launch
   protocol (commit an experiment's code to the PR before running it; launch only after Anatoli's
   explicit go).
5. **[working-with-claude.md](working-with-claude.md)** — *how to work here*. Behavioral standing
   instructions for an assistant on spiky: plan before big tasks, and don't merge PRs.
   (Previously a root `CLAUDE.md`; consolidated here.)

## Reusable skills

- **[skills/](skills/)** — portable, machine-agnostic Claude *skills* for this project.
  These are capabilities, not scientific findings — drop a skill here when it's worth
  sharing across every assistant and machine.
  - **[paper-writing](skills/paper-writing/SKILL.md)** — detect the LaTeX toolchain and
    compile a `.tex` to a real PDF on any host.
  - **[agent-cage](skills/agent-cage/SKILL.md)** — a frictionless "green zone" sandbox
    (`sbox`) + the PreToolUse classifier that decides what auto-runs vs. asks a human, so an
    autonomous body works freely in-cage and only *crossing a boundary* trips an approval.
  - **[slack-facade](skills/slack-facade/SKILL.md)** — put a machine-resident agent into a
    Slack workspace as a full participant that delegates real work, with approvals routed
    out-of-band so it never stalls or leaks in a channel. Depends on agent-cage. The design
    rationale is written up separately in **[slack-facade.md](slack-facade.md)** (the *why*;
    the skill is the *how*).

### Installing a skill on a host (this folder is *source*, not the discovery path)

**Adding a skill here does NOT make it invokable.** Claude Code discovers skills only from
**`~/.claude/skills/<name>/`** (the per-user skills path). This `claude/skills/` folder is the
version-controlled *source of truth* / knowledge base — so to actually use a skill on a host you
must **install** it: copy the skill's directory into that host's `~/.claude/skills/`, e.g.

```sh
cp -r claude/skills/agent-cage ~/.claude/skills/agent-cage   # then /agent-cage works next session
```

Skills are loaded at **session start**, so a new session (or restart) picks up a freshly
installed skill. Two consequences to remember: (1) *publishing* a skill to this repo and
*installing* it on a host are **two separate steps** — never assume a repo skill is live
anywhere; do the copy per host (and per replica). (2) The installed copies are **snapshots that
drift** from the repo as the SKILL is edited — re-copy after changes to keep a host current.

## Scope and boundaries

- **The scientific record and the working method.** thesis / journey / archive are the
  settled findings and the reasoning behind them; [experiment-methodology.md](experiment-methodology.md)
  is *how* experiments are run and coordinated across machines. Both are kept free of
  machine- and account-specific details (hostnames, credentials, SSH keys) — those live in
  each assistant's private per-machine notes, not here.
- **Anchor of truth:** the single most authoritative scientific document is the **LUTGPT
  research report** (`doc/lutorch/lutgpt_research_report.pdf` in this repo). Where these
  notes and the report disagree, the report wins. These notes are a distilled, navigable
  companion to it — not a replacement.
- **Epistemic caveat:** these are distilled observations, accurate as of when they were
  written. Trust the *lessons*; verify a specific experiment id, bpb figure, or code
  citation against the report and the archive ([experiment-archive.md](experiment-archive.md))
  before quoting it as fact. Code moves; the mechanistic conclusions have been stable.

This folder is meant to grow. When you learn something durable and general about the
science, add it here (in a PR) rather than leaving it in a single assistant's private memory.
