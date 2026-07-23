# Working conventions for AI assistants on spiky

*Behavioral standing instructions for any Claude/assistant doing work on spiky — how to work,
not what the science is. For the research thesis see [thesis.md](thesis.md); for how experiments
are structured and launched see [experiment-methodology.md](experiment-methodology.md).*

These are Anatoli's standing instructions. They used to live in a `CLAUDE.md` at the repo root;
they now live here, under `claude/`, with the rest of the shared knowledge base.

## Plan before big tasks

Before starting any sizable/big task, **first report the plan of what you're going to do and get
sign-off**, rather than diving straight into implementation. Especially important for large tasks.

- **Why:** Anatoli wants a chance to redirect scope/approach before effort is spent — a wrong big
  task is expensive to unwind, a quick plan check is cheap.
- **How to apply:** For anything beyond a small/mechanical change (new modules, multi-file
  features, migrations, experiments, anything that takes real time or tokens), surface a concise
  plan first — what you'll build, key decisions, files touched — and wait for sign-off before
  implementing. Small/obvious edits and read-only investigation don't need it. In
  UNATTENDED/delegated-task mode where blocking for input isn't possible, the plan itself is a
  valid RESULT: report the plan back and let Anatoli approve in Slack before proceeding.

## Don't merge PRs

**PR merges are done by Anatoli personally in most cases. Do not merge PRs yourself** — open them
for review and leave them for Anatoli to merge, unless he explicitly tells you to merge.

- **Why:** merging is the review gate; Anatoli keeps that final call so nothing lands without his
  sign-off.
- **How to apply:** finish the work, push the branch, open/update the PR for review — then stop.
  Never merge (no `gh pr merge`, no merge via API/helper, no fast-forward push to the target
  branch to sidestep review). Only merge when Anatoli explicitly says to for that specific PR.
  (This is also why the vetted gh helper deliberately omits `pr-merge`.)

## Run git in the cage, ONE command per call

**Run git commands inside the cage (`sbox git …`), and run them ONE AT A TIME — never
batch/group multiple commands into a single call.**

- **Why:** a single clean `sbox <argv>` is auto-allowed frictionlessly, and inside the cage EVERY
  git subcommand works (even ones the bare-`git` allowlist gates, like `rev-list`/`ls-tree`). But
  the auto-allow requires the call to be exactly ONE `sbox <argv>` — the moment two commands are
  chained (`;`, `&&`, multiple `sbox …` segments, or a `cd X && git …` compound), the whole call
  stops being a single clean sbox call and gets **gated**. Verified empirically (2026-07-23):
  `sbox git … status` and `sbox git … rev-list` each ran clean solo; chaining two `sbox git …` in
  one call was declined. Grouping IS the problem.
- **How to apply:** issue each git command as its own call, wrapped in `sbox` — e.g.
  `sbox git -C <repo> add <files>`, then a separate call `sbox git -C <repo> commit -F <msgfile>`.
  Do NOT combine `add && commit`, do NOT `status; log`, do NOT `cd repo && git …`. Use
  `git -C <repo>` instead of `cd`. Local ops (status/diff/log/add/commit/branch/checkout/
  rev-list/ls-tree/show…) all work in-cage since `.git` under `~/projects` is writable there.
- **Network ops are the exception:** `push`/`pull`/`fetch`/`clone` need network, which the cage
  does NOT have, so those cannot run under `sbox`. Run each of them **outside** the cage (bare
  `git …` to a configured remote is auto-allowed), still ONE command per call.
