# Working conventions for AI assistants on spiky

*Behavioral standing instructions for any Claude/assistant doing work on spiky — how to work,
not what the science is. For the research thesis see [thesis.md](thesis.md); for how experiments
are structured and launched see [experiment-methodology.md](experiment-methodology.md).*

These are Anatoli's standing instructions. They used to live in a `CLAUDE.md` at the repo root;
they now live here, under `claude/`, with the rest of the shared knowledge base.

## Language

**Always answer in English, even when the user writes or asks in Russian (or any other
non-English language).** Spiky is an international project, so English is the default and
preferred language for all communication.

- **Why:** the project spans multiple people and machines; keeping to one shared language
  (English) makes replies, issues, PRs, commit messages, and code comments readable by everyone.
- **How to apply:** default every response — and everything you write into the repo (commits,
  issues, PRs, comments) — to English, regardless of the language you were addressed in. The
  **only** exception is an explicit request to use another language — e.g. "translate to Russian",
  "answer in Russian", "переведи на русский", "ответь на русском", and equivalents — in which case
  honor the requested language for that response. An incidental non-English phrase in the prompt is
  not such a request; only an explicit instruction to switch languages is.

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
