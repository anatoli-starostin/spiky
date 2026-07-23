# Standing instructions

## Plan-before-big-tasks (standing instruction from Anatoli)

Before starting any sizable/big task, **first report the plan of what I'm going
to do and get sign-off**, rather than diving straight into implementation.
Especially important for large tasks.

- **Why:** Anatoli wants a chance to redirect scope/approach before effort is
  spent — a wrong big task is expensive to unwind, a quick plan check is cheap.
- **How to apply:** For anything beyond a small/mechanical change (new modules,
  multi-file features, migrations, experiments, anything that takes real time or
  tokens), surface a concise plan first — what I'll build, key decisions, files
  touched — and wait for sign-off before implementing. Small/obvious edits and
  read-only investigation don't need it. In UNATTENDED/delegated-task mode where
  blocking for input isn't possible, the plan itself is a valid RESULT: report
  the plan back and let Anatoli approve in Slack before proceeding.

## Don't-merge-PRs (standing instruction from Anatoli)

**PR merges are done by Anatoli personally in most cases. Do not merge PRs
yourself** — open them for review and leave them for Anatoli to merge, unless he
explicitly tells you to merge.

- **Why:** merging is the review gate; Anatoli keeps that final call so nothing
  lands without his sign-off.
- **How to apply:** finish the work, push the branch, open/​update the PR for
  review — then stop. Never merge (no `gh pr merge`, no merge via API/helper,
  no fast-forward push to the target branch to sidestep review). Only merge when
  Anatoli explicitly says to for that specific PR. (This is also why the vetted
  gh helper deliberately omits `pr-merge`.)
