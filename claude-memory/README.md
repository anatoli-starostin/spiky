# claude-memory

Durable knowledge retrieved from the two Nebius VMs used for the spiky / LUTorch
research project (Mar–Jun 2026), recovered after the SSH key to both machines was
lost. Raw conversation transcripts were **not** included (only distilled knowledge).

## Layout

### `main-h100/` — the main H100 training machine
- `memory-bank/` — 94 curated memory files (`MEMORY.md` master index + 82 `project_*`
  findings, 9 `feedback_*` working preferences, 2 `reference_*`). The record of what
  worked, what didn't, and why, across the whole experiment arc.
- `skills/` — `read-arxiv-paper`
- `commands/` — `bot-pr`, `ml-ds`
- `hooks/` — the Telegram bridge scripts
- `notes/` — experiment session logs, summaries, architecture notes, math `.tex`

### `secondary/` — the paper-writing machine
- `skills/` — `paper-writing`, `spiky-installer`
- `memory-bank/` — feedback + index
- `papers/` — the `Permutations-is-all-you-need` LaTeX project
- `notes/` — `spiky-feat` experiment notes

## Note on secrets
Credentials that were embedded in the source files (a GitHub PAT, a Telegram bot
token, a chat ID) have been **replaced with `<REDACTED_*>` placeholders** before
publishing to this public repo. They should be rotated at the source.
