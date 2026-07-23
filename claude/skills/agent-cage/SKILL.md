---
name: agent-cage
description: >
  A frictionless "green zone" sandbox for an autonomous coding/ML agent, plus the
  PreToolUse policy that classifies every tool call as ungated / green / gated.
  `sbox` runs a command inside a bubblewrap cage: whole system read-only EXCEPT a
  few working dirs (~/projects, /tmp, ~/.cache), GPU passed through, NO network,
  ~/.ssh masked — so in-cage work is auto-allowed and only crossing a boundary
  (network, a write outside the zone, a broad command) trips a human approval.
  cage_policy.py is the transport-free classifier (green = read-only tools + one
  clean `sbox <argv>` + scoped-safe git + writes-in-green-zone); it is the
  precondition the slack-facade skill's approvals depend on. Trigger on: sandboxing
  an agent's shell/GPU work, deciding what should auto-run vs ask, standing up the
  cage on a new host, or debugging why a command was gated.
---

# agent-cage — the frictionless green zone + the permission classifier

An autonomous agent that asks a human before every command is useless; one that runs
*everything* silently is dangerous. The cage resolves this with a **friction gradient**:
give the agent a roomy zone where work is free (no prompts), and make **crossing out of
it** — network, a write to a real path outside the zone, a broad system command — the thing
that trips an out-of-band approval. The cheap path is the safe path, so the agent learns to
prefer it, and the rare approvals that remain are high-signal.

This skill is the **precondition** for [slack-facade](../slack-facade/SKILL.md): the
Slack façade routes *how* a gated approval reaches a human; this skill decides *what* is
gated in the first place.

> **All paths derive from `$HOME`** (`~/projects`, `~/.claude/hooks`, `~/.config/spikybot/pat`)
> — nothing is hardcoded to one machine or user. The install convention: `sbox` on your
> `PATH` (`~/.local/bin`), `cage_policy.py` in `~/.claude/hooks/`, the helper in
> `~/work/gh-issue/` — a dir the cage mounts **read-only**, on purpose (see the helper
> section for why). Sensitive values (tokens, the GitHub PAT) live outside the repo.

## Files

| File | Where it lives | What it is |
|---|---|---|
| `sbox` | `~/.local/bin/sbox` (on `PATH`) | the bubblewrap cage wrapper |
| `cage_policy.py` | `~/.claude/hooks/cage_policy.py` | the classifier: `classify(tool, input) -> ungated \| green \| gated` |
| `gh_issue.py` | `~/work/gh-issue/gh_issue.py` | an optional scoped-safe helper (a green-listed GitHub-issue tool) |

`cage_policy.py` is imported by the slack-facade skill's `permission_gate.py` — deploy the
two together; the gate can't run without the classifier.

## `sbox` — the cage

`sbox CMD [args…]` runs `CMD` inside a [bubblewrap](https://github.com/containers/bubblewrap)
namespace:

- whole filesystem **read-only** (`--ro-bind / /`), EXCEPT **`~/projects`, `/tmp`,
  `~/.cache`** which are writable;
- **GPU passed through** — every existing `/dev/nvidia*` (+ `/dev/dri`) node is
  `--dev-bind`ed, so CUDA works in-cage (nodes are added conditionally, so the same script
  is correct on a GPU box and a CPU-only NUC);
- **no network** (`--unshare-net` → an empty net namespace: a raw socket gets
  `Errno 101 Network unreachable`, `curl` fails — enforcement is *below* the process, so
  switching language or hiding intent can't bypass it);
- **`~/.ssh` masked** (`--tmpfs`) so caged code can't even read keys;
- `--die-with-parent` so a caged process can't outlive its launcher.

**Chaining:** the wrapper only cages the argv it's given; the surrounding shell (`&&`, `|`,
`>`, `$(…)`) runs **uncaged**. So run a pipeline *inside* the cage with one call:

```sh
sbox bash -c 'cd ~/projects/x && python train.py | tee /tmp/log'
```

Keep single commands single — a bare `sbox X && ls` is multi-segment and gets gated on the
uncaged `&& ls`.

**⚠️ A heredoc gates even under `sbox`.** `sbox python3 - <<'EOF' … EOF` asks for approval
despite the `sbox` prefix, because the `<<` is an outer-shell **redirect** — and the policy
gates any outer redirect to a real target (only `/dev/null` and fd-dups like `2>&1` are
exempt). The prefix doesn't protect a line that has a top-level operator. To run a multi-line
script frictionlessly, **write it to a file** in a writable dir and run `sbox python3
script.py` (a single clean segment); for a short snippet use `sbox python3 -c '…'` with the
body single-quoted. Any `<`/`>`/`|`/`&&`/`$()`/heredoc you actually need must live *inside*
`sbox bash -c '…'`, never at the outer level.

### Standing `sbox` up on a new host

1. `sudo apt-get install -y bubblewrap socat`
2. Drop `sbox` at `~/.local/bin/sbox`, `chmod +x`.
3. **Ubuntu 24.04+ userns restriction:** if bwrap fails with `setting up uid map:
   Permission denied` / `loopback: … Operation not permitted`, the kernel has
   `kernel.apparmor_restrict_unprivileged_userns=1` **enforcing** and the host lacks the
   `bwrap-userns-restrict` AppArmor profile. Copy a working host's
   `/etc/apparmor.d/bwrap-userns-restrict` over and `apparmor_parser -r` it. **Match the
   ABI:** if the target's AppArmor is older (e.g. 4.0.1, ABI max 4.0) than the profile
   declares (`abi <abi/5.0>`), downgrade the profile's `abi` line to `<abi/4.0>` — all the
   rules exist in 4.0 — then reload. The profile persists across reboot.
4. Verify: `sbox nvidia-smi` (GPU visible), `sbox python3 -c 'import socket;
   socket.create_connection(("1.1.1.1",53))'` (should raise `Network unreachable`),
   `sbox bash -c 'echo x > ~/projects/t && echo x > ~/t'` (first ok, second read-only),
   `sbox ls ~/.ssh` (empty).

**Installing new packages** (`uv pip install`, `hf download`) needs network → those stay a
gated/ask action. Already-downloaded caches under `~/.cache` remain usable offline in-cage,
which is why `~/.cache` is one of the writable binds. Use `uv` for venvs (offline-friendly);
system `python3 -m venv` + pip may fail on Ubuntu's ensurepip regardless of the cage.

## `cage_policy.py` — the classifier (the fundamental constraint)

Standalone (imports only `re`, `shlex`), **no transport dependencies** — deliberately, so
the cage constraint can't get entangled with *how* approvals are asked. `classify(tool,
tool_input)` returns:

- **`ungated`** — a small read-only tool allowlist (`Read`, `Glob`, `Grep`, `LS`,
  `NotebookRead`, `TodoRead/Write`, `BashOutput`, …). Let Claude Code's own layer handle it.
- **`green`** — safe, auto-allow silently. Green covers:
  - a single **clean `sbox <argv>`** with **no shell operators**;
  - **scoped-safe git** (`status`/`diff`/`log`/`add`/`commit`/`branch`/`checkout`/`stash`,
    and `pull`/`fetch`/`push` to a *configured named remote*) — but **NOT** explicit URLs,
    `ext::`, `-c`, `config` writes, `--upload-pack`/`--exec` (exfil / RCE / prompt-injection
    vectors — "all git is safe" is false even without admin rights);
  - **file writes** (`Write`/`Edit`/`NotebookEdit`) whose **absolute** path is under the
    green zone (`~/projects`/`/tmp`/`~/.cache`), mirroring `sbox`'s writable binds;
  - optionally, one vetted helper by absolute path (see `gh_issue.py` below).
  - the **two fixed paired-unit `Monitor` arms** — the delegated-task body-watch and the
    paired Slack face — matched by **exact** command (`is_paired_unit_monitor`), so a caged
    agent self-arms its ears at SessionStart with no prompt. Fail-closed: any other `Monitor`
    command gates. (Both units live under the read-only-in-cage `_TOOLS_DIR`, so the caged
    agent can't rewrite what actually runs — same trust basis as `body_bridge.py`.)
- **`gated`** — **everything else** (default). Any plain uncaged command; ANY command with
  shell operators, even `sbox`-prefixed (operators drop to ask — permission is bound to
  *capability*, not syntax, so splitting a command doesn't evade the gate); broad
  Write/Edit outside the zone; `WebFetch`/`WebSearch`/MCP/unknown tools; installs.

**Security-critical parsing** (this is where a naïve classifier leaks):
- `_segments` uses `shlex.shlex(punctuation_chars=True)` so operators are isolated **even
  when attached** — `foo|curl` becomes `foo | curl`, it can't hide in a token.
- `_escape_ops` is a **quote-aware** scanner that flags redirects (`<`/`>`), substitution
  (`$(…)`/backticks), and a lone `&` **at the outer shell only** — so `sbox bash -c '…>…'`
  is green (the redirect is inside the caged argv) but `sbox foo > realfile` is gated.
- **Harmless outer redirects are allowed:** target `/dev/null` and fd-dups (`2>&1`, `>&2`)
  write nothing real, so `sbox … 2>/dev/null` stays green; `sbox … > realfile` gates.
- git is fully validated by an `is_safe_git` subcommand allowlist + flag/URL rejection.

The module ships with a unit-test battery (attached operators, quoted-redirect
false-positives, `$()` and subshell smuggling, `-c` injection, URL-exfil pushes). **When a
legitimate command over-gates, tune the classifier and re-test — don't widen the cage.**

`base_tokens(command)` is a small helper the gate reuses to name a command for the
"Always"-allow memory.

## `gh_issue.py` — an optional green-listed helper (the extension pattern)

A worked example of **safely widening the green zone with one vetted tool**. It's a
stdlib-only GitHub REST client (issues: list / view / create / comment / close / reopen /
label; PRs: pr-create / pr-list / pr-view / pr-diff / link-branch — and view / comment /
close / reopen / label also accept a PR number, since PRs are issues). **`pr-merge` is
deliberately absent** — merging stays a human action (the branch+PR review gate), so it's
kept out of this frictionless surface rather than green-listed. `cage_policy` green-lists the
helper by **absolute path** via
`is_safe_gh_issue()` (same precedent as any single trusted helper). It reads a PAT from
`~/.config/spikybot/pat` (never inline, never in the repo), defaults its repo from
`GH_ISSUE_REPO`, and takes issue bodies via `--body-file` (never inline) so the whole
invocation stays a **single green segment**. This is the template for adding a capability
without opening a general hole: one absolute-path helper, secrets read from a file, no
inline shell.

> **⚠️ A green-listed helper MUST live outside the writable cage.** `cage_policy` trusts
> `gh_issue.py` by its path — so if it sat in `~/projects` (or any `sbox`-writable dir), the
> caged agent could *silently rewrite the helper* and it would still auto-run green,
> collapsing the whole "narrow, vetted capability" guarantee. Keep it somewhere the cage
> mounts **read-only** (cage_policy's single `_TOOLS_DIR` constant — default `~/work`,
> overridable via the `AGENT_TOOLS_DIR` env var), so tampering with the trusted
> tool requires a human approval — exactly like `cage_policy.py`/`permission_gate.py` living
> in `~/.claude/hooks/` rather than in the cage. The directory *name* is incidental; the
> **read-only-in-cage** property is the requirement. Same reasoning applies to the
> slack-facade consciousness (`app.py`), which runs uncaged and holds the Slack tokens.

## Relationship to the Slack façade

`permission_gate.py` (in [slack-facade](../slack-facade/SKILL.md)) is the PreToolUse entry:
it calls `cage_policy.classify`, silently allows `green`, defers `ungated`, and for `gated`
asks the human via the active transport (Slack DM or console). This skill is that gate's
foundation — the classifier and the cage it refers to. The two skills are deployed together.
