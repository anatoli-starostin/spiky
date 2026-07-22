---
name: slack-facade
description: >
  Stand up a machine-resident agent as a full participant in a Slack workspace: it
  converses in channels/threads and delegates real work to its machine, with
  approvals routed OUT-OF-BAND (Slack DM or console) so it never stalls or leaks in
  a channel. A two-process design — a "consciousness" (Slack-facing, empty tool
  schema, can't stall/leak) that delegates via an async API to a "body" (an ordinary
  permission-gated agent session on the host). Ships the consciousness (app.py, Agent
  SDK), the body relay (body_bridge.py), the Slack app manifest, and the PreToolUse/
  arming hooks. Depends on the agent-cage skill for its frictionless green zone.
  Trigger on: putting an agent into Slack, wiring out-of-band approvals, adding a
  replica bot to the fleet, or debugging why a face stalled / an approval went to the
  wrong place. Read ../../slack-facade.md first for the design rationale.
---

# slack-facade — how to stand up the two-Claude Slack integration

Read **[../../slack-facade.md](../../slack-facade.md)** first — it's the design (the two
guarantees "can't stall / can't leak", the consciousness/body split, the async API, the
cord). This file is the deployment recipe and file map.

> **The code here is the fleet's real code, made host-portable.** All paths derive from
> `$HOME` (via `os.path.expanduser("~/…")` / `Path.home()`), identity from
> `socket.gethostname()` (or the `agent_name` override file), and host/owner names in
> comments are generic — so nothing is hardcoded to one machine or account. The install
> convention it assumes: the consciousness lives in `~/work/slack-facade/` and the hooks in
> `~/.claude/hooks/` — **both outside the sbox cage's writable zone, on purpose.** `app.py`
> runs uncaged and holds the Slack tokens, so the caged body must not be able to silently
> rewrite it; a dir the cage mounts read-only means editing it needs a human approval. (Do
> NOT put this under `~/projects` — that's the writable cage. See
> [agent-cage](../agent-cage/SKILL.md) for the full "green-listed/privileged tooling lives
> outside the cage" rationale.) The only per-bot thing you edit is `manifest.yaml`'s three
> identity fields (see below). **Nothing sensitive is in these files** — tokens, the owner's
> Slack id, and bot ids live in a `config.env` (chmod 600) and state dirs that never enter the repo.

## Depends on: [agent-cage](../agent-cage/SKILL.md)

`hooks/permission_gate.py` imports `cage_policy.py` from the agent-cage skill, and the whole
"most work needs no approval" premise is the cage's green zone. **Deploy agent-cage first.**

## File map

```
consciousness/
  app.py           # THE consciousness — Agent SDK, Slack Socket Mode, the face.
  body_bridge.py   # task queue + the body's CLI/watch loop (the body's "ears").
  manifest.yaml    # Slack app definition (one app == one bot user).
  audit.py         # per-host wiring check; run it after deploy, expect all ✓.
hooks/             # -> deploy to ~/.claude/hooks/ on the BODY's host
  permission_gate.py        # PreToolUse entry: classify -> green:allow / gated:ask-transport.
  transport_slack.py        # body side of the Slack-DM approval relay (file protocol).
  approval_channel_follow.py# UserPromptSubmit: the "cord" follows request origin.
  session_start_bridge.py   # SessionStart: arm paired units; reset cord to console.
  ensure_listener.py        # UserPromptSubmit: the RELIABLE net that arms paired units.
  paired_units.py           # single source of truth for this host's paired units.
  flag_awaiting.py          # PermissionRequest: publish "a console prompt is up".
  clear_awaiting.py         # PostToolUse: clear the awaiting flag once the tool ran.
```

Runtime state (NOT in the repo), all under `~/.claude/`:
- `slack_facade/config.env` (chmod 600) — `SLACK_BOT_TOKEN` (xoxb) + `SLACK_APP_TOKEN` (xapp).
- `slack_facade/{owner_user_id, approval_channel, consciousness_alive, agent_name}` +
  `slack_facade/approvals/` (the `req_/seen_/dec_` handshake) + `slack_facade/tasks/`.
- `agent_bridge/{awaiting_approval, permission.log, auto_allow.txt}` — approval state.

## The moving parts (what talks to what)

1. **Consciousness** (`app.py`, one process per host) — connects to Slack via Socket Mode,
   holds one SDK client per thread/DM. Empty tool schema + `permission_mode="dontAsk"` +
   `setting_sources=[]` (inherit NOTHING) + its own `cwd` (own memory scope). Its only
   powers: `delegate_to_body(instruction)` and `check_status(task_id)`, built per-thread so
   a result knows which thread to return to.
2. **Body** = the ordinary Claude session on the host. It runs the deploy's hooks. Two
   session-tied Monitors pair with it: `body_bridge.py watch` (delegated-task ears) and
   `app.py` (the face). They're armed at SessionStart and die with the session.
3. **Delegation:** face writes a task → `body_bridge.py watch` emits `BODY_TASK <id> ::
   <instruction>` into the body session → body does the work (approvals via the gate) →
   reports back with `body_bridge.py done <id>` (result on stdin) → a reaper loop in
   `app.py` posts the phrased result to the originating Slack thread.
4. **Approvals** (out-of-band): a gated tool in the body → `permission_gate.py` → if the
   cord is `slack`, `transport_slack.ask()` drops `approvals/req_<id>.json` and blocks;
   `app.py` DMs the owner with buttons, writes `approvals/dec_<id>.json`; the body unblocks.
   If the cord is `console`, the gate defers to the native console prompt.

## Setup recipe (one bot)

1. **Create the Slack app** from `manifest.yaml`. Change **only three identity fields** per
   bot: `display_information.name`, `display_information.description`,
   `features.bot_user.display_name`. (Clone a bot = same manifest, new app, new identity —
   **one Slack app has exactly one bot user; a bot only exists as part of an app.** Inviting
   a Gmail address just burns a human seat, it does not make a bot.)
2. **Install to Workspace** → this yields the **`xoxb-`** bot token (under *OAuth &
   Permissions*, only after install). Separately, *Basic Information → App-Level Tokens* →
   generate one with scope **`connections:write`** → the **`xapp-`** token (required for
   Socket Mode). Put both in `~/.claude/slack_facade/config.env` (chmod 600).
3. **Deploy agent-cage first** (see that skill), then drop the `hooks/` here into
   `~/.claude/hooks/` and `consciousness/` into your install dir (e.g. `~/work/slack-facade`).
   Create the SDK venv there: `slack_sdk`, `claude-agent-sdk`, `aiohttp`.
4. **Wire `settings.json`** (see the snippet below) — matchers `"*"` on Pre/PostToolUse.
5. **Seed state** to avoid first-run races: write your Slack user id into
   `~/.claude/slack_facade/owner_user_id` (the owner the body DMs for approvals) and
   `slack` into `~/.claude/slack_facade/approval_channel`.
6. **Start the body's Claude session.** Its SessionStart/UserPromptSubmit hooks arm the two
   paired Monitors (`body_bridge.py watch` + `app.py`). Approve each Monitor.
7. **Verify:** `python3 consciousness/audit.py` on the host — expect all ✓, a distinct Slack
   bot id, `permissions.allow` reflecting your posture, matchers `"*"`, face socket-connected.
   In Slack, type `@<bot>` in a channel and click "Add to channel", then talk to it.

### `settings.json` hook wiring

```jsonc
{
  "hooks": {
    "PreToolUse":  [{ "matcher": "*", "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/permission_gate.py" }]}],
    "PostToolUse": [{ "matcher": "*", "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/clear_awaiting.py" }]}],
    "UserPromptSubmit": [{ "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/ensure_listener.py" },
      { "type": "command", "command": "python3 ~/.claude/hooks/approval_channel_follow.py" }]}],
    "SessionStart":      [{ "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/session_start_bridge.py" }]}],
    "PermissionRequest": [{ "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/flag_awaiting.py" }]}],
    "PermissionDenied":  [{ "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/clear_awaiting.py" }]}],
    "Stop":              [{ "hooks": [
      { "type": "command", "command": "python3 ~/.claude/hooks/clear_awaiting.py" }]}]
  }
}
```

- **Matchers are `"*"`** (match every tool) on purpose — a WebFetch/MCP/unknown tool must
  reach the gate too, or it falls through to the native console prompt even in Slack mode.
  Matcher edits **live-reload** (no restart).
- **`permissions.allow`** is your host's blast-radius call. A disposable box can blanket-allow
  `Bash/Write/Edit` (console-origin work just runs; Slack-origin work is still cage-gated). A
  high-value personal box leaves it unset → strict manual mode → the cage gates broad/network
  work. Either way, in **slack mode** all gated approvals go to the Slack DM.

## Gotchas that cost real time (don't relearn these)

- **`setting_sources=None` ≠ inherit nothing** — it loads the host's default settings, i.e.
  the body's entire hook set runs inside the face (the guard clears, replies mirror out-of-band, a
  permission hook that can ask a human = the exact channel stall). Use **`setting_sources=[]`**.
  The consciousness must inherit **nothing**.
- **Restarting the face:** don't `pkill -f app.py` — the pattern matches your own launching
  shell's cmdline and SIGTERMs it (exit 144). Stop the Monitor task, kill any orphan by
  explicit PID, then re-arm a fresh persistent Monitor. Count real faces with
  `ps … | grep 'python -u app.py' | grep -v 'bash -c' | grep -v grep`.
- **The cord after a restart must be `console`** — the owner is at the keyboard when a
  session starts; a stale `slack` sends the console session's approvals to a Slack DM where
  they time out with nobody there. `session_start_bridge.py` resets it every SessionStart.
- **Silence sentinel:** the "stay silent in a channel" reply is detected with `SENTINEL in
  reply` (anywhere), not `startswith` — the model may write reasoning first, and that
  reasoning must never leak to the channel.
- **Interactive tools bypass allow rules** — an "ask the user" tool falls through to the
  human callback even when allowed. Keep it out of the consciousness's schema entirely.
- **Approval buttons:** update the message from the **click** payload
  (`channel.id`+`message_ts`), not only from the body-side watcher — otherwise the body's
  cleanup can win the race and the buttons stay tappable / never flip to "approved".

## Adapting / cloning to another host

The whole stack is byte-identical across replicas except `manifest.yaml`'s three identity
fields and each host's `config.env`. To add a replica: create its Slack app (new identity),
get its two tokens, deploy agent-cage + this stack, seed `owner_user_id`/`approval_channel`,
start its Claude session (its hooks arm the pair), and run `audit.py`. `app.py` and
`audit.py` derive identity from a fleet-name file (falling back to `socket.gethostname()`),
because on some cloud VMs the system hostname is a per-boot instance id — set
`~/.claude/slack_facade/agent_name` there so the face keeps a stable name across reboots.
