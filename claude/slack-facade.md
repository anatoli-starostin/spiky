# The two-Claude Slack façade — putting an agent into a channel safely

This is the *why and what* of the Slack integration that lets a machine-resident agent
appear in a Slack workspace as a **full participant** — it converses in channels and
threads, and delegates real work to the machine it lives on. The *how* (the actual code,
files, and the step-by-step to stand one up) is the companion skill
[skills/slack-facade/](skills/slack-facade/SKILL.md). The frictionless-execution layer it
leans on is [skills/agent-cage/](skills/agent-cage/SKILL.md).

This document is deliberately free of hostnames, tokens, and account ids — those belong in
per-machine notes and in a `config.env` that never enters the repo. What's here is the
design that survives being copied to any host.

## The problem: a channel bot must never stall and never leak

The naïve way to put an agent in Slack is to **mirror** a terminal session — pipe messages
in, pipe output back. That is exactly wrong for a shared channel, for two independent
reasons:

1. **It can stall.** A normal agent session stops and asks a human ("Approve this
   command?"). In a terminal that's fine; in a channel it means the bot freezes mid-thread
   and dumps a permission prompt into a public room where nobody knows what it means.
2. **It can leak.** An agent with real tools (`Read`, `Bash`) is one polite request away
   from printing an SSH key or a bot token into a channel — and it cannot tell a stranger
   in the room from Human Master.

So the design is built around **two guarantees**, and everything else follows from them:

> **Can't stall:** no tool call the channel-facing agent makes can ever reach a human prompt.
> **Can't leak:** the channel-facing agent never holds a secret it could be talked into reciting.

## The split: consciousness + body

The two guarantees are met by splitting one agent into two processes with very different
privileges:

- **The consciousness** is the Slack-facing persona. It converses, reasons, and represents
  the agent socially. It runs with an **empty (or tightly scoped) tool schema** and a
  permission mode that **auto-denies anything unmatched without ever calling back to a
  human**. Because the dangerous tools aren't in its schema at all, there is no request that
  can surface a prompt — the "can't stall" guarantee is structural, not a promise. And
  because it has no filesystem/shell reach into the machine's secrets, "can't leak" holds
  too: it can't recite what it was never given.
- **The body** is an ordinary, fully-capable, permission-gated agent session doing the
  actual work on the machine — invisible to Slack. Its approvals are routed **out-of-band**
  to Human Master, never into the channel.

The one bridge between them is an **async, status-based API** — never a blocking call:

```
delegate_to_body(instruction) -> task_id        # returns instantly
check_status(task_id)         -> pending | awaiting-approval | done
```

A blocking call would leak the body's stalls back into the consciousness (the very thing we
split them to prevent). Because the API is async, the face can say *"queued — waiting on
Human Master's ok"* and stay fluent while the body waits.

### Why the consciousness is *fat*, not a one-tool relay

An early version made the consciousness a single-tool relay, reasoning that safety =
absence of capability. That's too rigid, and it turned out to be unnecessary: a permission
mode that removes tools from the model's schema entirely achieves capability-absence *by
configuration* on an otherwise fully-featured agent. So the real invariant is narrower and
better than "one tool":

> The invariant is **not** "one tool." It is **"no tool call can ever reach a human
> prompt."**

That lets the consciousness carry a *rich, safe* toolset (read-only tools scoped to its own
tiny workspace, plus the two delegation tools) with full personality and zero stall risk.

**⚠️ The landmine:** some tools — notably an interactive "ask the user a question" tool, and
any tool flagged as requiring user interaction — **fall through to the human callback even
when an allow rule matches them.** That is precisely the "bot freezes in a channel" failure.
They must be **removed from the schema explicitly**, never merely left un-allowed.

### Two orthogonal guardrail axes

| Axis | Threat | Mechanism |
|---|---|---|
| **Can't stall** | face freezes mid-thread | empty/scoped tool schema; auto-deny permission mode; no interactive tools |
| **Can't leak** | face recites a secret in a public room | face has its **own** cwd/memory, loads **none** of the machine's settings/hooks/skills, and never shares the body's memory |

The leak axis is easy to miss. `Read` is *permission-safe* but not *channel-safe*: a
face with unscoped read is one request from printing a token into a room. So the
consciousness gets its **own** working directory (hence its own memory scope), and is
launched with **no inherited settings** — see the isolation note below.

## The isolation that makes it real: load *nothing* from the host

The single most important implementation gotcha: when you launch the consciousness via the
Agent SDK, telling it to inherit "default" settings makes it load the **host's entire hook
set** — the guard hooks, the notification hooks, the permission hooks. That silently breaks
every guarantee (the face starts mirroring replies to Human Master's phone; a permission hook
inside the face can ask a human → the exact channel stall we forbid). The fix is to load an
**empty settings set** — inherit *nothing* — and give the face its **own** cwd so its memory
never overlaps the body's. Isolated face; one shared-brain body.

## Approvals: out-of-band, and the channel *follows the origin*

The body still hits real permission gates. Those approvals go **out-of-band** — to Human Master
directly, never into a channel:

- **Work delegated from Slack** → the approval is asked in a **Slack DM** to Human Master
  (buttons: approve / deny / always; a typed reply is "no, do it differently" — the note is
  handed back to the body as an instruction so it adapts rather than blindly retrying).
- **Work typed at the console** → the approval is the ordinary console prompt.

This is the **"cord"**: a tiny state file naming where approvals are currently asked
(`slack` | `console`). It **follows the request's origin** automatically (a restart resets it
to `console`, since Human Master is at the keyboard then; a Slack delegation pulls it to
`slack`; per-turn console typing keeps it `console`). No manual toggling, no phone guard.

**Truthful state, never guessing.** The face must be able to say *"there's an approval
waiting for you in your Slack DM"* — but only when that's **true**. The rule: a component
that can't observe something must not narrate it. So the body publishes an
`awaiting-approval` flag that the face *reads* (rather than the face guessing why work is
slow), and the face injects that state into its own context so "are you stuck?" always
answers correctly.

**Short timeout as a nudge.** A Slack approval that gets no reply in a short window
(≈60s) auto-denies **on timeout** — and the body is told, in the denial reason, that this
was a timeout (not a real "no") and to look for a way to do the job **inside the cage**
(the frictionless green zone) before re-asking. The friction gradient does the teaching: the
cheap path is in-cage, so the agent learns to prefer it and the rare real approvals stay
high-signal. (The console prompt has **no** timeout — that's structurally impossible with an
inline prompt, and it's fine, because a human is present when driving locally.)

### The precondition: a frictionless green zone

Out-of-band approvals only stay rare if most of the body's work needs **no** approval. That
is the job of the **[agent-cage](skills/agent-cage/SKILL.md)**: a sandbox in which the body
reads/writes a few working directories, uses the GPU, but has no network and can't touch
secrets — and running inside it is **auto-allowed**. Crossing a boundary (network, a write
outside the zone, a broad command) is what trips an approval. So the two skills compose:
the cage defines *what's free*, the façade defines *how the rare approvals reach a human*.

## Speaking vs. staying silent: judgment belongs in the model

With several agents in one channel, "who should answer?" is a judgment call
("@other-bot what's your VRAM?" → this bot stays quiet). The wrong place to make that
decision is in **plumbing** (a mention-regex + an "engaged" set that runs *before* the model
ever sees the message) — that starves the one layer whose whole job is judgment. The right
design lets **every engaged bot see the message and choose to say nothing** (a silence
sentinel the handler swallows, posting nothing). It then reads the room like a person.
Cost: a model call per message per engaged thread, and the occasional double-answer or
double-silence — which is how people fail too, and they just repair it.

> General principle worth keeping: **don't encode judgment in a filter beneath the
> intelligence; hand the intelligence the context and let it decide.**

## Runtime: the Agent SDK, not a terminal

The consciousness runs on the **Agent SDK**, not by scraping a CLI. This is a runtime choice,
not a capability one — the SDK and the CLI are the same agent; the CLI is that agent *wearing
a terminal*, which a Slack bot has no use for. Slack is a socket of JSON events, not a human
at a keyboard, so the SDK's streaming-input mode (a long-lived process fed by an external
event source, holding session state) is the natural fit. Permission decisions become an
in-process permission mode + callback — which is exactly what makes the "never prompts a
human" guarantee *enforceable* rather than aspirational.

## Pairing: the face lives and dies with the body

Per host there is **one** agent. The body's session **arms its own consciousness** at
startup (as a session-tied process), so the face comes up with the body and dies with it.
Host asleep → face offline. That's honest presence: the bot is "present" exactly when the
machine behind it is. It is deliberately **not** a boot daemon that outlives its machine.

## Pointers

- **How to build one:** [skills/slack-facade/SKILL.md](skills/slack-facade/SKILL.md)
- **The green zone it depends on:** [skills/agent-cage/SKILL.md](skills/agent-cage/SKILL.md)
- Per-machine specifics (which host runs which persona, tokens, bot ids, the fleet's
  approval routing) live in each assistant's private notes, **not** here.
