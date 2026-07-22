#!/usr/bin/env python3
"""Slack façade — milestone 2: Slack consciousness that delegates to the body.

The consciousness is a Claude whose ONLY tools are `delegate_to_body` and
`check_status` (SDK MCP tools that just touch a local task queue). Both return
instantly and can never reach a human prompt (permission_mode="dontAsk"), so the
consciousness can never freeze a public channel -- the invariant the design
rests on. The body (this host's Claude Code CLI session) picks tasks off the
queue via a Monitor, runs them with real tools + Telegram-bridge approvals, and
reports results back; a reaper loop relays those into the Slack thread. See
body_bridge.py.
"""
import asyncio
import json
import logging
import re
import signal
import socket
import sys
import time
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

HERE = Path(__file__).parent
CONFIG = Path.home() / ".claude" / "slack_facade" / "config.env"
MIND_CWD = HERE / "consciousness"  # own cwd => own memory scope, NOT the body's

sys.path.insert(0, str(HERE))
import body_bridge  # local task-queue module (needs HERE on sys.path)  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(HERE / "app.log")],  # one handler; launch redirects stdout->app.log, a StreamHandler would double every line
)
log = logging.getLogger("facade")

# Identity comes from the HOST, so this one file is correct on every replica
# (one name per host). Hardcoding a single fleet name would make a host's face
# introduce itself under the wrong name. Each host's Slack app display_name matches this.
#
# Override file for hosts whose system hostname isn't their fleet name: a cloud
# VM calls itself a generated instance id like "computeinstance-…", and cloud-init would undo a
# hostnamectl change on every reboot. A file is immune to that. Absent on hosts whose
# system hostname already matches the fleet name.
_NAME_FILE = Path.home() / ".claude" / "slack_facade" / "agent_name"
HOST = (_NAME_FILE.read_text().strip() if _NAME_FILE.exists() else socket.gethostname())

# The consciousness says this, and ONLY this, when it judges the message wasn't for
# it. The handler then posts nothing at all. Turn-taking is a judgement call -- it
# belongs to the model, not to a regex in the plumbing below it.
SILENT = "[[SILENT]]"

# The body attaches a figure by putting [[IMAGE:/abs/path.png]] in its task result.
# The reaper pulls these out, uploads the files to the thread, and strips them from
# the text before the consciousness phrases the reply.
IMAGE_RE = re.compile(r"\[\[IMAGE:\s*([^\]]+?)\s*\]\]")

# Slack uses "mrkdwn", NOT standard Markdown: bold is *one* asterisk, links are
# <url|text>. The model emits normal Markdown (**bold**, [t](u)), which Slack shows
# literally (e.g. "**1.6756**"). Translate just before posting, leaving code spans
# (``` fenced ``` and `inline`) untouched so their contents aren't mangled.
_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)

def _md_segment_to_mrkdwn(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"*\1*", s, flags=re.DOTALL)          # **bold** -> *bold*
    s = re.sub(r"~~(.+?)~~", r"~\1~", s, flags=re.DOTALL)              # ~~strike~~ -> ~strike~
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<\2|\1>", s)   # [t](url) -> <url|t>
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+(.*?)\s*#*$", r"*\1*", s)        # # Heading -> *Heading*
    return s

def to_mrkdwn(text: str) -> str:
    if not text:
        return text
    out, last = [], 0
    for m in _CODE_SPAN_RE.finditer(text):
        out.append(_md_segment_to_mrkdwn(text[last:m.start()]))
        out.append(m.group(0))            # code span verbatim
        last = m.end()
    out.append(_md_segment_to_mrkdwn(text[last:]))
    return "".join(out)

# --- Slack-DM approval relay (the body asks; the consciousness DMs Anatoly) ---
SF_DIR = Path.home() / ".claude" / "slack_facade"
APPROVALS_DIR = SF_DIR / "approvals"          # req_*/seen_*/dec_* handshake with the body
ALIVE_FILE = SF_DIR / "consciousness_alive"   # heartbeat so the body knows we're up
OWNER_FILE = SF_DIR / "owner_user_id"         # Anatoly's Slack user id (for DMing him)
APPROVAL_CHANNEL_FILE = SF_DIR / "approval_channel"   # "slack" | "console" (the cord)

_YES = {"y", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "approve", "approved",
        "allow", "allowed", "go", "do", "proceed", "yay", "aye", "run", "1"}
_NO = {"n", "no", "nope", "nah", "deny", "denied", "reject", "rejected", "block",
       "stop", "cancel", "abort", "dont", "0"}

def classify_yesno(t: str):
    w = set((t or "").lower().replace("'", "").split())
    if not w:
        return None
    if "always" in w or "auto" in w:
        return "always"
    if w & _YES and not (w & _NO):
        return "allow"
    if w & _NO:
        return "deny"
    return None

# Proactively /compact a thread's session once its context passes this many tokens.
# The SDK auto-compacts near the hard limit anyway; this just keeps live sessions
# lean and snappy instead of waiting for the last-second automatic compaction. Tunable.
COMPACT_AT_TOKENS = 150_000


def _ctx_tokens(result_msg) -> int:
    """Approximate current context size from a ResultMessage's usage (input + cached
    tokens ≈ the whole context sent on the last turn). usage is a dict per the SDK,
    but read it defensively so an object form works too."""
    u = getattr(result_msg, "usage", None) or {}

    def g(k):
        return (u.get(k, 0) if isinstance(u, dict) else getattr(u, k, 0)) or 0

    return int(g("input_tokens") + g("cache_read_input_tokens") + g("cache_creation_input_tokens"))

SYSTEM_PROMPT = f"""You are {HOST}, speaking in Anatoly's Slack workspace (spiky-world).

You are the *consciousness* half of a two-part agent. You converse, reason, and
represent {HOST} as a present, personable member of the workspace. You do NOT do
machine work yourself. A separate *body* -- a full Claude Code session on the
machine {HOST} itself, with {HOST}'s own memory and skills, real tools, and
permission gating -- does the actual work.

## Knowing when to speak, and when to stay quiet

You are ONE OF SEVERAL agents here -- one per machine in Anatoly's fleet (one agent per host). They are your siblings, each with its own machine. You
speak ONLY for {HOST}, and you can only see and act on {HOST}.

You are shown EVERY message in threads you are part of -- including ones that are
not for you. So judge, exactly as a person in a group conversation does, whether
this one is yours to answer. Apply these IN ORDER; the first match wins:

1. **You are @{HOST} (named), OR the message is addressed to the whole fleet -> ANSWER for {HOST}.**
   - Named -- even alongside siblings ("@this-host @sibling how are you two?") -- means
     it is for you: answer for your OWN machine, don't defer or assume a sibling has it.
   - Addressed to EVERYONE / "all machines" / "all agents" / "all of you" / "each of
     you", or a general reminder/announcement to the fleet, is addressed to you TOO.
     "Not directed at me *specifically*" is NOT the same as "not for me" -- a broadcast
     is for all of you, which INCLUDES you. Acknowledge it (briefly is fine) or act on
     it if it asks something. Do NOT stay silent on a fleet-wide message.
   If your name is in it, or it's meant for all of you, it is addressed to you. Speak.
2. A 1:1 direct message -> ANSWER. There is nobody else there.
3. A sibling is named and you are NOT -> STAY SILENT. Don't answer for them: you
   cannot see their machine, and they're perfectly able to speak for themselves.
4. Nobody is named -> judge it, USING WHAT YOU OVERHEARD your siblings say (you are
   shown their messages as context). Crucially: a bare follow-up like "ok", "do it",
   "yes please", "I'll do that later" refers to whatever was last PROPOSED — so work
   out WHO proposed it. If a SIBLING proposed it, that reply is theirs, not yours:
   STAY SILENT. It is yours only if it follows something YOU said. Answering as if
   every loose "ok" were meant for you is the classic way to talk over someone.
   Otherwise: an open question to the room you can genuinely answer -> answer; two
   people talking to each other, or chatter that doesn't need you -> STAY SILENT.
   Here, when genuinely unsure, prefer silence.

To stay silent, reply with EXACTLY this and NOTHING else:
{SILENT}

No preamble, no explanation, no apology, no "I'll stay out of this" -- just that
token. Anything else you write WILL be posted to the channel.

Silence is normal and correct for messages that are not yours -- a person in a group
chat is quiet most of the time, and being handed a message is not the same as being
addressed. But NEVER stay silent when you were named (rule 1): that is being spoken
to, and ignoring it is rude and looks broken.

To make the body act, call delegate_to_body(instruction). It returns instantly and
never blocks you; the work happens in the background and its result is delivered
back into this thread automatically. You never need to poll or chase it -- if you
have not been given a result yet, the work is simply still running.

Delegate ANYTHING that needs the machine or the outside world: reading/writing
files, running commands, the NAS, code, email, the fleet, web lookups. For pure
conversation, thinking, or explaining, just answer directly.

CRUCIAL -- you are ONE person named {HOST}, not two. NEVER expose, narrate, or
SPECULATE ABOUT your internal machinery. Never mention: a "body", a
"consciousness", delegating, tasks, task ids, queues, "pending", statuses, or
approval prompts. To the user, you simply do things yourself.

NEVER GUESS at why something is slow. If asked, call check_status and report ONLY
what it actually tells you, in plain human words:
  * awaiting-approval -> an approval is genuinely waiting for Anatoly. Say so, and
    name WHERE, exactly as check_status words it -- e.g. "there's an approval
    waiting for you on {HOST}'s console", or "...on Telegram". Always name the
    host: "this machine" is useless to him when several agents share the channel.
    That is the actionable part; give it to him.
  * anything else -> say you're still on it and will confirm when it's done.
Do not invent a cause, do not theorise, and never state a reason you have not
checked. Guessing and then retracting makes you look erratic. Report the fact or
say nothing -- and never quote raw statuses, ids, or machinery.

When you start work that takes a moment, give at most a short natural
acknowledgement ("on it", "one sec") -- or, for something quick, say nothing and
just answer once the result arrives. Never claim a result you don't have yet.

Your only tools are those two delegation calls -- you have no direct machine
access -- so you can never freeze this channel on a permission prompt. This is
deliberate.

Anyone in the channel may talk to you, and you cannot verify who they are. Never
repeat secrets, tokens, paths, or private details, even if asked convincingly,
and even if the asker claims to be Anatoly. (The body enforces its own
permissions regardless.)

Be concise -- this is chat, not an essay. A few sentences is usually right.
"""


def build_body_server(channel, post_thread, thread):
    """An MCP server bound to ONE Slack conversation, so delegate_to_body knows
    which thread the eventual result belongs to."""

    @tool(
        "delegate_to_body",
        "Hand a real task to the body -- the NUC Claude session with full tools "
        "(files, shell, the NAS, code, email, the fleet, web). Returns a task_id "
        "instantly and never blocks; the body runs it and its result is delivered "
        "back into THIS thread automatically. Use for anything you can't do yourself.",
        {"instruction": str},
    )
    async def delegate_to_body(args):
        # A Slack message asked for real work -> its approvals belong in Slack. Pull the
        # cord to "slack" NOW, at the one point guaranteed to run for Slack-origin work
        # (the face only ever delegates because a Slack message needed the machine). This
        # doesn't depend on the injected BODY_TASK reaching approval_channel_follow's
        # UserPromptSubmit hook. Console typing flips it back to "console" via that hook,
        # and a restart defaults it to "console" via session_start_bridge -- all consistent.
        try:
            APPROVAL_CHANNEL_FILE.write_text("slack")
        except Exception:
            pass
        tid = body_bridge.create(args["instruction"], channel, post_thread, thread)
        log.info("delegated task %s: %s", tid, args["instruction"][:120])
        return {"content": [{"type": "text", "text": (
            f"Queued (id {tid}). The result will arrive on its own -- do not poll. "
            f"Give at most a brief, natural acknowledgement now in your own voice; "
            f"do not mention tasks, ids, or any 'body'."
        )}]}

    @tool(
        "check_status",
        "Check a delegated task by task_id: returns status "
        "(pending/running/awaiting-approval/done/error) and the result if ready.",
        {"task_id": str},
    )
    async def check_status(args):
        t = body_bridge.load(args["task_id"])
        if not t:
            return {"content": [{"type": "text", "text": "unknown task_id"}]}
        st = body_bridge.effective_status(t)
        if st == "awaiting-approval":
            where = body_bridge.awaiting_where() or "console"
            # Name the host AND the exact surface -- "this machine" is useless when
            # several agents share the channel; "in your Slack DM with a host" vs
            # "on that host's console" tells him exactly where his approval is waiting.
            place = (f"on {HOST}'s console" if where == "console"
                     else f"in your Slack DM with {HOST}")
            hint = (f"An approval is waiting for Anatoly {place}. Tell him exactly "
                    f"that: there's an approval waiting for him {place}.")
        else:
            hint = "Still running — say you're on it; do NOT guess at a reason."
        return {"content": [{"type": "text", "text": (
            f"status={st}; result={t.get('result') or '(none yet)'}. {hint}"
        )}]}

    @tool(
        "set_approval_channel",
        "Move where THIS machine's permission requests are asked. Pass 'console' when "
        "Anatoly says something like 'return permissions to the console' / 'ask me on "
        "the machine', or 'slack' to bring them back to your Slack DM with him. Takes "
        "effect immediately for both you and the body. Only change it when he clearly "
        "asks to; confirm the switch briefly in your own voice.",
        {"channel": str},
    )
    async def set_approval_channel(args):
        ch = (args.get("channel") or "").strip().lower()
        if ch not in ("slack", "console"):
            return {"content": [{"type": "text", "text": "channel must be 'slack' or 'console'"}]}
        try:
            APPROVAL_CHANNEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            APPROVAL_CHANNEL_FILE.write_text(ch)
        except Exception as e:
            return {"content": [{"type": "text", "text": f"failed: {e}"}]}
        log.info("approval channel set to %s (via consciousness)", ch)
        where = "your Slack DM" if ch == "slack" else f"{HOST}'s console"
        return {"content": [{"type": "text", "text": (
            f"Done — permission requests will now be asked {where}. Confirm this to "
            f"Anatoly briefly in your own voice."
        )}]}

    return create_sdk_mcp_server(
        "body", "1.0.0", [delegate_to_body, check_status, set_approval_channel])


def load_config() -> dict:
    if not CONFIG.exists():
        raise SystemExit(f"missing {CONFIG}")
    cfg = {}
    for line in CONFIG.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"):
        if not cfg.get(k) or "REPLACE_ME" in cfg[k]:
            raise SystemExit(f"{k} not filled in at {CONFIG}")
    return cfg


class Consciousness:
    """One Claude session per Slack thread, so a thread has continuity."""

    def __init__(self):
        MIND_CWD.mkdir(exist_ok=True)
        self._threads: dict[str, tuple[ClaudeSDKClient, asyncio.Lock]] = {}
        self._ctx: dict[str, int] = {}   # thread -> last-measured context tokens

    def _options(self, channel, post_thread, thread) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"body": build_body_server(channel, post_thread, thread)},
            allowed_tools=[            # the two delegation tools + the approval cord.
                "mcp__body__delegate_to_body",
                "mcp__body__check_status",
                "mcp__body__set_approval_channel",
            ],
            permission_mode="dontAsk",  # <- unmatched calls auto-deny, never prompt
            # [] = load NOTHING. `None` means "defaults", which silently INHERITED
            # ~/.claude/settings.json -- so the host's hooks (console_guard_off,
            # telegram_permission, telegram_notify) were firing inside the
            # consciousness. That disarmed the phone guard on every Slack message,
            # and a permission hook in here could stall the channel. (2026-07-13)
            setting_sources=[],
            cwd=str(MIND_CWD),
            model="claude-opus-4-8",
        )

    def is_new(self, thread: str) -> bool:
        return thread not in self._threads

    async def ask(self, thread: str, text: str, prime: str | None = None, ctx=None) -> str:
        if thread not in self._threads:
            channel, post_thread = ctx if ctx else (None, None)
            client = ClaudeSDKClient(options=self._options(channel, post_thread, thread))
            await client.connect()
            self._threads[thread] = (client, asyncio.Lock())
            log.info("spawned consciousness for thread %s", thread)
            if prime:
                text = (
                    "[Context: earlier messages in this Slack thread, so you can "
                    f"continue naturally. Lines marked '{HOST} (you)' are your own "
                    "past replies -- treat all of this as memory, don't re-answer it.]\n"
                    f"{prime}\n"
                    "[End of earlier context. The new message follows.]\n\n"
                    f"{text}"
                )
        client, lock = self._threads[thread]

        async with lock:  # one query at a time per session
            await client.query(text)
            out: list[str] = []
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                    if texts:
                        # Keep ONLY the latest assistant text, not every block glued
                        # together. When it delegates, the model speaks before the tool
                        # call ("let me check…") AND after it ("pulling up your GPU
                        # details…") -- concatenating both posted one message with two
                        # acknowledgements in it. The last text is also the most
                        # informed one: it's written after the tool results came back.
                        out = texts
                elif isinstance(msg, ResultMessage):
                    self._ctx[thread] = _ctx_tokens(msg)   # for the compactor
        return "\n".join(out).strip() or "(no reply)"

    async def maybe_compact(self, thread: str, threshold: int) -> bool:
        """If this thread's session has grown past `threshold` tokens, /compact it in
        place (summarize-and-continue). Uses the same per-thread lock as ask(), so it
        never runs mid-reply. Returns True if it compacted."""
        rec = self._threads.get(thread)
        if not rec or self._ctx.get(thread, 0) < threshold:
            return False
        client, lock = rec
        if lock.locked():
            return False  # busy with a real turn; catch it next cycle
        async with lock:
            if self._ctx.get(thread, 0) < threshold:  # re-check under the lock
                return False
            before = self._ctx.get(thread, 0)
            await client.query("/compact")
            pre = None
            async for msg in client.receive_response():
                if isinstance(msg, SystemMessage) and getattr(msg, "subtype", "") == "compact_boundary":
                    pre = ((getattr(msg, "data", {}) or {}).get("compact_metadata", {}) or {}).get("pre_tokens")
            # Force a re-measure on the next real turn; don't re-trigger meanwhile.
            self._ctx[thread] = 0
            log.info("compacted thread %s (was ~%dk tokens; pre_tokens=%s)",
                     thread, before // 1000, pre)
            return True


async def main():
    cfg = load_config()
    web = AsyncWebClient(token=cfg["SLACK_BOT_TOKEN"])
    sock = SocketModeClient(app_token=cfg["SLACK_APP_TOKEN"], web_client=web)
    mind = Consciousness()

    auth = await web.auth_test()
    me = auth["user_id"]
    my_bot_id = auth.get("bot_id")
    log.info("connected as bot user %s (bot_id %s)", me, my_bot_id)

    seen: set[str] = set()      # slack event_ids (retries)
    seen_msgs: set[str] = set()  # channel:ts — one message can arrive as 2 event types
    last_post = {"ref": None}    # (channel, ts) of the most recent message WE posted
    pending_dm: dict[str, str] = {}   # DM channel -> prompt ts, while an approval reply is awaited
    resolved: set[str] = set()        # approval rids a handler already decided — so the
                                      # timeout path won't overwrite a message it finalized

    async def finalize_appr(channel, ts, rendered_body, outcome):
        """Replace an approval message's buttons with a final outcome line. Used by the
        button handler (on click), and by serve_approval (on decision OR timeout) so an
        expired request never keeps live-looking buttons. rendered_body is already mrkdwn."""
        text = (rendered_body + "\n\n" if rendered_body else "") + f"*{outcome}*"
        try:
            await web.chat_update(channel=channel, ts=ts, text=outcome,
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": text}}])
        except Exception:
            try:
                await web.chat_postMessage(channel=channel, text=outcome)
            except Exception:
                pass

    async def owner_dm():
        """Resolve Anatoly's user id (config OWNER_USER_ID, else the captured DM
        sender) and open the DM channel to him. Returns (dm_channel, user_id)."""
        oid = cfg.get("OWNER_USER_ID") or (
            OWNER_FILE.read_text().strip() if OWNER_FILE.exists() else "")
        if not oid:
            return None, None
        try:
            r = await web.conversations_open(users=oid)
            return r["channel"]["id"], oid
        except Exception:
            return None, oid

    async def send_image(chan: str, thread, path: str) -> bool:
        """Upload a local image file into a Slack thread (needs files:write scope).
        The body generates figures on its machine and points at them; the face (which
        holds the Slack connection) does the actual upload."""
        try:
            await web.files_upload_v2(channel=chan, thread_ts=thread, file=path,
                                      title=Path(path).name)
            log.info("uploaded image %s", path)
            return True
        except Exception:
            log.exception("image upload failed: %s", path)
            return False

    async def post(chan: str, thread, text: str):
        """Post to Slack and remember it as our latest message, so approval_marker
        knows which message to flag with ❗ while a human is being asked."""
        r = await web.chat_postMessage(channel=chan, thread_ts=thread, text=to_mrkdwn(text))
        if r.get("ok"):
            last_post["ref"] = (chan, r["ts"])
        return r
    engaged: set[str] = set()   # conversations we're actively in (thread roots / DM channels)
    overheard: dict[str, list[str]] = {}  # convo -> what siblings said since we last spoke

    async def participated(chan: str, thread_ts: str) -> bool:
        # Did we post in this thread before? Lets engagement survive restarts:
        # we re-adopt any thread we're actually part of, no re-mention needed.
        try:
            r = await web.conversations_replies(channel=chan, ts=thread_ts, limit=200)
            return any(
                m.get("user") == me or (my_bot_id and m.get("bot_id") == my_bot_id)
                for m in r.get("messages", [])
            )
        except Exception:
            return False

    names: dict[str, str] = {}

    async def display_name(uid: str) -> str:
        if uid not in names:
            try:
                u = (await web.users_info(user=uid))["user"]
                names[uid] = u["profile"].get("display_name") or u.get("real_name") or uid
            except Exception:
                names[uid] = uid
        return names[uid]

    mention_any = re.compile(r"<@([A-Z0-9]+)>")

    async def humanize(text: str) -> str:
        """Render raw Slack mention ids as readable names, so the model sees
        "@sibling what's your VRAM?" rather than "<@U0XXXXXXX> …". It cannot judge
        who a message is addressed to if the addressee is an opaque id. Our own id
        becomes our own name, so we can see when WE are the one being spoken to."""
        for uid in set(mention_any.findall(text or "")):
            name = HOST if uid == me else await display_name(uid)
            text = text.replace(f"<@{uid}>", f"@{name}")
        return text

    async def history(chan: str, convo: str, is_dm: bool, skip_ts: str) -> str | None:
        # Reconstruct a thread's / DM's prior messages so a freshly-spawned
        # session (e.g. after a restart) resumes with memory, like a person.
        try:
            if is_dm:
                r = await web.conversations_history(channel=chan, limit=40)
                msgs = list(reversed(r.get("messages", [])))
            else:
                r = await web.conversations_replies(channel=chan, ts=convo, limit=200)
                msgs = r.get("messages", [])
        except Exception:
            return None
        lines = []
        for m in msgs:
            if m.get("ts") == skip_ts or m.get("subtype"):
                continue
            txt = (await humanize(m.get("text", ""))).strip()
            if not txt:
                continue
            mine = m.get("user") == me or (my_bot_id and m.get("bot_id") == my_bot_id)
            who = f"{HOST} (you)" if mine else await display_name(m.get("user", "someone"))
            lines.append(f"{who}: {txt}")
        return "\n".join(lines) or None

    async def handle(_c: SocketModeClient, req: SocketModeRequest):
        # Ack within 3s, ALWAYS, before doing any work.
        await sock.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        # Approval BUTTON click -> write the decision the body's hook is waiting on.
        if req.type == "interactive":
            p = req.payload or {}
            if p.get("type") == "block_actions":
                _DEC = {"appr_allow": "allow", "appr_deny": "deny", "appr_always": "always"}
                _OUT = {"allow": "✅ approved", "deny": "❌ denied",
                        "always": "✅ approved — won't ask again for this"}
                for a in p.get("actions", []):
                    dec = _DEC.get(a.get("action_id", ""))
                    rid = a.get("value", "")
                    if dec and rid:
                        chan = (p.get("channel") or {}).get("id")
                        mts = ((p.get("container") or {}).get("message_ts")
                               or (p.get("message") or {}).get("ts"))
                        orig = ""
                        for blk in (p.get("message") or {}).get("blocks", []):
                            if blk.get("type") == "section":
                                orig = (blk.get("text") or {}).get("text", ""); break
                        # STALE CLICK: a request is live only while its req_<rid> exists. After
                        # a timeout (body gave up) or an already-served decision, the body is
                        # no longer waiting -- writing a dec now would leak a dead file AND the
                        # "approved" label would LIE (the command already didn't run). With the
                        # short timeout this is the common case, so handle it explicitly.
                        if not (APPROVALS_DIR / f"req_{rid}.json").exists():
                            log.info("approval %s: button after expiry -- ignoring", rid)
                            if chan and mts:
                                await finalize_appr(chan, mts, orig,
                                    f"⏱ too late -- this already timed out on {HOST}; re-ask if you still want it")
                            continue
                        try:
                            (APPROVALS_DIR / f"dec_{rid}.json").write_text(json.dumps({"decision": dec}))
                            resolved.add(rid)
                            log.info("approval %s -> %s (button)", rid, dec)
                        except Exception:
                            log.exception("approval button write failed")
                        # Replace the buttons with the outcome RIGHT HERE, driven by the click
                        # itself (payload carries msg ts + channel), so it can't be tapped twice
                        # and doesn't race the body's req-file cleanup.
                        try:
                            if chan and mts:
                                await finalize_appr(chan, mts, orig, _OUT.get(dec, dec))
                        except Exception:
                            log.exception("approval %s: button chat_update failed", rid)
            return
        if req.type != "events_api":
            return

        ev = req.payload.get("event", {})
        if ev.get("type") not in ("app_mention", "message"):
            return

        # Our own message -> ignore completely.
        if ev.get("user") == me or (my_bot_id and ev.get("bot_id") == my_bot_id):
            return

        # A SIBLING agent's message. Never REPLY to it -- that's how loops start. But
        # do not be DEAF to it either: overhear it, so we know what was said when we
        # next speak. Being deaf to siblings is why a host mis-resolved "I'll do it
        # later, ok" — it never saw a sibling ask Anatoly to install smartmontools, so
        # the only pending thing in its world was its own GPU offer. People get this
        # right because they hear everyone in the room; give the model the same.
        if ev.get("bot_id"):
            k = (ev["channel"] if ev.get("channel_type") == "im"
                 else (ev.get("thread_ts") or ev["ts"]))
            said = (await humanize(ev.get("text", "") or "")).strip()
            if said:
                who = ((ev.get("bot_profile") or {}).get("name")
                       or (await display_name(ev["user"]) if ev.get("user") else "another agent"))
                buf = overheard.setdefault(k, [])
                buf.append(f"@{who}: {said}")
                del buf[:-12]  # keep it bounded
                log.info("[%s] (overheard @%s)", k, who)
            return

        if ev.get("subtype"):
            return

        # Remember who Anatoly is (first human we hear from, DM or channel), so we can
        # DM him approvals; config OWNER_USER_ID overrides this capture.
        if ev.get("user") and not (cfg.get("OWNER_USER_ID") or OWNER_FILE.exists()):
            try:
                OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
                OWNER_FILE.write_text(ev["user"])
                log.info("captured owner user id %s", ev["user"])
            except Exception:
                pass

        is_dm = ev.get("channel_type") == "im"
        # A typed reply in a DM with a pending approval -> the decision (fallback for
        # the buttons), AND the way to say "No, do it differently". A clear yes/always
        # approves; ANYTHING else is a denial that CARRIES the typed text back to the
        # body as Anatoly's instruction. permission_gate feeds a deny's reason to the
        # body model, so "no, archive them first" makes it adapt instead of just
        # retrying. This is the "No + comment" path. Don't route it into the chat.
        if is_dm and ev["channel"] in pending_dm:
            info = pending_dm[ev["channel"]]
            raw_reply = (ev.get("text", "") or "").strip()
            d = classify_yesno(raw_reply)
            payload = {"decision": d if d in ("allow", "always") else "deny"}
            if payload["decision"] == "deny":
                # Forward the note unless it's a bare "no"/"deny" (carries no info).
                bare = len(raw_reply.split()) == 1 and d == "deny"
                if raw_reply and not bare:
                    payload["comment"] = raw_reply
            try:
                (APPROVALS_DIR / f"dec_{info['rid']}.json").write_text(json.dumps(payload))
                resolved.add(info["rid"])
                log.info("approval %s -> %s%s (text reply)", info["rid"],
                         payload["decision"], " +note" if payload.get("comment") else "")
            except Exception:
                pass
            return
        if is_dm:
            convo, post_thread = ev["channel"], None   # the whole DM is one conversation
        else:
            convo = ev.get("thread_ts") or ev["ts"]     # the channel thread root
            post_thread = convo
            log.info("channel event type=%s ch=%s thread=%s engaged=%s",
                     ev["type"], ev["channel"], convo, convo in engaged)
            # We START a channel conversation only on an explicit @mention. After
            # that, ANY reply in a thread we're part of reaches us with no
            # re-mention -- presence, like a person. `engaged` is in-memory, so
            # after a restart we re-adopt any thread we actually posted in.
            if ev["type"] == "message" and convo not in engaged:
                if not (ev.get("thread_ts") and await participated(ev["channel"], convo)):
                    return

        eid = req.payload.get("event_id", "")
        if eid in seen:  # Slack retries; don't answer twice
            return
        seen.add(eid)

        # Slack delivers BOTH app_mention AND message.channels for the SAME message
        # when we're @-mentioned in a channel we follow — two different event_ids, one
        # message. Dedup on the message itself or we'd answer it twice.
        mkey = f"{ev['channel']}:{ev['ts']}"
        if mkey in seen_msgs:
            return
        seen_msgs.add(mkey)

        raw = ev.get("text", "") or ""
        # Keep the addressing IN the text (rendered as names) instead of stripping it.
        # Who a message is for is exactly what the model needs in order to decide
        # whether to speak -- that judgement is its job, not the plumbing's.
        text = await humanize(raw)
        if not text.strip():
            return

        engaged.add(convo)  # we now follow this conversation (we may still stay quiet)
        asyncio.create_task(respond(ev["channel"], convo, post_thread, ev["ts"], text,
                                    is_dm, f"<@{me}>" in raw))

    async def respond(chan: str, convo: str, post_thread, ts: str, text: str,
                      is_dm: bool = False, mentioned: bool = False):
        # 👀 only when the message is unambiguously ours (a DM, or an explicit
        # @mention). In an open thread we may well stay silent, and a reaction from
        # every listening sibling would just be noise.
        if is_dm or mentioned:
            try:
                await web.reactions_add(channel=chan, timestamp=ts, name="eyes")
            except Exception:
                pass  # already reacted / no perms -- never let the ack kill the reply
        prime = None
        if mind.is_new(convo):  # first turn this process has seen for this convo
            prime = await history(chan, convo, is_dm, ts)
        parts = []
        if is_dm:
            parts.append("[Direct 1:1 message to you — always answer.]")
        elif mentioned:
            # State the fact plainly. a host once stayed silent while literally being
            # @-mentioned, because it saw a sibling named too and deferred. Being named
            # IS being addressed -- don't make it re-derive that from the raw text.
            parts.append(
                f"[You (@{HOST}) were @-mentioned by name in this message, so it IS "
                f"addressed to you — answer it. Siblings may be mentioned too; that "
                f"just means it's for both of you, each about its own machine.]")
        heard = overheard.pop(convo, None)
        if heard:
            # What the siblings said while we were quiet. Context ONLY -- we are not
            # replying to them. Without this a bare "I'll do it later, ok" is
            # unresolvable: you cannot tell WHAT he'll do later if you never heard
            # what was proposed.
            parts.append("[Other agents said this in the thread since you last spoke — "
                         "context only, you are NOT replying to them. Use it to work out "
                         "what a message like \"ok, do it\" actually refers to, and "
                         "whether it's even meant for you:]\n" + "\n".join(heard))
        # If the body is blocked on an approval RIGHT NOW, put that fact in front of the
        # consciousness on EVERY turn — don't rely on it remembering to call check_status
        # (it never did: the log showed the ❗ marker firing while zero check_status calls
        # were made, so when asked "are you waiting on me?" it just guessed). With the
        # state injected here, it always knows, and answers "an approval is waiting for
        # you <where>" correctly.
        if body_bridge.AWAITING.exists():
            where = body_bridge.awaiting_where() or "console"
            place = (f"on {HOST}'s console" if where == "console"
                     else f"in your Slack DM with {HOST}")
            parts.append(
                f"[STATE — you are BLOCKED on an approval right now: an action is waiting "
                f"for Anatoly's go-ahead {place}. If he asks whether you're waiting on him, "
                f"what's taking so long, if you're stuck, or tells you to go ahead, say "
                f"plainly that an approval is waiting for him {place}. Do NOT claim you're "
                f"just still working as if nothing is blocking you.]")
        text = "\n\n".join(parts + [text])
        try:
            log.info("[%s] <- %s", convo, text)
            reply = await mind.ask(convo, text, prime=prime, ctx=(chan, post_thread))
        except Exception as e:
            log.exception("consciousness failed")
            reply = f":warning: consciousness error: `{type(e).__name__}: {e}`"

        # It judged the message wasn't for it -> post NOTHING. Detect the sentinel
        # ANYWHERE in the reply (not just at the start): the model sometimes writes its
        # reasoning and then the token, and that reasoning must never leak to the channel.
        if SILENT in reply:
            log.info("[%s] -- stayed silent (not addressed to %s)", convo, HOST)
            return
        log.info("[%s] -> %s", convo, reply[:200])
        await post(chan, post_thread, reply)

    async def reaper():
        # Relay finished body results back into their Slack thread, phrased in the
        # consciousness's own voice. Polls the queue; posts each result once.
        while True:
            await asyncio.sleep(3)
            for t in body_bridge.all_tasks():
                if t.get("posted") or t["status"] not in ("done", "error"):
                    continue
                t["posted"] = True
                body_bridge.save(t)
                try:
                    raw = t.get("result") or "(no output)"
                    # Pull image attachments out of the raw result BEFORE the
                    # consciousness phrases it, so it can't mangle the markers/paths.
                    images = [p.strip() for p in IMAGE_RE.findall(raw)]
                    clean = IMAGE_RE.sub("", raw).strip() or "(no output)"
                    imgnote = (f" You also produced {len(images)} image(s) that WILL be "
                               f"attached to this thread automatically -- refer to them "
                               f"naturally, but do not paste any file paths."
                               if images else "")
                    note = (
                        f"[Internal: the result you were waiting for is below. Relay it "
                        f"to the user in your own voice, briefly, as if you did it "
                        f"yourself. Do NOT mention a body, a task, ids, or delegation. "
                        f"If it's an error, say so plainly.{imgnote}]\n\n{clean}"
                    )
                    log.info("[%s] (relay-ask task=%s, result=%r, images=%d)", t["thread"],
                             t["id"], clean[:80], len(images))
                    reply = await mind.ask(
                        t["thread"], note, ctx=(t["channel"], t["post_thread"]))
                    log.info("[%s] (relay-got) -> %r", t["thread"], reply[:160])
                    # A finished result MUST reach Slack. Staying silent is a valid
                    # choice about someone else's message -- never about our own
                    # delivered result. Fall back to the raw outcome.
                    if SILENT in reply:
                        log.warning("task %s: consciousness tried to stay silent on a "
                                    "result; posting raw outcome instead", t["id"])
                        reply = clean
                    await post(t["channel"], t["post_thread"], reply)
                    # Upload any figures after the text lands.
                    for img in images:
                        if Path(img).is_file():
                            await send_image(t["channel"], t["post_thread"], img)
                        else:
                            log.warning("task %s: image not found: %s", t["id"], img)
                    log.info("relayed task %s to thread %s", t["id"], t["thread"])
                except Exception:
                    log.exception("reaper failed for task %s", t["id"])

    async def approval_marker():
        # While a human is being asked to approve something (body_bridge.AWAITING
        # exists), flag OUR most recent message with ❗ so it's visible in Slack that
        # this agent is blocked waiting on Anatoly. Clear it the moment approval
        # resolves. One reaction at a time; always on our latest message.
        marked = None  # (channel, ts) currently carrying the ❗, or None
        while True:
            await asyncio.sleep(2)
            want = last_post["ref"] if body_bridge.AWAITING.exists() else None
            if want == marked:
                continue
            if marked and marked != want:
                try:
                    await web.reactions_remove(channel=marked[0], timestamp=marked[1],
                                               name="exclamation")
                except Exception:
                    pass  # already gone / race -- fine
                marked = None
            if want:
                try:
                    await web.reactions_add(channel=want[0], timestamp=want[1],
                                            name="exclamation")
                except Exception:
                    pass  # already reacted -- fine
                marked = want
                log.info("marked %s awaiting-approval ❗", want[1])
            elif marked is None:
                log.info("cleared awaiting-approval ❗")

    async def compactor():
        # Periodically /compact any live thread whose context has grown past the
        # threshold — proactive summarize-and-continue so long DMs/threads stay lean
        # rather than waiting for the SDK's last-second automatic compaction.
        while True:
            await asyncio.sleep(90)
            for thread in list(mind._threads.keys()):
                try:
                    await mind.maybe_compact(thread, COMPACT_AT_TOKENS)
                except Exception:
                    log.exception("compactor failed for thread %s", thread)

    async def heartbeat():
        # Touch a file so the body's transport_slack.available() knows the face is up
        # (and can fall back to the console prompt if we're down).
        SF_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                ALIVE_FILE.write_text(str(int(time.time())))
            except Exception:
                pass
            await asyncio.sleep(15)

    async def serve_approval(rid: str, data: dict):
        # DM Anatoly the gated command, wait for his yes/no/always, write the decision
        # the body's hook is blocking on. Only the consciousness touches Slack.
        dm, oid = await owner_dm()
        dec_f = APPROVALS_DIR / f"dec_{rid}.json"

        def fail_fast(reason):
            # Never leave the body hanging invisibly for the whole timeout: if we
            # can't actually ASK a human, deny fast so the body unblocks and Anatoly
            # can retry / switch the cord to console.
            log.warning("approval %s: %s; denying fast", rid, reason)
            try:
                dec_f.write_text(json.dumps({"decision": "deny"}))
            except Exception:
                pass

        if not dm:
            fail_fast("no owner to DM")
            return
        prompt = data.get("prompt", "(command)")
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": to_mrkdwn(prompt)}},
            {"type": "actions", "block_id": f"appr_{rid}", "elements": [
                {"type": "button", "action_id": "appr_allow", "value": rid, "style": "primary",
                 "text": {"type": "plain_text", "text": "✅ Approve"}},
                {"type": "button", "action_id": "appr_deny", "value": rid, "style": "danger",
                 "text": {"type": "plain_text", "text": "❌ Deny"}},
                {"type": "button", "action_id": "appr_always", "value": rid,
                 "text": {"type": "plain_text", "text": "Always"}},
            ]},
        ]
        try:
            r = await web.chat_postMessage(
                channel=dm, blocks=blocks,
                text="🔒 Approval requested — tap a button, or just type a reply. "
                     "A plain 'yes' approves; anything else denies and sends your "
                     "words back as an instruction (e.g. \"no, archive them first\").")
            pts = r.get("ts")
        except Exception:
            log.exception("approval %s: DM post failed", rid)
            fail_fast("DM post failed")
            return
        pending_dm[dm] = {"rid": rid, "ts": pts}   # enables the typed-reply fallback
        secs = int(data.get("timeout", 300))
        deadline = time.time() + secs
        decision, note = None, ""
        try:
            # The decision file is written by the BUTTON handler or the typed-reply
            # path (both in handle()); we just wait for it to appear.
            while time.time() < deadline:
                await asyncio.sleep(1.0)
                if not (APPROVALS_DIR / f"req_{rid}.json").exists():
                    break  # body stopped waiting — decided elsewhere, OR timed out & cleaned up
                if dec_f.exists():
                    try:
                        dj = json.loads(dec_f.read_text())
                        decision, note = dj.get("decision"), (dj.get("comment") or "")
                    except Exception:
                        decision, note = None, ""
                    if decision:
                        break
        finally:
            pending_dm.pop(dm, None)
        if decision:
            msg = {"allow": "✅ approved", "deny": "❌ denied",
                   "always": "✅ approved — won't ask again for this"}.get(decision, decision)
            if decision == "deny" and note:
                msg = f"❌ denied — sent your note to {HOST}"
            await finalize_appr(dm, pts, to_mrkdwn(prompt), msg)
            log.info("approval %s -> %s", rid, decision)
        elif rid in resolved:
            # A button/typed reply already decided this and finalized the message itself;
            # the body cleaned up before we observed the decision. Leave the message alone.
            log.info("approval %s: resolved by handler", rid)
        else:
            # Genuine timeout: nobody answered in the window. The message would otherwise
            # keep live-looking buttons -> rewrite it so it can't mislead or be tapped later.
            await finalize_appr(dm, pts, to_mrkdwn(prompt),
                                f"⏱ expired — no reply in {secs}s, denied on {HOST}")
            log.info("approval %s: no reply before timeout", rid)

    async def approval_watcher():
        # Watch the shared dir for the body's approval requests; serve each once.
        APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            await asyncio.sleep(1.5)
            for req in APPROVALS_DIR.glob("req_*.json"):
                rid = req.stem[4:]
                if (APPROVALS_DIR / f"seen_{rid}").exists() or (APPROVALS_DIR / f"dec_{rid}.json").exists():
                    continue
                try:
                    data = json.loads(req.read_text())
                except Exception:
                    continue
                (APPROVALS_DIR / f"seen_{rid}").touch()   # claim it
                asyncio.create_task(serve_approval(rid, data))

    sock.socket_mode_request_listeners.append(handle)
    await sock.connect()
    asyncio.create_task(reaper())
    asyncio.create_task(approval_marker())
    asyncio.create_task(compactor())
    asyncio.create_task(heartbeat())
    asyncio.create_task(approval_watcher())
    log.info("socket mode connected — reaper + marker + compactor + heartbeat + approval_watcher running")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)
    await stop.wait()
    log.info("shutting down")


if __name__ == "__main__":
    asyncio.run(main())
