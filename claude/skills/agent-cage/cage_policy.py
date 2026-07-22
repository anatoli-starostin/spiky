#!/usr/bin/env python3
"""The fundamental permission constraint — the CAGE. Transport-agnostic.

This module answers ONE question about a gated tool call: may it run WITHOUT a
human, or must a human be asked? It knows nothing about the approval transport, guards,
or how approval is delivered — those are transport concerns layered on top by
`permission_gate.py`. If every transport were removed, this policy still stands.

  classify(tool, tool_input) -> "ungated" | "green" | "gated"
    ungated : not a gated tool (Write/Edit/…) -> caller defers to Claude Code.
    green   : the safe zone -> auto-allow, no human.
    gated   : everything else -> a human must approve (caller routes it).

The green zone = read-only commands + a single clean `sbox <argv>` (the ML cage)
+ the scoped-safe `git` surface. Everything else is gated. The design rule is
"auto-allow only the trivially-simple / known-safe; gate on anything else" — the
failure mode is always toward asking, never toward silent broad execution.
"""
import os
import re
import shlex

GATED_TOOLS = {"Bash", "Skill", "Agent", "Workflow"}
FILE_TOOLS = {"Write", "Edit", "NotebookEdit"}

# Read-only / no-side-effect tools that never need a human -> stay UNGATED (deferred
# to Claude Code, which auto-approves them). The list is the ONLY escape hatch: any
# tool NOT here and not green is GATED, so in slack mode EVERY real permission
# (WebFetch, WebSearch, MCP tools, and any future tool) is asked in Slack -- not
# silently sent to the native console prompt. Human Master's rule: "all permissions should
# be asked on Slack in slack mode." Failure mode stays toward asking, per the cage.
# (In console mode a gated tool still just defers to the native prompt -- unchanged.)
SAFE_TOOLS = {
    "Read", "Glob", "Grep", "LS", "NotebookRead", "TodoWrite", "TodoRead",
    "BashOutput", "KillShell", "KillBash", "ExitPlanMode",
}

# Directory holding the green-listed helper scripts (body_bridge.py, gh_issue.py). They are
# trusted by ABSOLUTE PATH, so they MUST live where the sbox cage mounts READ-ONLY — else the
# caged agent could rewrite a trusted helper and have it still auto-run green (see the
# agent-cage SKILL). This is the one knob for that location: the directory NAME is incidental,
# the read-only-in-cage property is the requirement. Override with AGENT_TOOLS_DIR to deploy
# the helpers elsewhere.
_TOOLS_DIR = os.path.expanduser(os.environ.get("AGENT_TOOLS_DIR", "~/work"))

# Writable green zone for the file tools — mirrors the sbox writable binds. A file
# write INSIDE these auto-allows (frictionless, like the cage); anywhere else is
# gated (routed to the approval channel).
_WRITABLE = [os.path.realpath(os.path.expanduser(p)) for p in ("~/projects", "/tmp", "~/.cache")]

def _in_green(path):
    """True iff `path` is an ABSOLUTE path inside a writable green-zone dir.
    Relative / unknown paths are conservatively NOT green (they get gated)."""
    if not path:
        return False
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        return False
    try:
        rp = os.path.realpath(expanded)
    except Exception:
        return False
    return any(rp == b or rp.startswith(b + os.sep) for b in _WRITABLE)

# ── read-only safe-prefix allowlist ─────────────────────────────────────────
# Every top-level segment of the shell line must match one of these to auto-allow.
SAFE_BASH_PATTERNS = [
    r"^\s*ls(\s|$)", r"^\s*cat\s", r"^\s*head(\s|$)", r"^\s*tail(\s|$)",
    r"^\s*grep\s", r"^\s*egrep\s", r"^\s*rg\s", r"^\s*find\s",
    r"^\s*ps(\s|$)", r"^\s*pgrep(\s|$)", r"^\s*df(\s|$)", r"^\s*du\s",
    r"^\s*free(\s|$)", r"^\s*uptime(\s|$)", r"^\s*uname(\s|$)",
    r"^\s*ping\s", r"^\s*ip\s", r"^\s*ss(\s|$)", r"^\s*nvidia-smi",
    r"^\s*sensors(\s|$)", r"^\s*wakeonlan\s", r"^\s*tailscale\s+(status|ip|ping)",
    r"^\s*systemctl\s+(status|is-active|is-enabled|list-units|show)",
    r"^\s*journalctl\s",   # NOTE: git is handled entirely by is_safe_git(), not here
    r"^\s*which\s", r"^\s*echo\s", r"^\s*pwd(\s|$)", r"^\s*awk\s",
    r"^\s*sed\s+-n", r"^\s*wc(\s|$)", r"^\s*sort(\s|$)", r"^\s*uniq(\s|$)",
    r"^\s*jq\s", r"^\s*column\s", r"^\s*paste\s", r"^\s*cut\s", r"^\s*tr\s",
    r"^\s*basename\s", r"^\s*dirname\s", r"^\s*realpath\s", r"^\s*date(\s|$)",
    r"^\s*hostname(\s|$)", r"^\s*whoami(\s|$)", r"^\s*id(\s|$)",
    r"^\s*mkdir\s+-p", r"^\s*chmod\s+\+x\s", r"^\s*kill\s+-0\s",
    r"^\s*until\s", r"^\s*\[\s+", r"^\s*test\s", r"^\s*sleep\s",   # sleep = harmless wait
]

# ── body-report special case (verbatim from the original hook) ──────────────
# The body filing a delegated task's result: an ABSOLUTE-path body_bridge call,
# optionally feeding a QUOTED heredoc (report text is literal DATA, not shell).
# Matched as a WHOLE line so the apostrophe/pipe-laden report never reaches shlex.
# body_bridge.py lives in the trusted-tools dir (_TOOLS_DIR); escape it into the pattern.
_BODY_BRIDGE = f"{_TOOLS_DIR}/slack-facade/body_bridge.py"
_BODY_HEAD = re.compile(
    r"^\s*(\S+/)?python3?\s+"
    + re.escape(_BODY_BRIDGE)
    + r"\s+(done|error|running)\s+\S+\s*"
    + r"(<<-?\s*(['\"])([A-Za-z_][A-Za-z0-9_]*)\4\s*)?$"
)

def _is_body_report(command):
    lines = command.split("\n")
    mm = _BODY_HEAD.match(lines[0])
    if not mm:
        return False
    delim = mm.group(5)
    if not delim:
        return len(lines) == 1 or not "".join(lines[1:]).strip()
    body = lines[1:]
    while body and not body[-1].strip():
        body.pop()
    return bool(body) and body[-1].strip() == delim


# ── shell segmentation (top-level, quote-respecting, operator-aware) ─────────
# punctuation_chars=True makes shlex ISOLATE the operators ();<>|& into their own
# tokens even when attached to a word (foo|curl -> foo | curl), while still keeping
# operators INSIDE quotes literal ('a|b' stays one token). That reliability is what
# lets us trust segment boundaries — plain shlex.split merges foo|curl into one token.
_SEPS = {"|", "||", "&&", ";", "&"}

def _segments(command):
    """Split a shell line into top-level command segments (token lists).
    Returns None if unparseable."""
    cmd = command.strip()
    if not cmd:
        return []
    try:
        lx = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        tokens = list(lx)
    except ValueError:
        return None
    if not tokens:
        return []
    parts, cur = [], []
    for t in tokens:
        if t in _SEPS:
            if cur:
                parts.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    return parts

def base_tokens(command):
    """First token of each top-level segment: 'touch x && chmod y' -> ['touch','chmod']."""
    segs = _segments(command) or []
    return [seg[0] for seg in segs if seg]


# ── quote-aware escape-operator scanner ─────────────────────────────────────
def _read_redirect(s, i):
    """At s[i] = start of a redirect operator run (< > &), parse operator + target.
    Returns (next_index, safe) where safe is True iff the redirect writes NOTHING
    real — target is /dev/null, or it's a file-descriptor dup (2>&1, >&2). Any
    redirect to a real path is unsafe."""
    n = len(s)
    j = i
    while j < n and s[j] in "<>&":
        j += 1
    op = s[i:j]
    k = j
    while k < n and s[k] in " \t":
        k += 1
    # read the target word (handle a quoted target)
    if k < n and s[k] in "'\"":
        q = s[k]; k += 1; start = k
        while k < n and s[k] != q:
            if q == '"' and s[k] == "\\" and k + 1 < n:
                k += 2; continue
            k += 1
        target = s[start:k]
        if k < n:
            k += 1
    else:
        start = k
        while k < n and s[k] not in " \t\n<>|&;'\"`":
            k += 1
        target = s[start:k]
    safe = target == "/dev/null" or ("&" in op and target.isdigit())
    return k, safe


def _escape_ops(s):
    """True if the command has, at the OUTER shell level, a REDIRECT to a real path,
    a COMMAND SUBSTITUTION ($(...) or `...`), or a BACKGROUND '&' — the operators
    that let a wrapped command write outside the cage or spawn an uncaged process.
    Honors quoting exactly as bash does:
      single quotes -> everything literal;
      double quotes -> $(...) and `...` still active, but < > & literal;
      unquoted      -> redirects, `, $( and a lone & are active.
    HARMLESS outer redirects — to /dev/null or fd-dups like 2>&1 — are allowed
    (they write nothing real and are ubiquitous). Pipes / ; / && / || are NOT
    flagged here (reliable segmentation handles those). Unbalanced quotes -> True.
    So `sbox foo > outside` gates, but `sbox foo 2>/dev/null` and
    `sbox bash -c '... > in_cage'` (redirect inside quotes) stay green."""
    i, n, state = 0, len(s), None  # state: None | "'" | '"'
    while i < n:
        c = s[i]
        if state == "'":
            if c == "'":
                state = None
            i += 1; continue
        if state == '"':
            if c == "\\" and i + 1 < n:
                i += 2; continue
            if c == '"':
                state = None; i += 1; continue
            if c == "`":
                return True
            if c == "$" and i + 1 < n and s[i + 1] == "(":
                return True
            i += 1; continue
        if c == "\\" and i + 1 < n:
            i += 2; continue
        if c == "'":
            state = "'"; i += 1; continue
        if c == '"':
            state = '"'; i += 1; continue
        if c == "`":
            return True
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            return True
        if c in "<>":                       # redirect: safe only to /dev/null or fd-dup
            k, safe = _read_redirect(s, i)
            if not safe:
                return True
            i = k; continue
        if c == "&":
            if i + 1 < n and s[i + 1] == "&":   # && chain -> segmentation handles it
                i += 2; continue
            if i + 1 < n and s[i + 1] == ">":   # &> redirect (both streams)
                k, safe = _read_redirect(s, i)
                if not safe:
                    return True
                i = k; continue
            return True                          # lone & -> background
        i += 1
    return state is not None


# ── the sbox cage tier ──────────────────────────────────────────────────────
def is_safe_sbox(command, segs):
    """A single, simple `sbox <argv>`. (_escape_ops is checked separately by the
    caller, so redirects / substitution / background are already excluded.)"""
    if segs is None or len(segs) != 1:
        return False
    seg = segs[0]
    return len(seg) >= 2 and seg[0] == "sbox"


# ── the scoped-safe git tier ────────────────────────────────────────────────
SAFE_GIT_SUBS = {
    "status", "diff", "log", "show", "branch", "remote", "reflog", "stash",
    "add", "commit", "checkout", "switch", "restore", "fetch", "pull", "push",
    "rev-parse", "describe", "tag", "merge", "rebase", "cherry-pick", "mv", "worktree",
}
# a token naming a network endpoint: scheme://…, ext::…, file://…, or scp user@host:path
_URL_RE = re.compile(r"://|^ext::|^file://|^[\w.+-]+@[\w.-]+:")
_GIT_OPT_WITH_ARG = {"-C", "--git-dir", "--work-tree", "--namespace"}

def is_safe_git(seg):
    """git limited to the everyday surface, with the code-exec / exfil sharp edges
    excluded: no `-c`, no config writes, no --exec/--upload-pack/--receive-pack,
    and no explicit URL/ext::/scp remote (those force an approval)."""
    if not seg or seg[0] != "git" or len(seg) < 2:
        return False
    i, sub = 1, None
    while i < len(seg):
        t = seg[i]
        if t == "-c" or t.startswith("-c") or t.startswith("--exec") \
           or t.startswith("--upload-pack") or t.startswith("--receive-pack"):
            return False
        if t in _GIT_OPT_WITH_ARG:
            i += 2; continue
        if t.startswith(("--git-dir=", "--work-tree=", "--namespace=")) \
           or t in ("-P", "--no-pager", "--paginate", "--no-replace-objects"):
            i += 1; continue
        if t.startswith("-"):
            i += 1; continue
        sub = t
        break
    if sub not in SAFE_GIT_SUBS:
        return False
    for t in seg[1:]:
        if _URL_RE.search(t):
            return False
    return True


# ── the scoped-safe GitHub issue/PR tier ────────────────────────────────────
# One vetted helper — gh_issue.py — is the sanctioned surface for reading/writing
# GitHub issues (and opening PRs) on the spiky repo. Greenlighting it by absolute
# path is exactly the _is_body_report precedent: a fixed, bounded capability the
# owner has approved, so it needs no per-call approval despite reaching the network.
# The script itself only calls the GitHub API with the spikybot PAT; it cannot exec
# arbitrary code, so the green surface stays narrow. Bodies come via --body-file, so
# no heredoc / substitution ever rides the command line.
GH_ISSUE_SCRIPT = f"{_TOOLS_DIR}/gh-issue/gh_issue.py"
_PY = {"python", "python3"}

def is_safe_gh_issue(seg):
    """True for `python3 <abs>/gh_issue.py …` or a direct `<abs>/gh_issue.py …`."""
    if not seg:
        return False
    if seg[0] == GH_ISSUE_SCRIPT:
        return True
    return len(seg) >= 2 and os.path.basename(seg[0]) in _PY and seg[1] == GH_ISSUE_SCRIPT


# ── the composed green-zone test for Bash ───────────────────────────────────
def _seg_ok(seg):
    """A single command segment is green if it's a known read-only command, the
    scoped-safe git surface, or the vetted gh_issue.py helper."""
    if any(re.match(p, " ".join(seg)) for p in SAFE_BASH_PATTERNS):
        return True
    return is_safe_git(seg) or is_safe_gh_issue(seg)

def is_safe_bash(command):
    if _is_body_report(command):
        return True
    segs = _segments(command)
    if segs is None:
        return False
    if not segs:
        return True
    # No redirect / command substitution / background may ride along on an
    # auto-allowed command (those escape the cage or spawn uncaged work). Pipes and
    # chains are allowed — but only because every segment is checked independently.
    if _escape_ops(command):
        return False
    # A single clean `sbox <argv>` — the ML cage.
    if is_safe_sbox(command, segs):
        return True
    # Otherwise every top-level segment must itself be green (read-only or scoped git).
    return all(_seg_ok(seg) for seg in segs)


# ── the one entry point the orchestrator calls ──────────────────────────────
def classify(tool, tool_input):
    """The fundamental cage decision. See module docstring."""
    ti = tool_input or {}
    if tool in FILE_TOOLS:
        # A write is green only inside the writable green zone; anywhere else asks.
        path = ti.get("file_path") or ti.get("notebook_path")
        return "green" if _in_green(path) else "gated"
    if tool == "Bash":
        cmd = ti.get("command", "") or ""
        return "green" if is_safe_bash(cmd) else "gated"
    if tool in SAFE_TOOLS:
        return "ungated"          # read-only -> defer to Claude Code (auto-approves)
    # EVERYTHING else -> gated. Skill/Agent/Workflow AND anything not known-safe
    # (WebFetch, WebSearch, MCP tools, future tools) reaches outside the cage's
    # trivially-safe set, so a human must decide -> the caller routes it (Slack in
    # slack mode; native console prompt in console mode). This is the fix for network
    # tools like WebFetch silently escaping to the console instead of asking in Slack.
    return "gated"
