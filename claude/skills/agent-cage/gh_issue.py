#!/usr/bin/env python3
"""gh_issue — the sanctioned GitHub issue/PR surface for the spiky repo.

Reads the spikybot PAT from ~/.config/spikybot/pat and talks to the GitHub REST
API using only the Python stdlib. Every action is attributed to the token's
identity (spikyclaudebot).

This ONE script is greenlit in cage_policy.py (`_is_gh_issue`), exactly the way
body_bridge.py's done/error/running is — so invoking it needs no approval, even
though it makes network calls outside the sbox cage. Nothing else about it is
special; the narrow, fixed capability is the point. It is deliberately preferred
over the raw `gh` CLI, which is gated: `gh` exposes api/auth/extension/gist/secret
verbs that would each be a cage bypass or exfil channel, so it stays out of the
green zone. This helper can reach ONLY the specific endpoints its code names.

Bodies (issue/comment/PR text) are passed with --body-file, never inline, so the
command line stays a single simple segment with no heredoc or $(...) — which is
what keeps it inside the green zone.

Usage:
  gh_issue.py list [--state open|closed|all] [--label L] [--limit N] [--include-prs]
  gh_issue.py view <N>
  gh_issue.py create  --title T (--body B | --body-file F) [--label L ...]
  gh_issue.py comment <N> (--body B | --body-file F)
  gh_issue.py close   <N> [--comment C | --comment-file F]
  gh_issue.py reopen  <N>
  gh_issue.py label   <N> [--add L ...] [--remove L ...]
  gh_issue.py pr-create --title T --head BRANCH [--base main] (--body B | --body-file F)
  gh_issue.py pr-list   [--state open|closed|all] [--base B] [--head H] [--limit N]
  gh_issue.py pr-view   <N>
  gh_issue.py pr-diff   <N>
  gh_issue.py pr-edit   <N> [--title T] [--body B | --body-file F] [--base BRANCH]
  gh_issue.py link-branch <N> --branch BRANCH

view / comment / close / reopen / label also accept a PR number (PRs are issues).
pr-merge is deliberately NOT provided — merging a PR stays a human action (the
branch+PR review gate); adding it here would make it frictionless, defeating review.

Default repo: anatoli-starostin/spiky (override with --repo owner/name).
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error

PAT_PATH = os.path.expanduser("~/.config/spikybot/pat")
DEFAULT_REPO = os.environ.get("GH_ISSUE_REPO", "anatoli-starostin/spiky")
API = "https://api.github.com"


def _token():
    try:
        with open(PAT_PATH) as f:
            return f.read().strip()
    except OSError as e:
        sys.exit(f"error: cannot read PAT at {PAT_PATH}: {e}")


def _request(url, method, data, accept):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {_token()}")
    req.add_header("Accept", accept)
    req.add_header("User-Agent", "gh_issue-spikybot")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        try:
            detail = json.loads(detail).get("message", detail)
        except Exception:
            pass
        sys.exit(f"error: {method} {url} -> HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"error: network failure reaching {url}: {e.reason}")


def _request_text(url, accept):
    """Like _request but returns the RAW response text (for the diff media type,
    which is not JSON). GET only; used by pr-diff."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"token {_token()}")
    req.add_header("Accept", accept)
    req.add_header("User-Agent", "gh_issue-spikybot")
    try:
        with urllib.request.urlopen(req) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        sys.exit(f"error: GET {url} -> HTTP {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        sys.exit(f"error: network failure reaching {url}: {e.reason}")


def _api(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    return _request(f"{API}{path}", method, data, "application/vnd.github+json")


def _graphql(query, variables):
    data = json.dumps({"query": query, "variables": variables}).encode()
    resp = _request(f"{API}/graphql", "POST", data, "application/vnd.github+json")
    if resp.get("errors"):
        sys.exit("error: graphql: " + json.dumps(resp["errors"]))
    return resp["data"]


def _rp(repo, suffix):
    return f"/repos/{repo}{suffix}"


def _resolve_body(inline, path, required=False, what="body"):
    if path:
        try:
            with open(path) as f:
                return f.read()
        except OSError as e:
            sys.exit(f"error: cannot read --{what}-file {path}: {e}")
    if inline is not None:
        return inline
    if required:
        sys.exit(f"error: {what} required — pass --{what} or --{what}-file")
    return None


def cmd_list(a):
    q = f"?state={a.state}&per_page={a.limit}"
    if a.label:
        q += "&labels=" + urllib.parse.quote(",".join(a.label))
    for it in _api("GET", _rp(a.repo, f"/issues{q}")):
        if "pull_request" in it and not a.include_prs:
            continue
        labels = ",".join(l["name"] for l in it.get("labels", []))
        tag = " [PR]" if "pull_request" in it else ""
        print(f"#{it['number']} [{it['state']}]{tag} {it['title']}" + (f"  ({labels})" if labels else ""))


def cmd_view(a):
    it = _api("GET", _rp(a.repo, f"/issues/{a.number}"))
    print(f"#{it['number']} [{it['state']}] {it['title']}")
    print(f"author: {it['user']['login']}   url: {it['html_url']}")
    labels = ",".join(l["name"] for l in it.get("labels", []))
    if labels:
        print(f"labels: {labels}")
    print("-" * 60)
    print(it.get("body") or "(no body)")
    for c in _api("GET", _rp(a.repo, f"/issues/{a.number}/comments?per_page=100")):
        print("-" * 60)
        print(f"@{c['user']['login']}:")
        print(c.get("body") or "")


def cmd_create(a):
    payload = {"title": a.title, "body": _resolve_body(a.body, a.body_file, required=True)}
    if a.label:
        payload["labels"] = a.label
    print(_api("POST", _rp(a.repo, "/issues"), payload)["html_url"])


def cmd_comment(a):
    body = _resolve_body(a.body, a.body_file, required=True)
    print(_api("POST", _rp(a.repo, f"/issues/{a.number}/comments"), {"body": body})["html_url"])


def cmd_close(a):
    body = _resolve_body(a.comment, a.comment_file, required=False, what="comment")
    if body:
        _api("POST", _rp(a.repo, f"/issues/{a.number}/comments"), {"body": body})
    it = _api("PATCH", _rp(a.repo, f"/issues/{a.number}"), {"state": "closed"})
    print(f"#{it['number']} -> closed")


def cmd_reopen(a):
    it = _api("PATCH", _rp(a.repo, f"/issues/{a.number}"), {"state": "open"})
    print(f"#{it['number']} -> open")


def cmd_edit(a):
    """Edit an existing issue's title / body (the `gh issue edit` surface).

    PATCH /issues/{n} touches only the issue's own title/body — the same narrow,
    bounded capability as the rest of this helper (no merge, no api/gist/secret
    verbs). Body comes via --body-file, so no heredoc rides the command line."""
    payload = {}
    if a.title is not None:
        payload["title"] = a.title
    body = _resolve_body(a.body, a.body_file, required=False)
    if body is not None:
        payload["body"] = body
    if not payload:
        sys.exit("error: edit needs at least one of --title / --body / --body-file")
    it = _api("PATCH", _rp(a.repo, f"/issues/{a.number}"), payload)
    print(f"#{it['number']} edited -> {it['html_url']}")


def cmd_label(a):
    if a.add:
        _api("POST", _rp(a.repo, f"/issues/{a.number}/labels"), {"labels": a.add})
    for l in (a.remove or []):
        _api("DELETE", _rp(a.repo, f"/issues/{a.number}/labels/{urllib.parse.quote(l)}"))
    it = _api("GET", _rp(a.repo, f"/issues/{a.number}"))
    print("labels: " + ",".join(l["name"] for l in it.get("labels", [])))


def cmd_pr_create(a):
    payload = {"title": a.title, "head": a.head, "base": a.base,
               "body": _resolve_body(a.body, a.body_file, required=False) or ""}
    print(_api("POST", _rp(a.repo, "/pulls"), payload)["html_url"])


def cmd_pr_list(a):
    q = f"?state={a.state}&per_page={a.limit}"
    if a.base:
        q += "&base=" + urllib.parse.quote(a.base)
    if a.head:
        q += "&head=" + urllib.parse.quote(a.head)
    for pr in _api("GET", _rp(a.repo, f"/pulls{q}")):
        draft = " (draft)" if pr.get("draft") else ""
        print(f"#{pr['number']} [{pr['state']}]{draft} {pr['title']}  "
              f"({pr['head']['ref']} -> {pr['base']['ref']})")


def cmd_pr_view(a):
    pr = _api("GET", _rp(a.repo, f"/pulls/{a.number}"))
    draft = " (draft)" if pr.get("draft") else ""
    print(f"#{pr['number']} [{pr['state']}]{draft} {pr['title']}")
    print(f"author: {pr['user']['login']}   {pr['head']['ref']} -> {pr['base']['ref']}")
    print(f"mergeable: {pr.get('mergeable')} ({pr.get('mergeable_state')})   url: {pr['html_url']}")
    print("-" * 60)
    print(pr.get("body") or "(no body)")
    for r in _api("GET", _rp(a.repo, f"/pulls/{a.number}/reviews?per_page=100")):
        if r.get("state") == "COMMENTED" and not (r.get("body") or "").strip():
            continue   # empty COMMENTED review = just the container for inline notes
        print("-" * 60)
        print(f"review by {r['user']['login']}: {r['state']}")
        if (r.get("body") or "").strip():
            print(r["body"])
    for c in _api("GET", _rp(a.repo, f"/pulls/{a.number}/comments?per_page=100")):
        ln = c.get("line") or c.get("original_line") or c.get("position")
        print("-" * 60)
        print(f"inline @{c['user']['login']} on {c.get('path', '?')}:{ln}")
        print(c.get("body") or "")
    for c in _api("GET", _rp(a.repo, f"/issues/{a.number}/comments?per_page=100")):
        print("-" * 60)
        print(f"@{c['user']['login']}:")
        print(c.get("body") or "")


def cmd_pr_diff(a):
    print(_request_text(f"{API}{_rp(a.repo, f'/pulls/{a.number}')}",
                        "application/vnd.github.v3.diff"), end="")


def cmd_pr_edit(a):
    """Edit an existing PR's title / body / base (the `gh pr edit` surface).

    Only the PR's own metadata is touched via PATCH /pulls/{n}; still no merge,
    no api/gist/secret verbs — the same narrow, bounded capability as the rest of
    this helper. Body comes via --body-file, so no heredoc rides the command line."""
    payload = {}
    if a.title is not None:
        payload["title"] = a.title
    body = _resolve_body(a.body, a.body_file, required=False)
    if body is not None:
        payload["body"] = body
    if a.base is not None:
        payload["base"] = a.base
    if not payload:
        sys.exit("error: pr-edit needs at least one of --title / --body / --body-file / --base")
    pr = _api("PATCH", _rp(a.repo, f"/pulls/{a.number}"), payload)
    print(f"#{pr['number']} edited -> {pr['html_url']}")


def cmd_link_branch(a):
    """Create a real issue<->branch link (populates the issue's Development panel)
    via the GraphQL createLinkedBranch mutation, pointing at an EXISTING branch's tip."""
    issue = _api("GET", _rp(a.repo, f"/issues/{a.number}"))
    ref = _api("GET", _rp(a.repo, f"/git/ref/heads/{a.branch}"))
    q = ("mutation($issueId:ID!,$oid:GitObjectID!,$name:String){"
         "createLinkedBranch(input:{issueId:$issueId,oid:$oid,name:$name}){"
         "linkedBranch{ id ref{ name } }}}")
    data = _graphql(q, {"issueId": issue["node_id"], "oid": ref["object"]["sha"], "name": a.branch})
    lb = data["createLinkedBranch"]["linkedBranch"]
    print(f"linked branch '{lb['ref']['name']}' to #{a.number} ({issue['html_url']})")


def main():
    p = argparse.ArgumentParser(prog="gh_issue.py")
    p.add_argument("--repo", default=DEFAULT_REPO)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list"); s.add_argument("--state", default="open", choices=["open", "closed", "all"])
    s.add_argument("--label", action="append"); s.add_argument("--limit", type=int, default=30)
    s.add_argument("--include-prs", action="store_true"); s.set_defaults(func=cmd_list)

    s = sub.add_parser("view"); s.add_argument("number"); s.set_defaults(func=cmd_view)

    s = sub.add_parser("create"); s.add_argument("--title", required=True)
    s.add_argument("--body"); s.add_argument("--body-file", dest="body_file")
    s.add_argument("--label", action="append"); s.set_defaults(func=cmd_create)

    s = sub.add_parser("comment"); s.add_argument("number")
    s.add_argument("--body"); s.add_argument("--body-file", dest="body_file"); s.set_defaults(func=cmd_comment)

    s = sub.add_parser("close"); s.add_argument("number")
    s.add_argument("--comment"); s.add_argument("--comment-file", dest="comment_file"); s.set_defaults(func=cmd_close)

    s = sub.add_parser("reopen"); s.add_argument("number"); s.set_defaults(func=cmd_reopen)

    s = sub.add_parser("edit"); s.add_argument("number")
    s.add_argument("--title"); s.add_argument("--body"); s.add_argument("--body-file", dest="body_file")
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser("label"); s.add_argument("number")
    s.add_argument("--add", action="append"); s.add_argument("--remove", action="append"); s.set_defaults(func=cmd_label)

    s = sub.add_parser("pr-create"); s.add_argument("--title", required=True); s.add_argument("--head", required=True)
    s.add_argument("--base", default="main")
    s.add_argument("--body"); s.add_argument("--body-file", dest="body_file"); s.set_defaults(func=cmd_pr_create)

    s = sub.add_parser("pr-list"); s.add_argument("--state", default="open", choices=["open", "closed", "all"])
    s.add_argument("--base"); s.add_argument("--head"); s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_pr_list)

    s = sub.add_parser("pr-view"); s.add_argument("number"); s.set_defaults(func=cmd_pr_view)

    s = sub.add_parser("pr-diff"); s.add_argument("number"); s.set_defaults(func=cmd_pr_diff)

    s = sub.add_parser("pr-edit"); s.add_argument("number")
    s.add_argument("--title"); s.add_argument("--body"); s.add_argument("--body-file", dest="body_file")
    s.add_argument("--base"); s.set_defaults(func=cmd_pr_edit)

    s = sub.add_parser("link-branch"); s.add_argument("number"); s.add_argument("--branch", required=True)
    s.set_defaults(func=cmd_link_branch)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
