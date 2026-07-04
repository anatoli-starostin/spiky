---
name: feedback-memos-not-tasks
description: "When the user dumps a backlog of ideas via Telegram (or any channel) and says 'just a memo for myself, do not do anything', do NOT create TaskCreate entries. Memos are not tasks."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Memos are not tasks — don't create TaskCreate entries for them

**Rule**: When the user labels a list of ideas as "a memo for myself" / "not actions" / "do not do anything", do not create TaskCreate entries for them. Just acknowledge and move on.

**Why**: TaskCreate entries imply commitment to act. Memos are personal note-taking; the user keeps them in their own systems (Telegram, notebooks, etc.). Creating tasks for memos clutters the to-do list with non-actionable items and requires the user to clean up later.

**How to apply**:
- Default: when a Telegram dump contains brainstorming, ideas, or future experiments, treat as conversation. Do not auto-create tasks.
- Only create tasks for items the user has *explicitly asked me to do* (or that I've started doing).
- If the user later corrects me ("it was just a memo"), immediately delete the stale tasks; don't leave them around as a list of half-promises.
- Phrase confirmation as "noted" or "ack" — not "I'll track this" or "added to my list".
