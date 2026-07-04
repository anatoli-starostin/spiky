---
name: spikyclaudebot GitHub setup
description: SSH key and GitHub account for Claude to push PRs as a bot contributor
type: reference
---

GitHub bot account for Claude-authored PRs: `spikyclaudebot` (spikyclaudebot@gmail.com)

SSH private key: `/home/starost/spiky/.ssh/spikyclaudebot_ed25519` (gitignored, persistent)
SSH public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILq83f0hGxwdxBrYCSWlQMSl/r0Wi2kr81Boi7OvwWvT spikyclaudebot@gmail.com`
Classic PAT (repo scope): `<REDACTED_GITHUB_PAT>`

**How to apply:**
- Commit as: `git -c user.name="spikyclaudebot" -c user.email="spikyclaudebot@gmail.com" commit`
- Push as: `GIT_SSH_COMMAND="ssh -i /home/starost/spiky/.ssh/spikyclaudebot_ed25519 -o StrictHostKeyChecking=no" git push git@github.com:anatoli-starostin/spiky.git <branch>`
- Create PR via API: `curl -X POST -H "Authorization: token <REDACTED_GITHUB_PAT>" https://api.github.com/repos/anatoli-starostin/spiky/pulls -d '{...}'`
- Key is stored in repo under `.ssh/` (gitignored) — no need to regenerate unless explicitly rotated
