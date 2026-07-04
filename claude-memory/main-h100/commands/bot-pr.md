# bot-pr

Create a PR as spikyclaudebot from staged/modified changes.

## Setup (if SSH key missing)
If `/tmp/spikyclaudebot_ed25519` doesn't exist, regenerate:
```
ssh-keygen -t ed25519 -C "spikyclaudebot@gmail.com" -f /tmp/spikyclaudebot_ed25519 -N ""
```
Then tell the user to add the new public key to spikyclaudebot's GitHub SSH keys before continuing.

## Steps

1. Ask the user for a branch name, PR title, and description if not provided in the arguments.

2. Create a new git branch.

3. Commit all staged changes using the bot identity:
   ```
   git -c user.name="spikyclaudebot" -c user.email="spikyclaudebot@gmail.com" commit -m "<message>"
   ```

4. Push the branch using the bot SSH key:
   ```
   GIT_SSH_COMMAND="ssh -i /tmp/spikyclaudebot_ed25519 -o StrictHostKeyChecking=no" git push git@github.com:anatoli-starostin/spiky.git <branch>
   ```

5. Create the PR via GitHub API using the classic PAT:
   ```
   curl -s -X POST \
     -H "Authorization: token <REDACTED_GITHUB_PAT>" \
     -H "Content-Type: application/json" \
     https://api.github.com/repos/anatoli-starostin/spiky/pulls \
     -d '{"title":"<title>","body":"<body>","head":"<branch>","base":"main"}'
   ```

6. Print the resulting PR URL.
