# walker2d-viz — how this branch is used

This directory lives on the **`live/walker2d-viz`** branch, which is the **source-of-truth
for the live Walker2d viz demo**. What's here is what's deployed.

## What's live, and where

- **Client** (this dir's `client/`): published to **GitHub Pages** from the `gh-pages`
  branch → https://anatoli-starostin.github.io/spiky/walker2d-viz/
  (`client/src/main.js` → gh-pages `walker2d-viz/src/main.js`,
   `client/index.static.html` → gh-pages `walker2d-viz/index.html`).
- **Server** (this dir's `server/`): runs as a **Docker websocket container** on
  VM `89.169.96.79`, reachable at **`wss://89-169-96-79.sslip.io`** (Caddy TLS via sslip.io).

## Nothing auto-syncs — redeploy manually after any change here

The VM is **not a git checkout** (`~/spiky` on the box has no `.git`). A commit on
`live/walker2d-viz` changes nothing that's live until it's manually pushed out:

**Client** (after editing `client/`):
1. Mirror into the gh-pages worktree (`walker2d-viz/src/main.js`, `walker2d-viz/index.html`).
2. `git push origin gh-pages` — GitHub Pages redeploys (CDN propagation ~30–90 s).

**Server** (after editing `server/`):
1. `scp` the changed files to `nucstar@89.169.96.79:~/spiky/landing/walker2d-viz/server/…`
   (`ssh -i ~/.ssh/id_ed25519`). Never `git pull` on the VM.
2. `cd ~/spiky/landing/walker2d-viz && sudo docker compose up -d --build server`
   (rebuilds + recreates the container).
3. `sudo docker compose restart caddy` — so Caddy re-resolves the recreated server's IP,
   otherwise the demo goes unreachable.

## Verify after deploy
- Server files / running container match source (`md5sum` the `server/` tree vs the VM +
  `docker compose exec server md5sum …`).
- Live check over `wss://89-169-96-79.sslip.io`: the actor appears in the dropdown, the
  walker walks, controls (restart/pause/no_reset) work, and the spike raster + network
  panels still function.

Because it's protected/live: **don't force-push `live/walker2d-viz`.**
