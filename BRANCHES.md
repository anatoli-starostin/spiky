# Branch naming convention

A small, scannable convention so we can tell at a glance what a branch is for and how
much care it needs.

| Prefix       | Meaning                                                                          |
|--------------|----------------------------------------------------------------------------------|
| `main`       | Stable trunk. Everything lands here eventually.                                  |
| `live/`      | Canonical branch that a **running deployment** is served from. Treat as protected — **do not force-push**; a running service depends on it. |
| `exp/`       | Active experiments.                                                              |
| `research/`  | Analysis / writeup branches (docs, figures, investigations).                    |
| `archive/`   | Retired branches kept for reference. May also be tag-archived under `archive/<name>`. |

## Deployments

- **`live/walker2d-viz`** — the source-of-truth for the live Walker2d viz demo. It maps to
  two separately-hosted pieces:
  - **client**: the static site published to **GitHub Pages** from the `gh-pages` branch,
    at https://anatoli-starostin.github.io/spiky/walker2d-viz/
  - **server**: a **Docker websocket container** on VM `89.169.96.79`
    (`wss://89-169-96-79.sslip.io`).
- **Syncing is MANUAL.** The VM is **not** a git checkout — the server is deployed by
  `scp` of the files + a container rebuild, and the client is a `git push` to `gh-pages`.
  Nothing auto-syncs, so any change on `live/walker2d-viz` must be *manually* redeployed to
  take effect. See `landing/walker2d-viz/DEPLOYMENT.md` for the exact steps.
