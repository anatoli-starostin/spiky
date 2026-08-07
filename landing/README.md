# Spiky landing page — source of truth

This folder is the **reproducible source** for the public Spiky landing page and its interactive
Walker2d demo. It lives on the `landing` branch (NOT `gh-pages`, which is the auto-deployed production
site). Anyone can clone this branch and rebuild the fully-working landing page: a static **client** on GitHub
Pages plus a Python/MuJoCo **WebSocket server** on a small VM.

## Layout

```
landing/
├── README.md                 # this file
├── site/                     # the landing-page markup that is served at the gh-pages ROOT
│   ├── index.html            #   the landing page (cards linking to the demo)
│   └── style.css             #   its stylesheet
└── walker2d-viz/             # the FULL client+server framework for the live demo
    ├── client/               #   browser client (Three.js; three via CDN import map)
    │   ├── index.static.html #     node-free entry (this is what ships to Pages as index.html)
    │   ├── config.js         #     window.WALKER2D_WS = the server's wss:// URL
    │   └── src/main.js        #     renderer + FK (accurate MJCF mode + approximate; foldable UI)
    ├── server/               #   headless Python WS server (physics/state streaming only)
    │   ├── server.py, actors/, models/ (baked-in checkpoints), requirements.txt, Dockerfile
    ├── docker-compose.yml    #   server:8765 (internal) + Caddy:80/443 (TLS reverse proxy)
    ├── Caddyfile             #   auto Let's Encrypt for $DOMAIN, transparent ws upgrade
    ├── .env.example          #   DOMAIN, MAX_SESSIONS
    └── DEPLOY.md             #   full deploy runbook
```

> **Adding a new model/actor to the demo?** See
> **[walker2d-viz/ADDING_MODELS.md](walker2d-viz/ADDING_MODELS.md)** — the concrete, reproducible procedure
> (write an `Actor` subclass, store weights, commit here, rebuild the server on the VM).

These are the **current** (latest) versions: the client carries the accurate MJCF render mode + the
Approximate/Accurate toggle, the foldable controls panel, and `config.js` pointing at
`wss://89-169-96-79.sslip.io`; the server `requirements.txt` includes the full pinned dep set (with
`imageio`, needed for `gym.make("Walker2d-v5")` in a clean container).

## How this maps to what gets deployed to gh-pages

The `gh-pages` branch is a flat deployed site. This `landing/` folder maps onto it as:

| source here | deployed on gh-pages |
|---|---|
| `landing/site/index.html`  | `/index.html` (site root) |
| `landing/site/style.css`   | `/style.css` |
| `landing/walker2d-viz/client/index.static.html` | `/walker2d-viz/index.html` |
| `landing/walker2d-viz/client/config.js`         | `/walker2d-viz/config.js` |
| `landing/walker2d-viz/client/src/main.js`       | `/walker2d-viz/src/main.js` |

So the published URLs are `anatoli-starostin.github.io/spiky/` (landing) and
`.../spiky/walker2d-viz/` (the live demo). Publishing today is done by copying those client files into the
`gh-pages` walker2d-viz/ folder and pushing gh-pages. (The older per-page construction demos that also live on
gh-pages are unlisted from the landing page and are not part of this source of truth.)

## Reproduce / deploy

### Client (GitHub Pages, HTTPS)
The client is static (Three.js from a CDN import map — no build step). To publish:
copy `landing/site/index.html` + `style.css` to the gh-pages root, and
`landing/walker2d-viz/client/{index.static.html→index.html, config.js, src/main.js}` to gh-pages
`walker2d-viz/`, then push gh-pages. Set `config.js`'s `window.WALKER2D_WS` to your server's **`wss://`** URL
(Pages is HTTPS, so a plain `ws://` is blocked as mixed content).

### Server (a small CPU VM with a public IP)
From `landing/walker2d-viz/` on the box (needs Docker + the Compose plugin):
```
cp .env.example .env          # set DOMAIN=<hostname resolving to this box> and MAX_SESSIONS
docker compose up -d --build  # builds server:8765, starts server + Caddy(:80/:443)
```
Caddy auto-obtains a Let's Encrypt cert for `$DOMAIN` and proxies `wss://$DOMAIN → ws://server:8765`.
Requirements: **ports 80 and 443 open** to the internet (80 for the ACME challenge, 443 for wss), and a
hostname that resolves to the box — either a real domain's A-record or a no-domain option like
`<dashed-ip>.sslip.io` (the current deployment uses `89-169-96-79.sslip.io`). Full runbook +
concurrency notes: `walker2d-viz/DEPLOY.md`.

Gotcha (already fixed in `server/requirements.txt`): gymnasium's MuJoCo env creation imports `imageio`
(and other transitive deps) even headless; a bare `gymnasium` install omits them, so `gym.make` fails in a
clean `python:slim` image — the pinned full dep set here avoids that.
