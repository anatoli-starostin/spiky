# Deploying the Walker2d demo stand to the internet

Two halves:

- **Server** — a headless Python/MuJoCo WebSocket app, run in Docker on a small **CPU** VM. Streams sim
  state; each browser gets its **own** simulation. Fronted by **Caddy**, which terminates TLS so browsers
  connect over `wss://`.
- **Client** — the static page in `client/`, published to **GitHub Pages** (HTTPS). It connects to your
  server's `wss://` URL, set in `client/config.js`.

```
 Browser ──HTTPS──▶ GitHub Pages (static client)
    │
    └──────────wss://demo.example.com────────▶ Caddy :443  ──ws──▶ server :8765 (Docker)
```

You provision the VM and DNS yourself. Nothing here spends money or touches a cloud host.

---

## 1. Concurrency model (important)

The server runs **one isolated session per WebSocket connection**: each connected browser gets its own
Gymnasium env instance, its own selected actor, and its own pause / mode / free-fall state. Viewers never
share or fight over a single env. A global cap **`MAX_SESSIONS`** bounds CPU — once reached, new
connections are refused with `{"type":"error", ...}` (the client shows it as a failed connection).

Each session is one MuJoCo env stepping at `SPS` (default 30) steps/s — CPU-bound, no GPU. The initial
target box is a **2 vCPU / 8 GB Nebius Ice Lake** VM, and `MAX_SESSIONS` **defaults to 6** — comfortably
~5–8 concurrent viewers on 2 cores. Raise it (e.g. **16–24**) if you move to a larger box; watch CPU while
a few viewers are connected and tune.

---

## 2. Server: what you need on the box

- A Linux VM with a **public IP**, Docker + the Compose plugin.
- Ports **80** and **443** open to the internet (Caddy needs 80 for the ACME challenge, 443 for wss).
- A **DNS A record** (and AAAA if you have IPv6) for your hostname, e.g. `demo.example.com`, pointing at
  the VM's public IP. Verify with `dig +short demo.example.com` before starting.

Then, from a checkout of this repo on the box:

```bash
cp .env.example .env
# edit .env: set DOMAIN=demo.example.com  (and optionally MAX_SESSIONS)
docker compose up -d --build
```

That builds the server image (Python 3.12, pinned deps, model checkpoints baked in) and starts two
containers: `server` (internal only, port 8765) and `caddy` (ports 80/443). Caddy automatically obtains a
Let's Encrypt certificate for `$DOMAIN` and proxies `wss://$DOMAIN` → `ws://server:8765` (the WebSocket
upgrade is forwarded transparently).

Check it:

```bash
docker compose ps
docker compose logs -f caddy     # watch for successful certificate issuance
docker compose logs -f server    # "[server] +session (n/MAX)" as viewers connect
```

Config knobs (all env vars, set in `.env` / compose):

| var            | default        | meaning                                             |
|----------------|----------------|-----------------------------------------------------|
| `DOMAIN`       | *(required)*   | FQDN Caddy serves + gets a cert for                 |
| `MAX_SESSIONS` | `6`            | concurrent viewer cap (6 fits 2 vCPU; raise on a bigger box) |
| `SPS`          | `30`           | sim steps/sec streamed per session                  |
| `ENV_ID`       | `Walker2d-v5`  | Gymnasium env id                                    |
| `PORT`/`HOST`  | `8765`/`0.0.0.0` | internal server bind (rarely changed)             |

**Headless / GL:** the server only streams physics state (qpos) — it never renders — so no display, EGL,
OSMesa, or `MUJOCO_GL` is required. The Docker image installs `libgl1`/`libglib2.0-0` defensively only so
`import mujoco` is guaranteed to succeed on the slim base.

**Without a domain (quick test):** you can expose the raw server directly by publishing its port instead
of using Caddy (e.g. `docker run -p 8765:8765 --build-context ... ` or add a `ports: ["8765:8765"]` to the
`server` service) and point the client at `ws://VM_IP:8765`. This only works from an **HTTP** client page
(a local file / `http.server`), not from GitHub Pages, because HTTPS pages require `wss://`.

---

## 3. Client: publish to GitHub Pages

The client is plain static files — no build step (three.js loads from a CDN via an import map, paths are
relative so a project sub-path works).

1. Edit **`client/config.js`**:
   ```js
   window.WALKER2D_WS = "wss://demo.example.com";
   ```
2. Publish `client/` to Pages. Two options:

   **a) GitHub Actions (turnkey, included).** `.github/workflows/pages.yml` uploads `client/` to Pages on
   push to your default branch. In the repo: **Settings → Pages → Source = "GitHub Actions"**. Adjust the
   `branches:` in the workflow to your default branch if it isn't `main`.

   **b) gh-pages branch (manual).**
   ```bash
   git subtree push --prefix client origin gh-pages
   ```
   then **Settings → Pages → Source = "Deploy from a branch" → branch `gh-pages` / root**.

Your client is then at `https://<user>.github.io/<repo>/`. Open it; the connection indicator should read
**connected** and the walker should animate. The server URL field in the UI lets a viewer override the
target at runtime for debugging.

---

## 4. Local development (no Docker, no TLS)

```bash
# server
cd server && python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
PYTHONPATH=. ./.venv/bin/python server.py            # ws://localhost:8765

# client (separate terminal) — with config.js WALKER2D_WS="" it auto-targets the serving host
cd client && python -m http.server 5173              # open http://localhost:5173
```

---

## 5. Operating notes

- **Certs persist** in the `caddy_data` volume — don't delete it or you'll re-hit Let's Encrypt rate limits.
- **Update the deploy:** `git pull && docker compose up -d --build`.
- **Scale the cap:** raise `MAX_SESSIONS` in `.env` and `docker compose up -d` (recreates the server).
- Let's Encrypt has issuance rate limits; if you're iterating on DNS, test with a throwaway subdomain.
