# Adding a new model / actor to the Walker2d demo

This is the concrete, reproducible procedure for adding a control policy ("actor") to the live Walker2d
stand — for any fleet agent, assuming no prior exposure to the project. It reflects how the code **actually**
works (`server/actors/__init__.py`, `server/actors/base.py`, and the example actors).

An **actor** is one selectable policy in the demo's "Actor" dropdown. The server runs one MuJoCo
`Walker2d-v5` env per connected viewer and, each step, calls the selected actor's `act(obs)` to drive the
walker; the resulting state is streamed to the browser.

There are three parts: **(1)** write the actor, **(2)** store its weights, **(3)** redeploy the server so it
goes live. Steps (1)+(2) are also committed to the `landing` branch as the source of truth.

---

## 1. The actor interface — where model source goes

Drop a new Python file in **`server/actors/`** that subclasses `Actor` (from `base.py`) and implements
`act(obs)`. **No registration** is needed: `server/actors/__init__.py` auto-discovers every `Actor`
subclass in the package via a `pkgutil` scan and registers it under its `name`.

The interface (`server/actors/base.py`):

```python
class Actor:
    name: str = "base"                      # UNIQUE — this string is the registry key AND the dropdown label
    def __init__(self, action_space):       # action_space is the gym Box: shape (6,), low/high = -1/+1
        self.action_space = action_space
    def act(self, obs):                     # obs -> action
        raise NotImplementedError
```

Contract for `act(obs)`:
- **`obs`** is the **17-dim Walker2d-v5 observation** as a numpy array (positions/velocities of the planar
  walker; physical, not pre-normalized).
- **return** a plain **numpy array of shape `(6,)`**, dtype `float32`, in the action range **`[-1, 1]`**
  (the 6 continuous joint torques). Deterministic policies typically **`tanh`-squash** their pre-activation
  means to `[-1, 1]`.

**Obs normalization is model-dependent — match how YOUR model was trained:**
- Many of our models **standardize** the obs with stored mean/std, e.g. `lut_teacher.py`:
  `x = (obs - obs_mean) / (obs_std + 1e-6)`, where `obs_mean`/`obs_std` come from
  `models/walker_dataset_stats.json`.
- Some use the **raw** obs, e.g. `sac_baseline.py` (SB3 SAC does no obs normalization).

**Discovery gotchas** (from `actors/__init__.py`):
- The class must **subclass `Actor`** and be **defined in that module** (not imported into it).
- `name` must be **unique** across all actors — the registry is keyed by `name`, so a duplicate silently
  overwrites. Pick a distinct, human-readable label (it's what viewers see).
- The file `base.py` itself is skipped; any other `*.py` in `actors/` is scanned.

### Minimal template

```python
# server/actors/my_actor.py
import os
import numpy as np
from .base import Actor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

class MyActor(Actor):
    name = "My cool policy"                        # unique; shown in the client's Actor dropdown

    def __init__(self, action_space):
        super().__init__(action_space)
        # Load weights RELATIVE to this file (pure numpy — the server image has no torch).
        Q = np.load(os.path.join(_MODELS, "my_actor.npz"))
        self.W1 = Q["W1"].astype(np.float64)       # ... your params
        # If your model standardizes obs, also load its stats:
        # import json
        # S = json.load(open(os.path.join(_MODELS, "my_actor_stats.json")))
        # self.obs_mean = np.asarray(S["obs_mean"]); self.obs_std = np.asarray(S["obs_std"])

    def act(self, obs):
        x = np.asarray(obs, np.float64).reshape(-1)[:17]     # 17-dim Walker2d-v5 obs
        # x = (x - self.obs_mean) / (self.obs_std + 1e-6)     # ONLY if your model expects standardized obs
        mu = self.W1 @ x                                     # ... your forward pass -> 6 pre-squash means
        return np.tanh(mu).astype(np.float32)                # (6,) action in [-1, 1]
```

Study the real ones as references: `actors/lut_teacher.py` (int4 LUT hyperplane policy, **standardized** obs),
`actors/sac_baseline.py` (2-layer MLP, **raw** obs), `actors/random_actor.py` (one-liner: returns
`self.action_space.sample()`).

---

## 2. How to store the model weights

- **Where:** put checkpoints in **`server/models/`** — an `.npz` of arrays (+ an optional stats `.json`).
- **How the actor finds them:** resolve paths **relative to the actor file**, exactly as the examples do:
  ```python
  _MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
  Q = np.load(os.path.join(_MODELS, "my_actor.npz"))
  ```
  Never hard-code absolute paths — the actor must be self-contained and work inside the Docker image.
- **Naming:** name files after the actor/run so they're traceable, e.g. `my_actor.npz`,
  `my_actor_stats.json` (see `lut_sac_c21_seed4_20k_actor.npz`, `sac_baseline_actor.npz`,
  `walker2d_lut_actor_int4.npz`, `walker_dataset_stats.json`).
- **Prefer pure numpy — no torch.** The server image is deliberately torch-free (physics/state streaming
  only). Export your trained weights to numpy arrays and re-implement the (small) forward pass with numpy, as
  every current actor does. If a model genuinely needs a heavier runtime, you must add the dependency to
  `server/requirements.txt` **and** rebuild the image (see §3) — but that bloats the image, so avoid it when a
  numpy re-implementation is feasible.

---

## 3. Reproducibility — commit to the `landing` branch (source of truth)

`server/models/` is committed into the `landing` branch (unlike heavy training checkpoints elsewhere). So a
new model means committing **both** files there:

```
landing/walker2d-viz/server/actors/my_actor.py      # the actor
landing/walker2d-viz/server/models/my_actor.npz     # its weights (+ any *_stats.json)
```

Commit them on the `landing` branch and push (as nucstarbot). This keeps the branch a complete, cloneable
source of truth: anyone can rebuild the demo with your model from it.

---

## 4. Redeploy the server so the new model goes live

The live server runs in **Docker on the VM `89.169.96.79`** (project copy at `/home/nucstar/walker2d-viz`).
The Docker image **COPYs `server/actors/` and `server/models/` into the image at build time**, so new files
require a **rebuild**, not just a restart.

```sh
# 1) get the new actor + weights onto the VM (from a checkout / the landing branch):
rsync -av server/actors/ nucstar@89.169.96.79:/home/nucstar/walker2d-viz/server/actors/
rsync -av server/models/ nucstar@89.169.96.79:/home/nucstar/walker2d-viz/server/models/
#    (or re-ship the whole project as a tarball; scp works too)

# 2) rebuild + restart on the VM:
ssh nucstar@89.169.96.79
cd /home/nucstar/walker2d-viz
sudo docker compose up -d --build          # rebuilds the server image with the new actor/model, restarts

# 3) verify:
sudo docker compose logs server | tail      # startup line: "[server] actors: [...]" MUST list your new name
```

Then confirm end-to-end with a real wss client (the exact path the browser uses):

```python
# on any host with `websockets`:
import asyncio, json, websockets
async def m():
    async with websockets.connect("wss://89-169-96-79.sslip.io") as ws:
        print(json.loads(await ws.recv())["actors"])   # first frame is {"type":"actors","actors":[...]}
asyncio.run(m())
```

Your `name` should appear in that `actors` list. The **client needs no change** — it populates the Actor
dropdown from the server's `actors` message on connect, so viewers just **reload** the page
(`anatoli-starostin.github.io/spiky/walker2d-viz/`) to see the new option. `MAX_SESSIONS`, the Caddy `wss`
endpoint, TLS/Let's Encrypt, and ports 80/443 all stay as they are.

**If you added a Python dependency** for the model: add it to `server/requirements.txt` (pinned) and the
`docker compose up -d --build` rebuild will install it. Verify the container still starts (a missing/mismatched
dep shows up as a traceback in `docker compose logs server`). Keep the dep set minimal — the whole point of the
server image is to stay light and torch-free.

---

## Checklist

1. `server/actors/my_actor.py` — subclass `Actor`, unique `name`, `act(obs)->(6,) float32 in [-1,1]`, obs
   normalization matching training, weights loaded relative to the file.
2. `server/models/my_actor.npz` (+ optional `*_stats.json`) — pure-numpy weights.
3. Commit both under `landing/walker2d-viz/` on the `landing` branch; push.
4. Ship actor+models to the VM, `docker compose up -d --build`, confirm the startup log + a wss client list
   your `name`; reload the Pages client.
5. If you added a dependency: update `server/requirements.txt` and confirm the rebuild picks it up.
