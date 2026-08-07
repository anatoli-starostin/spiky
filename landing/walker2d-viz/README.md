# walker2d-viz

A tiny **server–client** app to visualize and control a Gymnasium MuJoCo **Walker2d** robot.
The Python server steps the env and streams state over WebSocket; the browser client renders the walker in
3D (Three.js) with camera-follow and live controls.

```
walker2d-viz/
├── server/                     # Python: gym env loop + WebSocket, streams state, takes control msgs
│   ├── server.py               #   run this
│   ├── requirements.txt
│   └── actors/                 # pluggable control policies (auto-discovered)
│       ├── base.py             #   Actor interface: act(obs) -> action
│       ├── random_actor.py     #   "random"
│       ├── zero_actor.py       #   "zero" (limp / stand)
│       └── __init__.py         #   discover_actors()
└── client/                     # Vite + Three.js browser front-end
    ├── index.html              #   UI (restart / mode / speed / actor)
    └── src/main.js             #   scene, pose reconstruction from qpos, camera follow, WS
```

## Run the server

Needs a MuJoCo-capable Python env (not installed on this box by default):

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # gymnasium[mujoco], mujoco, numpy, websockets
python server.py                          # ws://0.0.0.0:8765, Walker2d-v5 (falls back to v4/v3)
# options: --host --port --env Walker2d-v5 --sps 30
```

The server runs one env loop, streams a JSON `state` message per tick to all connected clients:
`{ type:"state", env, qpos:[x, z, torso_rot, R thigh/leg/foot, L thigh/leg/foot], reward, return, step, mode, actor, sps }`
and accepts control messages: `{cmd:"restart"}`, `{cmd:"mode", mode:"train"|"test"}`, `{cmd:"speed", sps}`,
`{cmd:"actor", name}`, `{cmd:"list_actors"}`. On connect it sends `{type:"actors", actors:[…], active}`.

**Adding an actor:** drop `server/actors/my_actor.py` with

```python
from .base import Actor
class MyActor(Actor):
    name = "my"
    def act(self, obs):
        return ...  # np.ndarray in self.action_space
```

It is auto-discovered and appears in the client's actor combo box. (E.g. later wire a trained LIFDetectorsMHL
policy in as an actor.)

**Train mode** is currently a stub: toggling to `train` logs "training not yet implemented" and keeps running
the selected actor so the viz continues — the hook is there to add a real training loop later.

## Run the client

Needs Node.js + npm:

```bash
cd client
npm install                               # three + vite
npm run dev                               # http://localhost:5173
```

Open the page, set the **server URL** field if not `ws://localhost:8765`, and use:
**Restart** · **Mode: test/train** toggle · **Speed** slider (sim steps/sec) · **Actor** combo box.

## Rendering notes

- The client reconstructs the pose with simple forward kinematics from `qpos` (torso + thigh/shin/foot per
  leg) and draws capsule/cylinder bones — an approximation of the walker2d MJCF, not a MJCF/URDF load. The
  two legs are offset slightly in depth for a pseudo-3D read of the 2D (x–z plane) walker. **Joint-angle
  signs are approximate**; if a joint bends the "wrong" way, flip the corresponding sign in
  `poseFromQpos()` in `client/src/main.js`. Segment lengths live in the `LEN` object there.
- The camera follows the torso along x with smoothing at a fixed height/distance (tweak `camOffset`).

## Dependencies to install

- **Server:** Python 3.10+, `gymnasium[mujoco]`, `mujoco`, `numpy`, `websockets` (see `requirements.txt`).
  MuJoCo wheels are self-contained (no separate MuJoCo binary needed). Headless is fine — the server never
  renders; only the env physics runs.
- **Client:** Node.js 18+, `three`, `vite` (installed via `npm install`).
