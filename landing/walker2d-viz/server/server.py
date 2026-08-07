"""Walker2d visualization/control server.

Runs a Gymnasium MuJoCo Walker2d env and streams state to WebSocket clients as JSON, accepting control
messages (restart / mode / speed / actor / pause / no_reset). Actors are auto-discovered from actors/.

Concurrency model: **one independent session per WebSocket connection**. Each connected browser gets its
own Sim (its own env instance, selected actor, pause/mode/free-fall state and stepping task), so multiple
viewers never fight over a shared env — required for a public multi-viewer demo. A global cap
(MAX_SESSIONS) bounds CPU: extra connections are refused with an {"type":"error"} message.

Config via env vars (CLI flags override): HOST, PORT, ENV_ID, SPS, MAX_SESSIONS.
Run:  python server.py [--host 0.0.0.0] [--port 8765] [--env Walker2d-v5] [--sps 30] [--max-sessions 8]
"""
import argparse
import asyncio
import contextlib
import json
import os

import numpy as np
import websockets

try:
    import gymnasium as gym
except Exception as e:  # pragma: no cover
    raise SystemExit("gymnasium is required: pip install 'gymnasium[mujoco]'  (%r)" % e)

from actors import discover_actors

# Discover the actor classes ONCE at import; the registry is read-only and shared across all sessions
# (each session instantiates its own actor from it against its own env's action space).
REGISTRY = discover_actors()
if not REGISTRY:
    raise SystemExit("no actors discovered in actors/")


def make_env(preferred):
    """Create the Walker2d env, falling back v5 -> v4 -> v3 if a version isn't installed."""
    candidates = [preferred, "Walker2d-v5", "Walker2d-v4", "Walker2d-v3"]
    seen, errors = set(), []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            env = gym.make(name)
            return env, name
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise SystemExit("could not create any Walker2d env:\n  " + "\n  ".join(errors))


class Sim:
    """One isolated session: its own env, actor, and control state. NOT shared between connections."""

    def __init__(self, env_name, sps):
        self.env, self.env_name = make_env(env_name)
        self.registry = REGISTRY
        # Default a fresh session to a GOOD walking policy so viewers don't see the random flail-and-fall on
        # load; fall back to "random" then the first discovered actor if that policy isn't present.
        _default = "fastlut_lse (exp19)"
        self.actor_name = (_default if _default in self.registry
                           else "random" if "random" in self.registry
                           else next(iter(self.registry)))
        self.actor = self.registry[self.actor_name](self.env.action_space)
        self.mode = "test"                       # "test" | "train"
        self.paused = False                      # when True, the stepper idles (no step, no broadcast)
        self._resume = asyncio.Event()           # set() while running; cleared while paused -> stepper awaits it
        self._resume.set()
        self.no_reset = False                    # free-fall: when True, don't auto-reset on termination
        self.terminated = False
        self.sps = float(sps)
        self.obs, _ = self.env.reset(seed=0)
        self.step_count = 0
        self.reward = 0.0
        self.ret = 0.0                            # episode return

    # ---- control ----
    def set_actor(self, name):
        if name in self.registry:
            self.actor_name = name
            self.actor = self.registry[name](self.env.action_space)

    def set_mode(self, mode):
        self.mode = "train" if mode == "train" else "test"

    def set_no_reset(self, value):
        self.no_reset = bool(value)

    def set_paused(self, value):
        self.paused = bool(value)
        if self.paused:
            self._resume.clear()                 # stepper will send one final frame then await _resume
        else:
            self._resume.set()                   # wake the idling stepper -> resume stepping/broadcast

    def restart(self):
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.reward = 0.0
        self.ret = 0.0
        self.terminated = False

    def step(self):
        action = self.actor.act(self.obs)
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        self.obs, r, terminated, truncated, _ = self.env.step(action)
        self.reward = float(r)
        self.ret += self.reward
        self.step_count += 1
        self.terminated = bool(terminated or truncated)
        # Free-fall mode: keep stepping the (fallen) body instead of resetting. MuJoCo happily keeps
        # integrating a terminated Walker2d state (verified), so the walker just lies/flails on the ground.
        if self.terminated and not self.no_reset:
            self.restart()

    def close(self):
        with contextlib.suppress(Exception):
            self.env.close()

    def actor_list_msg(self):
        return json.dumps({"type": "actors", "actors": sorted(self.registry.keys()), "active": self.actor_name})

    def state_msg(self):
        data = self.env.unwrapped.data
        return json.dumps({
            "type": "state",
            "env": self.env_name,
            "qpos": np.asarray(data.qpos).tolist(),   # [x, z, torso_rot, R thigh/leg/foot, L thigh/leg/foot]
            "reward": self.reward,
            "return": self.ret,
            "step": self.step_count,
            "mode": self.mode,
            "actor": self.actor_name,
            "sps": self.sps,
            "paused": self.paused,
            "no_reset": self.no_reset,
            "terminated": self.terminated,
        })


async def stepper(sim, ws):
    """Per-session loop: step this session's env at its sps and stream state to its one client.

    Steady cadence: pace to an absolute next-tick time (monotonic) so states arrive at a constant rate
    rather than drifting/bursting with the per-tick step+send cost.
    """
    loop = asyncio.get_event_loop()
    next_t = loop.time()
    try:
        while True:
            if sim.paused:
                # Send exactly ONE frame reflecting the frozen pose + paused=true, then go fully quiet:
                # no physics, no inference, no per-tick frames. Await the resume event (no busy-spin, ~0 CPU).
                # The connection is kept alive by the websockets library's built-in ping/pong while we idle.
                try:
                    await ws.send(sim.state_msg())
                except websockets.exceptions.ConnectionClosed:
                    break
                await sim._resume.wait()
                next_t = loop.time()      # resync pacing on resume (don't burst-catch-up the paused gap)
                continue
            sim.step()
            try:
                await ws.send(sim.state_msg())
            except websockets.exceptions.ConnectionClosed:
                break
            next_t += 1.0 / max(1.0, sim.sps)
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = loop.time()      # fell behind (or sps was lowered) -> resync, don't burst-catch-up
    except asyncio.CancelledError:
        pass


def make_handler(cfg, state):
    async def handler(ws):
        # Capacity gate: one env per session is CPU-bound; refuse beyond the cap instead of thrashing.
        if state["sessions"] >= cfg.max_sessions:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "error", "error": "Server at capacity — try again shortly."}))
                await ws.close()
            return
        state["sessions"] += 1
        sim = Sim(cfg.env_id, cfg.sps)
        task = asyncio.create_task(stepper(sim, ws))
        print(f"[server] +session ({state['sessions']}/{cfg.max_sessions})", flush=True)
        try:
            await ws.send(sim.actor_list_msg())
            async for raw in ws:
                try:
                    m = json.loads(raw)
                except Exception:
                    continue
                cmd = m.get("cmd")
                if cmd == "restart":
                    sim.restart()
                elif cmd == "mode":
                    sim.set_mode(m.get("mode", "test"))
                elif cmd == "speed":
                    sim.sps = max(1.0, float(m.get("sps", sim.sps)))
                elif cmd == "actor":
                    sim.set_actor(m.get("name", ""))
                elif cmd == "pause":
                    sim.set_paused(m.get("value", not sim.paused))
                elif cmd == "no_reset":
                    sim.set_no_reset(m.get("value", False))
                elif cmd == "list_actors":
                    await ws.send(sim.actor_list_msg())
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            sim.close()
            state["sessions"] -= 1
            print(f"[server] -session ({state['sessions']}/{cfg.max_sessions})", flush=True)
    return handler


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=_env_int("PORT", 8765))
    ap.add_argument("--env", default=os.environ.get("ENV_ID", "Walker2d-v5"))
    ap.add_argument("--sps", type=float, default=_env_float("SPS", 30.0))
    ap.add_argument("--max-sessions", type=int, default=_env_int("MAX_SESSIONS", 6))
    a = ap.parse_args()

    class Cfg:
        host, port, env_id, sps, max_sessions = a.host, a.port, a.env, a.sps, a.max_sessions
    cfg = Cfg()
    state = {"sessions": 0}
    print(f"[server] actors: {sorted(REGISTRY)}  |  ws://{cfg.host}:{cfg.port}  "
          f"|  max_sessions={cfg.max_sessions}  sps={cfg.sps}", flush=True)
    async with websockets.serve(make_handler(cfg, state), cfg.host, cfg.port):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())
