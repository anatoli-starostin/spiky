"""Walker2d visualization/control server.

Runs a Gymnasium MuJoCo Walker2d env and streams state to WebSocket clients as JSON, accepting control
messages (restart / mode / speed / actor / pause / no_reset). Actors are auto-discovered from actors/.

Concurrency model: **one independent session per WebSocket connection**. Each connected browser gets its
own Sim (its own env instance, selected actor, pause/mode/free-fall state and stepping task), so multiple
viewers never fight over a shared env — required for a public multi-viewer demo. A global cap
(MAX_SESSIONS) bounds CPU: extra connections get a {"type":"server_full"} message, then a clean close.

Config via env vars (CLI flags override): HOST, PORT, ENV_ID, SPS, MAX_SESSIONS.
Run:  python server.py [--host 0.0.0.0] [--port 8765] [--env Walker2d-v5] [--sps 30] [--max-sessions 8]
"""
import argparse
import asyncio
import contextlib
import json
import os
import time

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

# Auto-stop: after this many seconds of UNINTERRUPTED walking, a session is switched to the "zero" policy so
# a forgotten/idle viewer stops consuming compute. Any user interaction (model switch, pause/resume, restart,
# speed change) resets the timer. The "zero" policy is itself idle (~0 CPU): the stepper lets the body settle
# for ZERO_SETTLE frames (so it topples naturally) then goes quiet until the next interaction. AUTO_IDLE_S is
# env-overridable (mostly for tests).
AUTO_IDLE_S = float(os.environ.get("AUTO_IDLE_S", 180.0))
ZERO_SETTLE = 60                        # ~2 s at 30 sps
DEFAULT_ACTOR = "fastlut_lse (exp19)"   # preferred default actor if present (else "random", else first discovered)
MIN_SPS = 1.0
MAX_SPS = 60.0                          # hard ceiling on steps/sec: higher sps = more step+inference+send/sec
                                        # = more server CPU/bandwidth, so cap it (authoritative server-side guard)


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
        self.actor_name = (DEFAULT_ACTOR if DEFAULT_ACTOR in self.registry
                           else "random" if "random" in self.registry
                           else next(iter(self.registry)))
        self.actor = self.registry[self.actor_name](self.env.action_space)
        self.mode = "test"                       # "test" | "train"
        self.paused = False                      # when True, the stepper idles (no step, no broadcast)
        # One wake event drives ALL idle states (pause AND the settled zero policy): cleared while idle so the
        # stepper awaits it; any interaction sets it (see touch()). ~0 CPU / ~0 bandwidth while idle.
        self._wake = asyncio.Event()
        self._wake.set()
        self.no_reset = False                    # free-fall: when True, don't auto-reset on termination
        self.terminated = False
        self.sps = min(MAX_SPS, max(MIN_SPS, float(sps)))   # clamp even the startup SPS to the cap
        self.obs, _ = self.env.reset(seed=0)
        self.step_count = 0
        self.reward = 0.0
        self.ret = 0.0                            # episode return
        self._walk_since = time.monotonic()      # start of the current uninterrupted walk (auto-stop timer)
        self._auto_zeroed = False                # set once the auto-stop has fired for this walk
        self._zero_settle = ZERO_SETTLE if self.actor_name == "zero" else 0   # frames to step before idling on zero

    # ---- control ----
    def touch(self):
        """A user interaction: reset the auto-stop walk timer, re-arm it, and wake any idle stepper."""
        self._walk_since = time.monotonic()
        self._auto_zeroed = False
        self._wake.set()

    def set_actor(self, name):
        if name in self.registry:
            self.actor_name = name
            self.actor = self.registry[name](self.env.action_space)
            self._zero_settle = ZERO_SETTLE if name == "zero" else 0   # a (re)selected zero topples, then idles
            self.touch()

    async def set_actor_async(self, name):
        """Like set_actor, but constructs the actor OFF the event loop (thread executor). Some actors are
        heavy to build — e.g. the spiking-LUT actor imports torch + builds a 3024-neuron SNN (~2 s). Building
        that inline starves the single asyncio loop, so the websocket keepalive lapses and the socket drops,
        which silently kills send()-based controls (restart/pause/no-reset). Offloading keeps the loop live
        (the stepper keeps running the current actor) and swaps in the new actor once it's built."""
        if name in self.registry:
            loop = asyncio.get_event_loop()
            actor = await loop.run_in_executor(None, self.registry[name], self.env.action_space)
            self.actor_name = name
            self.actor = actor
            self._zero_settle = ZERO_SETTLE if name == "zero" else 0
            self.touch()

    def _auto_zero(self):
        """Internal (NOT a user interaction, so no touch): switch a forgotten walk to the zero policy."""
        if self.actor_name != "zero" and "zero" in self.registry:
            self.actor_name = "zero"
            self.actor = self.registry["zero"](self.env.action_space)
        self._auto_zeroed = True
        self._zero_settle = ZERO_SETTLE

    def set_mode(self, mode):
        self.mode = "train" if mode == "train" else "test"

    def set_no_reset(self, value):
        self.no_reset = bool(value)

    def set_paused(self, value):
        self.paused = bool(value)
        self.touch()                             # pause OR resume is an interaction: reset timer + wake

    def _reset_episode(self):
        """Start a fresh episode WITHOUT counting as a user interaction (no touch()).

        Used by the AUTOMATIC on-fall / on-truncation reset, which is part of one *uninterrupted* walk and
        must NOT re-arm the auto-stop timer. (Bug fix: a Walker2d episode truncates at the 1000-step limit
        ~every 33 s at 30 sps, and falls sooner — when the auto-reset called restart()->touch(), the 180 s
        auto-idle clock was reset every episode, so it never fired on a continuously-walking demo.)"""
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.reward = 0.0
        self.ret = 0.0
        self.terminated = False
        if self.actor_name == "zero":
            self._zero_settle = ZERO_SETTLE      # (re)started while on zero -> let it topple again, then idle

    def restart(self):
        """USER-initiated restart (Restart button): a fresh episode AND an interaction (resets the timer)."""
        self._reset_episode()
        self.touch()

    def step(self):
        action = self.actor.act(self.obs)
        action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        self.obs, r, terminated, truncated, _ = self.env.step(action)
        self.reward = float(r)
        self.ret += self.reward
        self.step_count += 1
        self.terminated = bool(terminated or truncated)
        # Automatic episode reset on fall/truncation. Uses _reset_episode() (NOT restart()), so it does NOT
        # touch()/re-arm the auto-stop timer — the walk is uninterrupted across episode boundaries. Guards:
        #  - no_reset (free-fall): user asked to keep stepping the fallen body instead of resetting.
        #  - actor "zero": never auto-reset a zeroed session — a forgotten walk switched to zero must lie
        #    down and idle, not spring back up (a reset would re-arm _zero_settle and loop fall->reset forever).
        if self.terminated and not self.no_reset and self.actor_name != "zero":
            self._reset_episode()

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
            # Auto-stop: a forgotten walk (untouched for AUTO_IDLE_S) -> zero policy, which then settles + idles.
            # (Skip if already on zero — a user-selected zero settles/idles on its own; don't re-settle it.)
            if (not sim.paused and not sim._auto_zeroed and sim.actor_name != "zero"
                    and (time.monotonic() - sim._walk_since) >= AUTO_IDLE_S):
                sim._auto_zero()
                try:
                    # Tell the client the model auto-switched, so its selector visibly shows "zero".
                    await ws.send(json.dumps({"type": "actor_changed", "actor": sim.actor_name}))
                except websockets.exceptions.ConnectionClosed:
                    break

            # Idle (~0 CPU / ~0 bandwidth): PAUSED, or on the zero policy once it has settled. Send exactly ONE
            # frozen/resting frame, then await sim._wake (no busy-spin) until an interaction. The connection is
            # kept alive by the websockets library's built-in ping/pong while we idle.
            if sim.paused or (sim.actor_name == "zero" and sim._zero_settle <= 0):
                try:
                    await ws.send(sim.state_msg())
                except websockets.exceptions.ConnectionClosed:
                    break
                sim._wake.clear()
                await sim._wake.wait()
                next_t = loop.time()      # resync pacing on wake (don't burst-catch-up the idle gap)
                continue

            sim.step()                    # normal walk, OR the brief zero-settle so the body topples naturally
            if sim.actor_name == "zero" and sim._zero_settle > 0:
                sim._zero_settle -= 1
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
                await asyncio.sleep(0)     # CRITICAL: always yield so the receiver coroutine gets serviced.
                # A heavy actor.act() (e.g. the spiking-LUT SNN, ~12-16 ms, more under load) keeps the loop
                # perpetually behind, so the pacing sleep above is skipped; and ws.send() for a small state
                # frame usually completes WITHOUT suspending. With no suspension point the single-threaded loop
                # never returns to the message receiver, so restart/pause/no_reset pile up unprocessed while the
                # stepper keeps walking -> "robot walks but the buttons are dead". Light actors finish inside the
                # frame budget, hit the delay>0 sleep, and yield -- which is why only the spiking actor broke.
    except asyncio.CancelledError:
        pass


def make_handler(cfg, state):
    async def handler(ws):
        # Capacity gate: one env per session is CPU-bound; refuse beyond the cap instead of thrashing.
        # Accept the socket just long enough to send a structured "server_full" message, then close
        # cleanly, so the client can show a friendly overload banner rather than see a raw drop.
        if state["sessions"] >= cfg.max_sessions:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({
                    "type": "server_full",
                    "message": "The demo server is at capacity, please come back later.",
                }))
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
                    sim.sps = min(MAX_SPS, max(MIN_SPS, float(m.get("sps", sim.sps))))   # clamp to [1, 60]
                    sim.touch()                          # speed change is an interaction: reset the auto-stop timer
                elif cmd == "actor":
                    await sim.set_actor_async(m.get("name", ""))   # build off-loop so the ws keepalive survives
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
    ap.add_argument("--max-sessions", type=int, default=_env_int("MAX_SESSIONS", 48))
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
