# Deploying the "Spiking LUT (handcrafted SNN)" actor — handoff for the deploy engineer

This actor is different from every other one in the demo: it runs a **real spiking network** on the
`spiky` engine rather than a numpy forward pass. That means the image now builds a native extension, and
there is **one way to get that wrong that produces a green build and a dead container**. Read §2 before
building.

Everything else follows the normal procedure in [`ADDING_MODELS.md`](ADDING_MODELS.md).

---

## 1. What is being deployed

| | |
|---|---|
| actor | `server/actors/spiking_lut.py`, dropdown name **"Spiking LUT (handcrafted SNN)"** |
| weights | `server/models/spiking_lut_actor.npz` (47.8 KiB) |
| what it is | a handcrafted 3-stage spiking network — 3024 neurons, 26,344 synapses, 309 ticks per inference — that reproduces the Walker2d LUT teacher. No training: every weight and delay is derived analytically from the teacher's own tables. |
| accuracy | R² 0.979–0.996 against the quantised-input LUT it reproduces; 0.90–0.98 against the true teacher |
| speed | **12.5 ms per `act()` (~80 Hz)** on CPU, measured. The demo runs at `SPS: 30` (33 ms/step). |
| normalisation | ships and applies **its own** `obs_mean`/`obs_var` — it does **not** use `models/walker_dataset_stats.json`, whose stats differ (max \|mean\| diff 1.29) |

---

## 1b. Pinned engine version

The image compiles `spiky_cuda` from the repo's own `native/spiky` + `src/spiky`, so **the checkout you
build from IS the engine you deploy**. This actor was built and validated against:

```
spnet commit  79838d87d7612bfbdeb6cc128b1c14c58b3cdf95
              "Add per-neuron reset mode (constant/subtractive) + refractory period + LIFNeuronMeta (#96)"
```

Two commits in that history are **required**, not optional:

| commit | why it is required |
|---|---|
| `79838d87` | adds **`LIFNeuronMeta`**, which all three stages construct. Without it the actor raises on import and the container dies at startup. |
| `61a2a5d8` | fixes `_grow_explicit(weights=)` putting ~60% of explicit weights on the wrong edges. The actor builds its whole network through that call, so without the fix the walker misbehaves **silently** — no error anywhere. |

Verify both are in the tree you are about to build:

```sh
git merge-base --is-ancestor 79838d87 HEAD && echo "LIFNeuronMeta: ok"
git merge-base --is-ancestor 61a2a5d8 HEAD && echo "weight-alignment fix: ok"
```

---

## 2. ⚠ The one thing that will bite you: where `spiky_cuda` gets compiled

`native/spiky/setup.py` picks its extension type from `torch.cuda.is_available()` **at build time**:

| build host sees | setup.py builds | works in a GPU-less container? |
|---|---|---|
| **no GPU** | `CppExtension`, `-DNO_CUDA` | ✅ yes — this is what we want |
| a GPU | `CUDAExtension`, links `-lcuda` | ❌ **no** — imports fine on the build host, fails inside the container |

**So the image must be built on a host with no visible GPU** (a normal cloud VM or the demo box itself is
fine). The Dockerfile compiles the extension *inside* the image precisely so this resolves correctly and
automatically — do **not** try to shortcut it by copying a prebuilt `.so` in.

The repo-root `.dockerignore` also excludes `native/spiky/build/` and `native/**/*.so`, so a developer's
locally-built CUDA artefacts cannot leak into the context. **Do not remove those two exclusion lines.**

**Verify before deploying** (this is the check that catches the failure mode):

```sh
docker compose run --rm server python -c "import spiky_cuda; print('spiky_cuda ok')"
```

If that errors with something about `libcuda.so` or a missing CUDA driver, the extension was built as the
CUDA variant — rebuild on a GPU-less host with `--no-cache`.

---

## 3. Build-context change (new in this PR)

The build context moved from `./server` to the **repo root**, because the actor needs `native/spiky` and
`src/spiky`, which live outside `server/`:

```yaml
# docker-compose.yml
build:
  context: ../..
  dockerfile: landing/walker2d-viz/server/Dockerfile
```

A repo-root `.dockerignore` allow-lists exactly what the build needs. **Measured: the daemon receives
2.6 MB, against 13.4 GB for the unfiltered repo.** If a build is suddenly slow or the daemon is shipping
gigabytes, that file has been broken or lost.

Because the context is now the repo root, **the deploy host needs the repo checked out, not just the
`walker2d-viz/` folder.**

---

## 4. Deploy sequence

Per `ADDING_MODELS.md` §4 — a **rebuild** is required; a restart is not enough, since actors and models are
COPYd into the image at build time.

```sh
# 0) on a GPU-LESS deploy host, check out the branch containing this PR (or landing after merge)
git checkout <branch>

# 1) ship the repo to the VM (the whole repo now, because of the build-context change)
rsync -av --exclude '.git' ./ nucstar@YOUR_SERVER_HOST:/home/nucstar/spiky/

# 2) rebuild + restart
ssh nucstar@YOUR_SERVER_HOST
cd /home/nucstar/spiky/landing/walker2d-viz
sudo docker compose build --no-cache server        # first build compiles the extension; several minutes
sudo docker compose run --rm server python -c "import spiky_cuda; print('spiky_cuda ok')"   # §2 CHECK
sudo docker compose up -d

# 3) verify the actor registered
sudo docker compose logs server | tail             # the "[server] actors: [...]" line must list it
```

Then confirm end-to-end over the real websocket:

```python
import asyncio, json, websockets
async def m():
    async with websockets.connect("wss://YOUR_SERVER_HOST") as ws:
        print(json.loads(await ws.recv())["actors"])
asyncio.run(m())
```

`"Spiking LUT (handcrafted SNN)"` must appear in that list. The client needs no change — viewers just
reload the Pages app.

---

## 5. Post-deploy validation

1. Select **"Spiking LUT (handcrafted SNN)"** in the Actor dropdown; the walker should walk, behaving
   close to the existing **LUT teacher** actor (it is a spiking reimplementation of the same policy).
2. Watch `docker compose logs server` for the first minute — a per-step exception would show there.
3. Check CPU. See §6.

---

## 6. ⚠ Capacity: this actor is far heavier than the others

Every other actor is a small numpy matmul. This one simulates 3024 neurons for 309 ticks per step.

- **12.5 ms per step**, against a **33 ms** budget at `SPS: 30` — fine for **one** viewer.
- `MAX_SESSIONS` defaults to **6**, and each session runs its own env and its own actor instance.
  Six concurrent viewers on this actor is **~75 ms of CPU per simulated step on a 2-vCPU box**, which will
  not keep up.

**Recommendation:** deploy first, then measure CPU under real load. If concurrent viewers pick this actor
and the stream degrades, lower `MAX_SESSIONS` (env var in `docker-compose.yml`, no rebuild needed) rather
than reverting the actor. This has **not** been load-tested — the 12.5 ms figure is single-session,
measured locally.

---

## 7. Rollback

The actor is additive: no existing actor, model, or client code was modified. To remove it, delete
`server/actors/spiking_lut.py` and rebuild — everything else keeps working. To roll back the build-context
change as well, revert this PR's `docker-compose.yml`, `Dockerfile`, `requirements.txt` and `.dockerignore`
together (they are interdependent).
