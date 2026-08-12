"""exp012: where does a steady-state round actually spend its wall clock?

Times one round end to end, split into the four stages the loop is made of, and separately
confirms device placement for the two things that are supposed to be on the GPU: the synapse
growth / build path and the per-tick simulation.

METHOD. Stage timers wrap the real primitives -- `T.build`, `T.score`, `T.mutate` -- and the
engine calls inside them (`_grow_explicit`, `add_connections`, `compile`, `process_ticks`,
`export_neuron_data`) are timed by monkeypatching the CLASS methods in this process only, so
nothing about the running job changes. `torch.cuda.synchronize()` brackets every GPU timer,
without which CUDA's asynchrony would credit the simulation's time to whatever host code ran
next.

    python tiny_profile.py --pool 512 --batch 256 --rounds 3
"""
import argparse
import json
import time

import numpy as np
import torch

import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder


TIMERS = {}


SEEN = {}


def _wrap(cls, name, key):
    orig = getattr(cls, name)

    def timed(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(self, *a, **k)
        torch.cuda.synchronize()
        TIMERS.setdefault(key, []).append(time.perf_counter() - t)
        if torch.is_tensor(r) and key not in SEEN:
            SEEN[key] = dict(device=str(r.device), dtype=str(r.dtype),
                             shape=list(r.shape), numel=int(r.numel()),
                             bytes=int(r.numel() * r.element_size()))
        for x in list(a) + list(k.values()):
            if torch.is_tensor(x):
                SEEN.setdefault(key + ".arg_devices", set()).add(str(x.device))
        return r
    setattr(cls, name, timed)
    return orig


def device_report(H):
    """Which tensors live where. Everything is observed on the REAL scored round -- probing
    the engine out of band (e.g. an extra export with a different batch size) reuses buffers
    sized for the previous call and faults."""
    sp, dev = H["spnet"], H["device"]
    nid = sp.get_neuron_ids_by_meta(0)
    return dict(cuda_available=torch.cuda.is_available(), torch_device_requested=dev,
                spnet_neuron_ids_device=str(nid.device), spnet_neuron_ids_dtype=str(nid.dtype),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--cull", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--w-max", type=float, default=60.0)
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from spiky.spnet.spnet import SpikingNet
    from spiky.util.synapse_growth import SynapseGrowthEngine
    _wrap(SpikingNet, "process_ticks", "engine.process_ticks")
    _wrap(SpikingNet, "export_neuron_data", "engine.export_neuron_data")
    _wrap(SpikingNet, "add_connections", "engine.add_connections")
    _wrap(SpikingNet, "compile", "engine.compile")
    _wrap(SynapseGrowthEngine, "_grow_explicit", "engine._grow_explicit")

    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    pool = [T.random_genome(rng, a.w_max) for _ in range(a.pool)]
    ewma = np.full(a.pool, np.nan)

    # one untimed warm-up round: first build pays CUDA context + kernel compilation
    H = T.build(pool, device=a.device)
    T.score(H, Xp[:a.batch], Yp[:a.batch], enc)
    TIMERS.clear()

    dev = device_report(H)
    TIMERS.clear()
    SEEN.clear()

    stages = {k: [] for k in ("genome", "build", "score", "host")}
    for rnd in range(a.rounds):
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, rnd)

        t = time.perf_counter()
        H = T.build(pool, device=a.device, seed=1)
        torch.cuda.synchronize()
        stages["build"].append(time.perf_counter() - t)

        t = time.perf_counter()
        s = T.score(H, Xb, Yb, enc)
        torch.cuda.synchronize()
        stages["score"].append(time.perf_counter() - t)

        t = time.perf_counter()
        mse = s["mse"]
        ewma = np.where(np.isnan(ewma), mse, 0.5 * mse + 0.5 * ewma)
        i = int(np.argmin(mse))
        T.affine_ceiling_and_r(s["first"][:, i, :], T.target_offsets(Yb))
        stages["host"].append(time.perf_counter() - t)

        t = time.perf_counter()
        for _ in range(a.cull):
            p = int(rng.integers(0, a.pool))
            if a.crossover:
                q = int(rng.integers(0, a.pool))
                kid = pool[p] if q == p else T.crossover(pool[p], pool[q], rng)
            else:
                kid = pool[p]
            pool[int(rng.integers(0, a.pool))] = T.mutate(kid, rng, a.w_max)
        stages["genome"].append(time.perf_counter() - t)

    # ---- inside the 'score' stage: a faithful re-run of harness.run_episode's body, step by
    # step. The stage timer says score is ~95 % of the round while the engine calls inside it
    # are ~4 %; this says WHICH host-side step owns the rest.
    from spiky.spnet.spnet import NeuronDataType
    sp, ids, P, devs = H["spnet"], H["ids"], H["P"], H["device"]
    B = a.batch
    X = Xb
    sub = {}

    def tick(k, t):
        sub[k] = sub.get(k, 0.0) + (time.perf_counter() - t)

    t = time.perf_counter(); ticks = enc(X); tick("encode", t)
    cols = ids[2]
    t = time.perf_counter()
    va = np.zeros((B, 32, cols.size), np.float32)
    tick("alloc va (285 MB)", t)
    t = time.perf_counter()
    for b in range(B):
        for j in range(T.N_IN):
            va[b, ticks[b, j], j::T.N_IN] = 200.0
    tick("fill va (python loop)", t)
    t = time.perf_counter()
    sp_ids = np.broadcast_to(cols.reshape(1, 1, cols.size),
                             (B, 32, cols.size)).astype(np.int32)
    sp_ids = sp_ids.copy()
    tick("build sparse ids (285 MB)", t)
    t = time.perf_counter()
    tv = torch.tensor(va, device=devs)
    ti = torch.tensor(sp_ids, device=devs)
    torch.cuda.synchronize()
    tick("H2D copy of both (570 MB)", t)
    oid = torch.tensor(ids[3], dtype=torch.int32, device=devs)
    t = time.perf_counter()
    sp.process_ticks(n_ticks_to_process=96, batch_size=B, n_input_ticks=32,
                     input_values=tv, sparse_input=ti, do_train=False,
                     do_record_voltage=False, do_reset_context=True, _stdp_period=32)
    torch.cuda.synchronize()
    tick("process_ticks (GPU sim)", t)
    t = time.perf_counter()
    R = sp.export_neuron_data(oid, B, NeuronDataType.Spike, 0, 95)
    torch.cuda.synchronize()
    tick("export_neuron_data", t)
    t = time.perf_counter()
    Rn = R.cpu().numpy()
    tick("D2H .cpu().numpy() (302 MB)", t)
    t = time.perf_counter()
    Rn = Rn.reshape(B, P, T.N_OUT, 96)
    w = Rn[..., 96 - 32:]
    first = np.where(w.any(-1), w.argmax(-1), 32).astype(np.float64)
    tick("numpy readout (argmax etc)", t)

    N = a.rounds
    eng = {k: float(np.sum(v)) / N for k, v in TIMERS.items()}
    st = {k: float(np.mean(v)) for k, v in stages.items()}
    total = sum(st.values())

    # the two big host<->device transfers the readout path performs every round
    B, P = a.batch, a.pool
    dev["export_bytes_per_round_full_batch"] = B * P * T.N_OUT * T.N_TICKS * 4
    h2d = B * 32 * (P * T.N_IN) * 4 + B * 32 * (P * T.N_IN) * 4      # values + sparse ids
    dev["input_bytes_per_round"] = int(h2d)

    seen = {k: (sorted(v) if isinstance(v, set) else v) for k, v in SEEN.items()}
    print(json.dumps(dev, indent=1))
    print("\nOBSERVED ON THE REAL ROUND:")
    print(json.dumps(seen, indent=1))
    print(f"\nROUND BREAKDOWN  (pool {a.pool}, batch {a.batch}, mean of {N} rounds)")
    print(f"  total round                    {total * 1000:9.1f} ms")
    for k in ("genome", "build", "score", "host"):
        print(f"    {k:28s} {st[k] * 1000:9.1f} ms   {100 * st[k] / total:5.1f} %")
    print("\n  ENGINE CALLS INSIDE THOSE (summed per round)")
    for k, v in sorted(eng.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v * 1000:9.1f} ms   {100 * v / total:5.1f} %")
    inside_score = eng.get("engine.process_ticks", 0) + eng.get("engine.export_neuron_data", 0)
    inside_build = (eng.get("engine._grow_explicit", 0) + eng.get("engine.add_connections", 0)
                    + eng.get("engine.compile", 0))
    print(f"\n  score  = {st['score'] * 1000:.1f} ms, of which engine {inside_score * 1000:.1f} ms"
          f"  -> host-side readout/encoding {((st['score'] - inside_score) * 1000):.1f} ms")
    print(f"  build  = {st['build'] * 1000:.1f} ms, of which engine {inside_build * 1000:.1f} ms"
          f"  -> host-side packing {((st['build'] - inside_build) * 1000):.1f} ms")
    print(f"\n  export tensor per round: {dev['export_bytes_per_round_full_batch'] / 1e6:.0f} MB "
          f"D2H   ·  input tensors: {dev['input_bytes_per_round'] / 1e6:.0f} MB H2D")
    ssum = sum(sub.values())
    print(f"\n  INSIDE run_episode, step by step (one pass, {ssum * 1000:.1f} ms total)")
    for k, v in sorted(sub.items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v * 1000:8.1f} ms   {100 * v / ssum:5.1f} %")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(dict(config=vars(a), device=dev, observed=seen, stages=st,
                                      engine=eng, run_episode_steps=sub, total_s=total)),
                      f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
