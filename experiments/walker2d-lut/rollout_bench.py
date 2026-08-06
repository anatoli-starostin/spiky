"""Rollout throughput vs env count for the actual training-style loop (policy act +
env.step, no update), to compare against the graph-captured pure-physics ceiling and
to size N. SHORT probe. No training."""
import os, time, threading, subprocess, sys
import numpy as np
import torch
from warp_env import WarpWalker2dVecEnv
from models import REGISTRY


def util_sampler(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, timeout=2)
            out.append(float(r.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.04)


def bench(N, steps=200, graph=False):
    env = WarpWalker2dVecEnv(num_envs=N, seed=0)
    if graph:
        env.build_physics_graph()
    ac = REGISTRY["mlp"](env.obs_dim, env.act_dim).to("cuda")
    obs = env.reset()
    for _ in range(10):                       # warmup (compile)
        a, _, _ = ac.act(obs); obs, *_ = env.step(a)
    torch.cuda.synchronize()
    stop = threading.Event(); s = []
    th = threading.Thread(target=util_sampler, args=(stop, s)); th.start()
    t0 = time.time()
    for _ in range(steps):
        a, _, _ = ac.act(obs)
        obs, r, term, trunc = env.step(a)
    torch.cuda.synchronize()
    dt = time.time() - t0
    stop.set(); th.join()
    return dict(N=N, env_per_s=N * steps / dt,
                gpu_util=float(np.mean(s)) if s else 0.0)


if __name__ == "__main__":
    import json
    rows = []
    for N in (4096, 8192, 16384, 32768):
        try:
            e = bench(N, graph=False)
            g = bench(N, graph=True)
            rows.append(dict(N=N, eager=e, graph=g))
            print(f"N={N:>6} | eager {e['env_per_s']:>10,.0f}/s ({e['gpu_util']:2.0f}%) "
                  f"-> phys-graph {g['env_per_s']:>10,.0f}/s ({g['gpu_util']:2.0f}%) "
                  f"| {g['env_per_s']/e['env_per_s']:.2f}x", flush=True)
        except Exception as ex:
            print(f"N={N:>6} FAILED {type(ex).__name__}: {str(ex)[:100]}", flush=True); break
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "rollout_bench_graph.json"), "w"), indent=1)
