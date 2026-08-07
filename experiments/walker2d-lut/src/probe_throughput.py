"""Phase-1 throughput probe: MuJoCo-Warp batched Walker2d on the H100.
Measures env-steps/s (physics-steps/4, frame_skip=4) and GPU utilization vs env count.
CUDA-graph-captured rollout so per-step python overhead is removed (purejaxrl-style).
SHORT probe only — no training."""
import os, time, threading, subprocess, sys
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

XML = os.path.join(os.path.dirname(mujoco.__file__), "..", "gymnasium",
                   "envs", "mujoco", "assets", "walker2d_v5.xml")
XML = os.path.abspath(XML)
FRAME_SKIP = 4          # Walker2d-v5: one env-step = 4 physics steps
CAP_STEPS = 20          # physics steps per captured graph
REPLAYS = 40            # graph replays in the timed region


def gpu_util_sampler(stop_evt, out):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=2)
            u, m = r.stdout.strip().split("\n")[0].split(",")
            out.append((float(u), float(m)))
        except Exception:
            pass
        time.sleep(0.04)


def probe(m, mjm, N):
    d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=N)
    rng = np.random.default_rng(0)
    ctrl = wp.array(rng.uniform(-1, 1, (N, mjm.nu)).astype(np.float32), dtype=wp.float32)
    # warmup + assign controls
    d.ctrl = ctrl
    for _ in range(3):
        mjw.step(m, d)
    wp.synchronize()
    # capture a CAP_STEPS rollout into a CUDA graph
    with wp.ScopedCapture() as cap:
        for _ in range(CAP_STEPS):
            mjw.step(m, d)
    graph = cap.graph
    wp.synchronize()
    # timed region
    stop = threading.Event(); samples = []
    th = threading.Thread(target=gpu_util_sampler, args=(stop, samples)); th.start()
    t0 = time.time()
    for _ in range(REPLAYS):
        wp.capture_launch(graph)
    wp.synchronize()
    dt = time.time() - t0
    stop.set(); th.join()
    phys = N * CAP_STEPS * REPLAYS
    phys_per_s = phys / dt
    env_per_s = phys_per_s / FRAME_SKIP
    util = np.array(samples) if samples else np.zeros((1, 2))
    return dict(N=N, dt=dt, phys_per_s=phys_per_s, env_per_s=env_per_s,
                gpu_util_mean=float(util[:, 0].mean()), gpu_util_max=float(util[:, 0].max()),
                mem_used_mb=float(util[:, 1].max()))


def main():
    print("XML:", XML, "exists:", os.path.exists(XML))
    mjm = mujoco.MjModel.from_xml_path(XML)
    # branch's adopted throughput solver setting: 10/8
    mjm.opt.iterations = 10
    mjm.opt.ls_iterations = 8
    print(f"nu={mjm.nu} nq={mjm.nq} nv={mjm.nv} timestep={mjm.opt.timestep} "
          f"solver={mjm.opt.solver} iters={mjm.opt.iterations}/{mjm.opt.ls_iterations}")
    m = mjw.put_model(mjm)
    rows = []
    for N in (1024, 4096, 8192, 16384, 32768):
        try:
            r = probe(m, mjm, N)
            rows.append(r)
            print(f"N={N:>6} | env-steps/s {r['env_per_s']:>12,.0f} | "
                  f"phys/s {r['phys_per_s']:>12,.0f} | GPU {r['gpu_util_mean']:4.0f}% "
                  f"(max {r['gpu_util_max']:.0f}%) | mem {r['mem_used_mb']:,.0f} MB", flush=True)
        except Exception as e:
            print(f"N={N:>6} | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            break
    import json
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "probe_results.json"), "w"), indent=1)
    if rows:
        best = max(rows, key=lambda r: r["env_per_s"])
        print(f"\nBEST: N={best['N']} -> {best['env_per_s']:,.0f} env-steps/s "
              f"({best['env_per_s']/135:.0f}x the old 135/s CPU single-env)")


if __name__ == "__main__":
    main()
