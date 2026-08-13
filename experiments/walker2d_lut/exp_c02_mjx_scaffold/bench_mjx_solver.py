"""exp_c02b — MJX throughput in *env-steps* (not physics steps), and the cost of
the solver setting (issue #75).

Two corrections over bench_mjx.py, both of which matter for an honest comparison:

  1. Walker2d-v5 uses frame_skip=4 — one env-step is FOUR mjx.step calls. The CPU
     baseline of 14,835 steps/s is env-steps. So physics-steps/s must be divided by
     4 before comparing. bench_mjx.py reported physics steps.
  2. gymnasium's walker2d_v5.xml ships solver iterations=100 (fine on CPU, where the
     solver exits early on an easy contact configuration; expensive on GPU, where a
     vmapped batch pays the worst case every step). Reducing iterations is standard
     MJX practice — but it CHANGES THE DYNAMICS, so it is a comparability tradeoff,
     not a free win. Measured here so the tradeoff is explicit.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false python bench_mjx_solver.py
"""
import json, os, subprocess, time

import jax
import mujoco
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_BASELINE_ENV_SPS = 14_835     # measured single-env Walker2d-v5 on CPU (env-steps)
FRAME_SKIP = 4                    # Walker2d-v5
GYM_ASSETS = os.path.expanduser(
    "~/projects/spiky/.venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets")
XML = os.environ.get("WALKER2D_XML", os.path.join(GYM_ASSETS, "walker2d_v5.xml"))


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        return tuple(int(x) for x in out.split(","))
    except Exception:
        return (None, None)


def bench(iters, ls_iters, n_envs, n_env_steps=50, seed=0):
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.iterations = iters
    m.opt.ls_iterations = ls_iters
    mx = mjx.put_model(m)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)

    @jax.jit
    def init(keys):
        return jax.vmap(lambda k: mjx.make_data(mx).replace(
            qpos=mjx.make_data(mx).qpos + jax.random.uniform(
                k, (m.nq,), minval=-5e-3, maxval=5e-3)))(keys)

    @jax.jit
    def env_step(dx):
        """One Walker2d-v5 env-step = FRAME_SKIP physics steps."""
        def body(d, _):
            return jax.vmap(mjx.step, in_axes=(None, 0))(mx, d), None
        dx, _ = jax.lax.scan(body, dx, None, length=FRAME_SKIP)
        return dx

    dx = init(keys)
    dx = env_step(dx)
    jax.block_until_ready(dx.qpos)

    t0 = time.perf_counter()
    for _ in range(n_env_steps):
        dx = env_step(dx)
    jax.block_until_ready(dx.qpos)
    dt = time.perf_counter() - t0
    return n_envs * n_env_steps / dt


def main():
    print(f"model {XML}  (frame_skip={FRAME_SKIP}; CPU baseline "
          f"{CPU_BASELINE_ENV_SPS:,} env-steps/s single env)\n")
    rows = []
    print(f"{'solver':>14} {'n_envs':>8} {'env-steps/s':>14} {'vs CPU':>9} "
          f"{'gpu%':>6} {'mem MB':>8}")
    for iters, ls in ((100, 50), (10, 8), (4, 4)):
        for n in (4096, 16384, 32768):
            try:
                sps = bench(iters, ls, n)
            except Exception as e:
                print(f"{f'{iters}/{ls}':>14} {n:>8}  FAILED: "
                      f"{type(e).__name__}: {str(e)[:50]}")
                continue
            util, mem = gpu_stats()
            print(f"{f'{iters}/{ls}':>14} {n:>8} {sps:>14,.0f} "
                  f"{sps/CPU_BASELINE_ENV_SPS:>8.1f}x {util:>6} {mem:>8}")
            rows.append(dict(iterations=iters, ls_iterations=ls, n_envs=n,
                             env_steps_per_sec=round(sps, 1),
                             speedup_vs_cpu=round(sps / CPU_BASELINE_ENV_SPS, 2),
                             gpu_util=util, gpu_mem_mb=mem))
    with open(os.path.join(HERE, "bench_mjx_solver_results.json"), "w") as f:
        json.dump(dict(cpu_baseline_env_sps=CPU_BASELINE_ENV_SPS,
                       frame_skip=FRAME_SKIP, xml=XML,
                       note="env-steps, not physics steps", results=rows), f, indent=1)
    print("\nwrote bench_mjx_solver_results.json")


if __name__ == "__main__":
    main()
