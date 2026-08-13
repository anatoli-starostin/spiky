"""exp_c02 — MJX batched-simulation throughput for Walker2d (issue #75).

Loads the SAME walker2d.xml that Gymnasium's Walker2d-v5 uses, so the model is
identical to the SAC baseline's (MJX re-expresses MuJoCo's dynamics in JAX; see the
caveats in the issue comment — the solver is not bit-for-bit identical on contacts).

Benchmarks env-steps/sec for a batch of N parallel envs stepped in one jitted,
vmapped kernel, against the measured 14,835 steps/s single-env MuJoCo CPU figure.

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python bench_mjx.py
"""
import functools, json, os, subprocess, time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_BASELINE_SPS = 14_835        # measured single-env Walker2d-v5 (gymnasium/mujoco, CPU)


# gymnasium lives in the OTHER venv (the torch/SB3 one running the SAC baseline);
# this venv deliberately has no torch, so point at the asset by path.
GYM_ASSETS = os.path.expanduser(
    "~/projects/spiky/.venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets")


def gym_walker2d_xml():
    """The exact walker2d_v5.xml that gymnasium's Walker2d-v5 loads."""
    return os.environ.get("WALKER2D_XML", os.path.join(GYM_ASSETS, "walker2d_v5.xml"))


def load_model(xml_path, solver_iterations=None):
    m = mujoco.MjModel.from_xml_path(xml_path)
    if solver_iterations:
        m.opt.iterations = solver_iterations
        m.opt.ls_iterations = max(4, solver_iterations // 2)
    return m


def gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        u, m = (x.strip() for x in out.split(","))
        return int(u), int(m)
    except Exception:
        return None, None


def bench(mx, n_envs, n_steps=200, seed=0):
    """Step n_envs in parallel for n_steps; return (steps_per_sec, compile_s)."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_envs)

    @jax.jit
    def init(keys):
        def one(k):
            d = mjx.make_data(mx)
            # small reset noise, matching the env's +/-5e-3 uniform
            qpos = d.qpos + jax.random.uniform(k, d.qpos.shape, minval=-5e-3, maxval=5e-3)
            return d.replace(qpos=qpos)
        return jax.vmap(one)(keys)

    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    t0 = time.perf_counter()
    dx = init(keys)
    dx = step(mx, dx)
    jax.block_until_ready(dx.qpos)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        dx = step(mx, dx)
    jax.block_until_ready(dx.qpos)
    dt = time.perf_counter() - t0
    return n_envs * n_steps / dt, compile_s


def main():
    xml = gym_walker2d_xml()
    print(f"model: {xml}")
    m = load_model(xml)
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"solver_iterations={m.opt.iterations} timestep={m.opt.timestep} "
          f"frame_skip=4 (v5)")
    mx = mjx.put_model(m)
    print(f"  MJX model on {jax.devices()[0].device_kind}\n")

    rows = []
    print(f"{'n_envs':>8} {'steps/s':>12} {'vs CPU':>9} {'compile':>9} "
          f"{'gpu%':>6} {'mem MB':>8}")
    for n in (1, 64, 256, 1024, 4096, 8192, 16384):
        try:
            sps, comp = bench(mx, n)
        except Exception as e:
            print(f"{n:>8}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            break
        util, mem = gpu_mem_mb()
        print(f"{n:>8} {sps:>12,.0f} {sps/CPU_BASELINE_SPS:>8.1f}x {comp:>8.1f}s "
              f"{util if util is not None else '-':>6} {mem if mem is not None else '-':>8}")
        rows.append(dict(n_envs=n, steps_per_sec=round(sps, 1),
                         speedup_vs_cpu=round(sps / CPU_BASELINE_SPS, 2),
                         compile_s=round(comp, 2), gpu_util=util, gpu_mem_mb=mem))

    out = dict(cpu_baseline_sps=CPU_BASELINE_SPS, xml=xml,
               device=jax.devices()[0].device_kind, results=rows)
    with open(os.path.join(HERE, "bench_mjx_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote bench_mjx_results.json")


if __name__ == "__main__":
    main()
