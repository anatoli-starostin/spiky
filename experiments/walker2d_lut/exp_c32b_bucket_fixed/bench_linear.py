"""exp_c32b — measure what the linear-recurrence membrane actually saves.

Three front-ends, one harness, one subprocess each (peak memory is a high-water mark that
never comes down, so two models in one process report the max of the pair):

  * c32b-linear    : this port, membrane via the exact O(N) first-order recurrence
  * c32b-quadratic : the SAME model with the reference O(N^2) pairwise formulation, so the
                     comparison isolates the membrane and nothing else
  * c31-pure       : exp_c31's PureLIF, the most expensive front-end in the chapter

Measured on the actor-loss BACKWARD at the training batch (512), because the pairwise
tensor is held live across the backward and a forward-only number understates it by
roughly half. FLOPs come from XLA's own cost analysis of the compiled HLO, not an estimate.

The two c32b variants are also asserted to AGREE -- the recurrence is exact, so if they
disagree beyond fp32 noise the linear path is wrong and the timing is meaningless.

Usage:
  python bench_linear.py --all          # driver, one subprocess per variant
  python bench_linear.py --variant c32b-linear
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
OBS, ACT, TPH, HEADS, BATCH = 17, 6, 32, 1, 512
N_BUCKETS, NAP = 16, 6
VARIANTS = ("c32b-linear", "c32b-quadratic", "c31-pure")


def build(variant, key):
    import jax.numpy as jnp
    sys.path.insert(0, os.path.join(D, "exp_c31_pure_lif"))
    sys.path.insert(0, HERE)

    if variant.startswith("c32b"):
        import jax_bucket_lif as M
        if variant.endswith("quadratic"):
            M.membrane_linear = M.membrane_quadratic      # resolved at call time
        p = M.init(key, N_BUCKETS, TPH, HEADS, OBS, 2 * ACT)

        def fn(p, x):
            return M.apply(p, x, 0.3, HEADS, TPH, N_BUCKETS, mode="st").sum(1)
        return p, fn

    import jax_pure_lif as P
    p = P.init(key, NAP, TPH, HEADS, OBS, 2 * ACT)

    def fn(p, x):
        return P.apply(p, x, 0.3, HEADS, TPH, NAP, mode="st").sum(1)
    return p, fn


def measure(variant):
    import time
    import jax
    import jax.numpy as jnp

    dev = jax.local_devices()[0]
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    p, fn = build(variant, k1)
    n_par = sum(int(v.size) for v in jax.tree.leaves(p))
    x = jax.random.normal(k2, (BATCH, OBS))
    base = dev.memory_stats().get("bytes_in_use", 0)

    @jax.jit
    def step(p, x):
        return jax.value_and_grad(lambda pp: (fn(pp, x) ** 2).sum())(p)

    lowered = step.lower(p, x).compile()
    try:
        ca = lowered.cost_analysis()
        ca = ca[0] if isinstance(ca, list) else ca
        flops = float(ca.get("flops", 0))
        bytes_accessed = float(ca.get("bytes accessed", 0))
    except Exception:
        flops, bytes_accessed = 0.0, 0.0

    v, g = step(p, x)
    jax.block_until_ready(g)
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        v, g = step(p, x)
    jax.block_until_ready(g)
    ms = 1000 * (time.perf_counter() - t0) / reps

    peak = int(dev.memory_stats().get("peak_bytes_in_use", 0))
    out = dict(variant=variant, params=n_par, ms_per_step=round(ms, 3),
               peak_bytes=peak, activation_gb=round((peak - base) / 2**30, 4),
               gflops=round(flops / 1e9, 3),
               gb_accessed=round(bytes_accessed / 2**30, 3))
    # Value check: the two c32b variants must agree to fp32 noise.
    out["loss"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default=None)
    a = ap.parse_args()
    if a.variant:
        print("RESULT " + json.dumps(measure(a.variant)), flush=True)
        return

    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_deterministic_ops=true",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    res = []
    for v in VARIANTS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--variant", v],
                           capture_output=True, text=True, env=env, cwd=HERE)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
        if not line:
            print(f"  {v:<16} FAILED\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
            continue
        d = json.loads(line[0][len("RESULT "):])
        res.append(d)
        print(f"  {d['variant']:<16} {d['ms_per_step']:>8.2f} ms/step  "
              f"activations {d['activation_gb']:>7.4f} GB  "
              f"{d['gflops']:>8.2f} GFLOP  {d['gb_accessed']:>7.2f} GB moved  "
              f"params {d['params']:,}", flush=True)

    by = {d["variant"]: d for d in res}
    if "c32b-linear" in by and "c32b-quadratic" in by:
        L, Q = by["c32b-linear"], by["c32b-quadratic"]
        agree = abs(L["loss"] - Q["loss"]) / max(1.0, abs(Q["loss"]))
        print(f"\n  linear vs quadratic (same model, membrane only):")
        print(f"    value agreement   rel {agree:.2e}  "
              f"{'OK — the recurrence is exact' if agree < 1e-5 else 'MISMATCH'}")
        print(f"    wall clock        {Q['ms_per_step']/L['ms_per_step']:.2f}x faster")
        print(f"    activations       {Q['activation_gb']/max(L['activation_gb'],1e-9):.1f}x smaller "
              f"({Q['activation_gb']:.4f} -> {L['activation_gb']:.4f} GB)")
        if L["gflops"] and Q["gflops"]:
            print(f"    FLOPs             {Q['gflops']/L['gflops']:.2f}x fewer")
    if "c32b-linear" in by and "c31-pure" in by:
        L, P = by["c32b-linear"], by["c31-pure"]
        print(f"\n  vs exp_c31 PureLIF (the previous most expensive front-end):")
        print(f"    wall clock        {P['ms_per_step']/L['ms_per_step']:.2f}x faster")
        print(f"    activations       {P['activation_gb']/max(L['activation_gb'],1e-9):.1f}x smaller "
              f"({P['activation_gb']:.4f} -> {L['activation_gb']:.4f} GB)")
    json.dump(res, open(os.path.join(HERE, "bench_linear.json"), "w"), indent=1)
    print("\nwrote bench_linear.json")


if __name__ == "__main__":
    main()
