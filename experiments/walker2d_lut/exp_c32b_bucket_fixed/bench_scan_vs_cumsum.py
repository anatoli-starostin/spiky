"""exp_c32b — our associative_scan recurrence vs nucstar's cumsum, head to head.

Both compute the identical membrane; the question is only which is cheaper. Measured, not
reasoned about.

METHOD, and why each choice:

  * ONE SUBPROCESS PER (variant, N). `peak_bytes_in_use` is a high-water mark that never
    comes down, so two variants in one process report the max of the pair.
  * `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or every variant "uses" 75% of the card.
  * BEST-OF-N (minimum), not mean. The GPU may be shared with training runs; contention can
    only ever make a sample slower, so the minimum is the robust estimator of the
    uncontended cost. The median is printed alongside so the spread is visible.
  * Timed at the REAL work: the actor-loss `value_and_grad` in mode="st", batch 512, plus a
    forward-only number separately. A membrane microbenchmark would flatter whichever
    variant fuses better in isolation and tell us nothing about the trainer.
  * The two variants are asserted to AGREE before any timing is reported. A faster wrong
    answer is not a result.

N is swept because the asymptotics differ: associative_scan is O(log N) DEPTH but does ~2x
the arithmetic of a plain prefix sum, while cumsum is O(N) depth with minimal arithmetic.
At small N the scan's overhead can dominate its better depth, which is exactly the honest
picture worth having.

Usage:
  python bench_scan_vs_cumsum.py --all
  python bench_scan_vs_cumsum.py --variant scan --n 17
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ACT, TPH, HEADS, BATCH, N_BUCKETS = 6, 32, 1, 512, 16
NS = (17, 33, 64)
VARIANTS = ("scan", "cumsum")
REPS = 60


def measure(variant, n):
    import jax
    import jax.numpy as jnp
    import numpy as np
    sys.path.insert(0, HERE)
    import jax_bucket_lif as M

    if variant == "cumsum":
        M.membrane_linear = M.membrane_cumsum          # resolved at call time

    dev = jax.local_devices()[0]
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    p = M.init(k1, N_BUCKETS, TPH, HEADS, n, 2 * ACT)
    x = jax.random.normal(k2, (BATCH, n))
    jax.block_until_ready(x)
    base = dev.memory_stats().get("bytes_in_use", 0)

    def fwd(p, x):
        return M.apply(p, x, 0.3, HEADS, TPH, N_BUCKETS, mode="st").sum(1)

    f_fwd = jax.jit(fwd)
    f_bwd = jax.jit(lambda p, x: jax.value_and_grad(
        lambda pp: (fwd(pp, x) ** 2).sum())(p))

    out = {}
    for name, fn in (("fwd", f_fwd), ("fwdbwd", f_bwd)):
        t0 = time.perf_counter()
        c = fn.lower(p, x).compile()
        out[f"compile_{name}_s"] = round(time.perf_counter() - t0, 3)
        try:
            ca = c.cost_analysis()
            ca = ca[0] if isinstance(ca, list) else ca
            out[f"gflop_{name}"] = round(float(ca.get("flops", 0)) / 1e9, 4)
        except Exception:
            out[f"gflop_{name}"] = 0.0
        r = fn(p, x)
        jax.block_until_ready(r)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = fn(p, x)
            jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000.0
        out[f"ms_{name}_min"] = round(float(ts.min()), 4)
        out[f"ms_{name}_med"] = round(float(np.median(ts)), 4)

    peak = int(dev.memory_stats().get("peak_bytes_in_use", 0))
    out.update(variant=variant, n=n, peak_gb=round(peak / 2**30, 4),
               activation_gb=round((peak - base) / 2**30, 4))
    # Value check against the other formulation, in-process, same params.
    v_scan = M.membrane_linear if variant == "scan" else M.membrane_cumsum
    import jax_bucket_lif as M2
    a = jnp.sort(jax.random.uniform(k2, (64, TPH, n), maxval=32.0), axis=-1)
    w = 2.0 * jax.nn.sigmoid(jax.random.normal(k1, (64, TPH, n)))
    tau = jnp.full((1, TPH, 1), 2.313)
    d = float(jnp.abs(M2.membrane_linear(a, w, tau)
                      - M2.membrane_cumsum(a, w, tau)).max()
              / jnp.abs(M2.membrane_cumsum(a, w, tau)).max())
    out["scan_vs_cumsum_rel"] = float(f"{d:.3e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--n", type=int, default=17)
    a = ap.parse_args()
    if a.variant:
        print("RESULT " + json.dumps(measure(a.variant, a.n)), flush=True)
        return

    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_deterministic_ops=true",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    res = []
    for n in NS:
        for v in VARIANTS:
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--variant", v, "--n", str(n)],
                               capture_output=True, text=True, env=env, cwd=HERE)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
            if not line:
                print(f"  N={n:<3} {v:<7} FAILED\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
                continue
            res.append(json.loads(line[0][len("RESULT "):]))

    print(f"{'N':>4} {'variant':<8}{'fwd ms':>9}{'fwd+bwd ms':>12}{'act GB':>9}"
          f"{'GFLOP f+b':>11}{'compile s':>11}")
    for d in res:
        print(f"{d['n']:>4} {d['variant']:<8}{d['ms_fwd_min']:>9.3f}"
              f"{d['ms_fwdbwd_min']:>12.3f}{d['activation_gb']:>9.4f}"
              f"{d['gflop_fwdbwd']:>11.4f}{d['compile_fwdbwd_s']:>11.2f}")

    print("\n  (min of 60; the GPU may be shared, and contention can only add time)")
    by = {(d["n"], d["variant"]): d for d in res}
    print(f"\n{'N':>4}  {'fwd  scan/cumsum':>18}  {'fwd+bwd  scan/cumsum':>22}  "
          f"{'activations':>13}")
    for n in NS:
        s, c = by.get((n, "scan")), by.get((n, "cumsum"))
        if not (s and c):
            continue
        print(f"{n:>4}  {s['ms_fwd_min']/c['ms_fwd_min']:>17.2f}x  "
              f"{s['ms_fwdbwd_min']/c['ms_fwdbwd_min']:>21.2f}x  "
              f"{s['activation_gb']/max(c['activation_gb'],1e-9):>12.2f}x")
    if res:
        print(f"\n  membrane agreement scan vs cumsum: "
              f"{max(d['scan_vs_cumsum_rel'] for d in res):.2e} (must be ~1e-7)")
    json.dump(res, open(os.path.join(HERE, "bench_scan_vs_cumsum.json"), "w"), indent=1)
    print("\nwrote bench_scan_vs_cumsum.json")


if __name__ == "__main__":
    main()
