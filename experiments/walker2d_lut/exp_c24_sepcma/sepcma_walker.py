"""exp_c24 step 3 — real sep-CMA-ES on the MJX Walker2d env (#75).

Pure evolution, no gradients. The optimiser is `sep_cma.py` (Ros & Hansen 2008), verified
on sphere / ellipsoid / Rosenbrock by `test_sepcma.py` before it was pointed at anything
expensive. The policy is MLP[32,32] (d=1,830) -- exp_c05's exact geometry, so the result
is directly comparable to the 100-episode re-scores in `rescore_c05_100ep.json`.

WHAT THIS RUN CHANGES RELATIVE TO exp_c05, and why:

  * **A real CMA-ES.** exp_c05's `--algo sepcma` was OpenAI-ES with Adam and a per-
    coordinate step-size heuristic (see sep_cma.py's docstring). This is the actual
    algorithm: weighted recombination, evolution paths, CSA, diagonal covariance.
  * **40x the budget.** 300 gens x pop 1024 x 2 eps x horizon 1000 = 614M env-steps
    against exp_c05's 15.36M. exp_c05's fitness curve was still climbing at its last
    generation (+94.6 over the final 25), so budget was the binding constraint.
  * **Horizon 1000, not 400.** exp_c05 optimised survival to step 400 and was then scored
    over 1000. Much of its +/-890 spread is policies that were never asked to walk that
    far.
  * **A large population, deliberately.** CMA-ES's textbook lambda = 4+3ln(d) is 26 here.
    On this harness that would waste the GPU almost entirely: an MJX rollout is
    latency-bound on the sequential physics scan, so measured, 4x the envs costs only
    ~1.5x the wall-clock. A 26-member generation costs nearly what a 1024-member one
    does. Large lambda also damps fitness noise, which CMA-ES is more sensitive to than
    rank-shaped ES because its covariance is learned from which candidates won.

The rollout harness is exp_c23's, which is ~3x cheaper than exp_c05's loop for two
reasons kept here: the mean-policy evaluation rides inside the population rollout as a
zero-perturbation member instead of a second scan, and the rollout exits once every env
is done (numerically identical -- those steps are multiplied by alive=0).

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python sepcma_walker.py --gens 300 --pop 1024
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
for p in ("exp_c02_mjx_scaffold", "exp_c04_jax_lut", "exp_c05_es"):
    sys.path.insert(0, os.path.join(BASE, p))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx            # noqa: E402
import es_mjx                     # noqa: E402
import sep_cma                    # noqa: E402


def make_fitness(mx, apply, norm, episodes, horizon, eval_episodes):
    """Population fitness + unperturbed-mean fitness from ONE rollout.

    Index `P` of the candidate batch is the distribution mean, appended by the caller.
    """
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)

    def fitness(cand, key):
        P = cand.shape[0] - 1                      # last row is the mean
        n = P * episodes + eval_episodes
        st = v_reset(jax.random.split(key, n))
        who = jnp.concatenate([jnp.repeat(jnp.arange(P), episodes),
                               jnp.full((eval_episodes,), P)])
        # Materialise per-env parameters ONCE, outside the scan (exp_c05 found that
        # gathering inside re-ran every physics step and dominated generation time).
        per_env = cand[who]

        def one(carry):
            st, ret, alive, t = carry
            act = jax.vmap(lambda p, o: apply(p, o, norm))(per_env, st.obs)
            nst = v_step(st, act)
            ret = ret + nst.reward * alive
            alive = alive * (1.0 - nst.done)
            return (nst, ret, alive, t + 1)

        def cond(carry):
            _, _, alive, t = carry
            return (t < horizon) & (alive.sum() > 0)

        st, ret, alive, steps = jax.lax.while_loop(
            cond, one, (st, jnp.zeros(n), jnp.ones(n), 0))
        pop_ret, mean_ret = ret[:P * episodes], ret[P * episodes:]
        return pop_ret.reshape(P, episodes).mean(1), mean_ret.mean(), steps

    return jax.jit(fitness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=1024, help="lambda")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--sigma0", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    st = json.load(open(os.path.join(BASE, "exp_c03_distillation", "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))

    # exp_c05's own spec, imported rather than re-implemented, so the geometry and the
    # forward pass are identical to the runs this is compared against.
    d, apply, init_fn = es_mjx.mlp_spec(a.hidden)

    key = jax.random.PRNGKey(a.seed)
    key, ki = jax.random.split(key)
    x0 = init_fn(ki)

    state = sep_cma.init(d, lam=a.pop, sigma0=a.sigma0, x0=x0)
    mx = mjx.put_model(W.make_model())
    fitness = make_fitness(mx, apply, norm, a.episodes, a.horizon, a.eval_episodes)

    name = f"sepcma_mlp{a.hidden}_s{a.seed}{a.tag}"
    print(f"[{name}] sep-CMA-ES (Ros & Hansen 2008) | MLP[{a.hidden},{a.hidden}] "
          f"d={d:,} | lambda={state['lam']} mu={state['mu']} mu_eff={state['mu_eff']:.1f} "
          f"| textbook lambda would be {sep_cma.default_lambda(d)} "
          f"| {a.pop*a.episodes*a.horizon:,} env-steps/gen x {a.gens} gens", flush=True)
    print(f"[{name}] c_sigma={state['c_sigma']:.4f} d_sigma={state['d_sigma']:.4f} "
          f"c_c={state['c_c']:.4f} c_1={state['c_1']:.2e} c_mu={state['c_mu']:.2e}",
          flush=True)

    best_mean, rows, steps_done = -1e9, [], 0
    t0 = time.time()
    for g in range(a.gens):
        key, k1, k2 = jax.random.split(key, 3)
        x, _z = sep_cma.ask(state, k1)
        cand = jnp.concatenate([x, state["m"][None, :]])     # mean rides along
        f, fm, roll = fitness(cand, k2)
        state = sep_cma.tell(state, x, f, maximise=True)

        fm = float(fm)
        best_mean = max(best_mean, fm)
        steps_done += int(roll) * (a.pop * a.episodes + a.eval_episodes)
        rows.append(dict(gen=g, mean_policy=fm, pop_best=float(f.max()),
                         pop_mean=float(f.mean()), sigma=float(state["sigma"]),
                         C_min=float(state["C"].min()), C_max=float(state["C"].max()),
                         rollout_steps=int(roll), env_steps=steps_done,
                         elapsed_s=round(time.time() - t0, 1)))
        if g % 10 == 0 or g == a.gens - 1:
            print(f"  gen {g:>4}/{a.gens}  mean-policy {fm:8.1f}  pop best "
                  f"{float(f.max()):8.1f}  pop mean {float(f.mean()):8.1f}  "
                  f"(best mean {best_mean:8.1f})  sigma {state['sigma']:.4f}  "
                  f"C {float(state['C'].min()):.1e}-{float(state['C'].max()):.1e}  "
                  f"roll {int(roll):>4}  {time.time()-t0:6.0f}s", flush=True)

        # Save every generation: a 2.5h run should not lose everything to a late crash.
        np.save(os.path.join(HERE, f"{name}_mu.npy"), np.asarray(state["m"]))

    out = dict(name=name, algo="sep-CMA-ES (Ros & Hansen 2008)", d=d, hidden=a.hidden,
               seed=a.seed, gens=a.gens, lam=state["lam"], mu=state["mu"],
               mu_eff=state["mu_eff"], episodes=a.episodes, horizon=a.horizon,
               sigma0=a.sigma0, final_sigma=float(state["sigma"]),
               total_env_steps=steps_done,
               best_mean_policy_mjx=best_mean,
               final_mean_policy_mjx=rows[-1]["mean_policy"],
               wall_s=rows[-1]["elapsed_s"], history=rows)
    json.dump(out, open(os.path.join(HERE, f"{name}.json"), "w"), indent=1)
    print(f"[{name}] best mean-policy MJX fitness {best_mean:.1f} over "
          f"{steps_done:,} env-steps in {rows[-1]['elapsed_s']/60:.1f} min "
          f"(MJX PROXY -- the headline number comes from the 100-episode CPU eval)",
          flush=True)


if __name__ == "__main__":
    main()
