"""exp_c18 — WHAT does seed 4 actually do differently? (#75). MJX venv.

Seed 4 scores 5286.6 against a pack at 4112 +/- 160, reproducibly. The previous diagnostic
established that this is NOT explained by the init or by how the addressing moved. This one
stops looking at weights and looks at BEHAVIOUR, plus the two model-space questions that
can be asked without weight-space alignment.

Three blocks:

  1. GAIT. An instrumented re-run of the exact CPU-reference rollout -- same physics, same
     reward, same 100 episode seeds, same deterministic action rule -- recording per episode
     the length, return, forward velocity, torso height and pitch, action magnitude, and
     action chatter. Because it is the same rollout, the mean return must reproduce the
     committed CPU-reference number; that equality is the check that this instrumentation
     did not perturb anything, and it is asserted rather than assumed.

  2. GEOMETRY ON-POLICY. Row usage measured on the states the policy ACTUALLY VISITS, not
     on the distillation dataset. That distinction matters: a policy is only as good as its
     addressing on its own trajectory distribution.

  3. FUNCTION-SPACE DISTANCE. Weight-space distance between these six models is ILL-POSED
     and deliberately not reported: canonical_full_coverage gives each seed a different
     assignment of comparators to (table, bit) slots, so table 7 of seed 0 and table 7 of
     seed 4 are not the same object, and no alignment is available. What IS well-defined is
     the distance between the FUNCTIONS: evaluate all six on one common observation set and
     compare the actions. Classical MDS on that distance matrix then answers "is seed 4 far
     off on its own, or further along an axis the pack shares?"
"""
import json, os, sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C03 = os.path.join(D, "exp_c03_distillation")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac"):
    sys.path.insert(0, os.path.join(D, p))

import perturb                    # noqa: E402
import eval_cpu                   # noqa: E402  (reuse ITS loader, so the actor is
                                  #             built exactly as the committed eval does)
import jax_lut_grad as L          # noqa: E402

SEEDS = (0, 1, 2, 3, 4, 5)
STAR = 4
EPISODES, MAX_STEPS = 100, 1000
FRAME_SKIP = perturb.FRAME_SKIP
N_OBS_COMMON = 20000
OUT_JSON = os.path.join(HERE, "behavior_stats.json")
OUT_NPZ = os.path.join(HERE, "behavior_arrays.npz")


def rollout_instrumented(model, policy_fn, episodes=EPISODES, max_steps=MAX_STEPS,
                         seed0=0):
    """perturb.eval_batched, instrumented. The physics, reward, termination test, reset
    noise and episode seeds are copied line for line; only recording is added."""
    dt = model.opt.timestep * FRAME_SKIP
    datas, alive = [], np.ones(episodes, bool)
    rets = np.zeros(episodes)
    length = np.zeros(episodes, int)
    vel = [[] for _ in range(episodes)]
    z_l = [[] for _ in range(episodes)]
    ang_l = [[] for _ in range(episodes)]
    a2 = np.zeros(episodes)
    chatter = np.zeros(episodes)
    prev_act = np.zeros((episodes, model.nu))
    have_prev = np.zeros(episodes, bool)
    visited = []                       # observations actually visited (subsampled)

    for ep in range(episodes):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, model.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, model.nv)
        mujoco.mj_forward(model, d)
        datas.append(d)

    for step in range(1, max_steps + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        if step % 10 == 0:
            visited.append(obs.copy())
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            x0 = d.qpos[0]
            d.ctrl[:] = act[j]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, d)
            v = (d.qpos[0] - x0) / dt
            rets[i] += 1.0 + v - 1e-3 * float(act[j] @ act[j])
            length[i] = step
            vel[i].append(v)
            z_l[i].append(d.qpos[1])
            ang_l[i].append(d.qpos[2])
            a2[i] += float(act[j] @ act[j])
            if have_prev[i]:
                chatter[i] += float(np.abs(act[j] - prev_act[i]).mean())
            prev_act[i] = act[j]
            have_prev[i] = True
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False

    per_ep = dict(
        ret=rets, length=length.astype(float),
        fell=(length < max_steps).astype(float),
        vel_mean=np.array([np.mean(v) for v in vel]),
        vel_sd=np.array([np.std(v) for v in vel]),
        z_mean=np.array([np.mean(v) for v in z_l]),
        z_sd=np.array([np.std(v) for v in z_l]),
        ang_mean=np.array([np.mean(v) for v in ang_l]),
        ang_absmax=np.array([np.max(np.abs(v)) for v in ang_l]),
        act_energy=a2 / np.maximum(length, 1),
        chatter=chatter / np.maximum(length - 1, 1))
    return per_ep, np.concatenate(visited, 0) if visited else np.zeros((0, 17))


def rows_and_entropy(actor_npz, obs_raw, om, osd):
    """obs_raw is RAW observation; the model standardises internally, so this must too."""
    z = np.load(actor_npz)
    w, b = np.asarray(z["w"], np.float64), np.asarray(z["b"], np.float64)
    x = (np.asarray(obs_raw, np.float64) - om) / (osd + 1e-6)
    a = np.einsum("bd,tnd->btn", x, w) + b[None]
    bit = a > 0
    nap = w.shape[1]
    K = 2 ** nap
    powers = (2 ** np.arange(nap - 1, -1, -1)).astype(np.int64)
    rows = (bit.astype(np.int64) * powers[None, None, :]).sum(-1)
    T = w.shape[0]
    used, ent, top1 = np.zeros(T, int), np.zeros(T), np.zeros(T)
    for t in range(T):
        c = np.bincount(rows[:, t], minlength=K).astype(np.float64)
        p = c / c.sum()
        nz = p[p > 0]
        used[t] = len(nz)
        ent[t] = -(nz * np.log2(nz)).sum()
        top1[t] = nz.max()
    return dict(rows_used_mean=float(used.mean()), rows_used_min=int(used.min()),
                entropy_bits=float(ent.mean()), top_row_share=float(top1.mean()),
                nap=int(nap))


def main():
    m = perturb.make_model(None, 1.0)
    committed = {s: json.load(open(os.path.join(
        C09, f"lut_sac_c18_seed{s}_cpueval.json")))["cpu_reference_mean"]
        for s in SEEDS}

    stats, arrays, visited = {}, {}, {}
    fns = {}
    for s in SEEDS:
        fn, _n = eval_cpu.load_actor(
            os.path.join(C09, f"lut_sac_c18_seed{s}_actor.npz"), forward_mode="hard")
        fns[s] = fn
        print(f"rolling out seed {s} ...", flush=True)
        per_ep, vis = rollout_instrumented(m, fn)
        got, want = float(per_ep["ret"].mean()), committed[s]
        assert abs(got - want) < 0.5, (
            f"seed {s}: instrumented rollout {got:.1f} != committed CPU-reference "
            f"{want:.1f} — the instrumentation changed the rollout, numbers not usable")
        print(f"  return {got:.1f} (matches committed {want:.1f})", flush=True)
        stats[s] = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                            median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                            p90=float(np.percentile(v, 90)))
                    for k, v in per_ep.items()}
        stats[s]["n_full_horizon"] = int((per_ep["length"] >= MAX_STEPS).sum())
        stats[s]["n_fell"] = int(per_ep["fell"].sum())
        stats[s]["score"] = want
        arrays[f"ret_{s}"] = per_ep["ret"]
        arrays[f"len_{s}"] = per_ep["length"]
        arrays[f"vel_{s}"] = per_ep["vel_mean"]
        visited[s] = vis

    # ---- 1. gait table ------------------------------------------------------
    print("\n=== 1. GAIT — 100 deterministic episodes each ===")
    print(f"{'seed':>5}{'score':>8}{'ep len':>9}{'full 1000':>11}{'fell':>6}"
          f"{'fwd vel':>9}{'vel sd':>8}{'z mean':>8}{'z sd':>7}{'|ang|max':>10}"
          f"{'energy':>8}{'chatter':>9}")
    for s in SEEDS:
        t = stats[s]
        mark = " <--" if s == STAR else ""
        print(f"{s:>5}{t['score']:>8.0f}{t['length']['mean']:>9.0f}"
              f"{t['n_full_horizon']:>8}/100{t['n_fell']:>6}"
              f"{t['vel_mean']['mean']:>9.3f}{t['vel_mean']['sd']:>8.3f}"
              f"{t['z_mean']['mean']:>8.3f}{t['z_sd']['mean']:>7.3f}"
              f"{t['ang_absmax']['mean']:>10.3f}{t['act_energy']['mean']:>8.3f}"
              f"{t['chatter']['mean']:>9.4f}{mark}")
    print("\n  per-episode RETURN distribution:")
    print(f"{'seed':>5}{'mean':>9}{'sd':>8}{'p10':>9}{'median':>9}{'p90':>9}"
          f"{'min':>9}{'max':>9}")
    for s in SEEDS:
        r = arrays[f"ret_{s}"]
        print(f"{s:>5}{r.mean():>9.1f}{r.std(ddof=1):>8.1f}"
              f"{np.percentile(r, 10):>9.1f}{np.median(r):>9.1f}"
              f"{np.percentile(r, 90):>9.1f}{r.min():>9.1f}{r.max():>9.1f}")

    # ---- 2. on-policy geometry ---------------------------------------------
    print("\n=== 2. GEOMETRY on the states each policy ACTUALLY VISITS ===")
    print(f"{'seed':>5}{'score':>8}{'rows used':>11}{'min':>6}{'entropy':>10}"
          f"{'top-row share':>15}{'visited states':>16}")
    st0 = json.load(open(os.path.join(C03, "dataset_stats.json")))
    om = np.asarray(st0["obs_mean"], np.float64)
    osd = np.asarray(st0["obs_std"], np.float64)
    geo = {}
    for s in SEEDS:
        g = rows_and_entropy(os.path.join(C09, f"lut_sac_c18_seed{s}_actor.npz"),
                             visited[s], om, osd)
        geo[s] = g
        print(f"{s:>5}{stats[s]['score']:>8.0f}{g['rows_used_mean']:>11.1f}"
              f"{g['rows_used_min']:>6}{g['entropy_bits']:>9.2f}b"
              f"{100*g['top_row_share']:>14.1f}%{len(visited[s]):>16,}")
    print(f"  (entropy out of {geo[0]['nap']} bits; 'top-row share' = fraction of "
          f"visits landing in a table's single most-used row)")

    # ---- 3. function-space distance ----------------------------------------
    o = np.load(os.path.join(C03, "obs.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(o), N_OBS_COMMON, replace=False))
    xc = np.asarray(o[idx], np.float32)
    acts = {s: np.asarray(fns[s](xc), np.float64) for s in SEEDS}
    Dm = np.zeros((len(SEEDS), len(SEEDS)))
    for i, a_ in enumerate(SEEDS):
        for j, b_ in enumerate(SEEDS):
            Dm[i, j] = np.sqrt(np.mean((acts[a_] - acts[b_]) ** 2))
    print("\n=== 3. FUNCTION-SPACE DISTANCE (RMS action difference on 20k common obs) ===")
    print("  weight-space distance is NOT reported: each seed assigns comparators to "
          "(table, bit) slots differently, so the tables are not corresponding objects.")
    print("       " + "".join(f"{'s'+str(s):>8}" for s in SEEDS))
    for i, s in enumerate(SEEDS):
        print(f"    s{s} " + "".join(f"{Dm[i, j]:>8.3f}" for j in range(len(SEEDS))))
    pack = [i for i, s in enumerate(SEEDS) if s != STAR]
    star_i = SEEDS.index(STAR)
    d_pack = Dm[np.ix_(pack, pack)][~np.eye(len(pack), dtype=bool)]
    d_star = Dm[star_i, pack]
    print(f"\n  pack-to-pack mean {d_pack.mean():.3f} (sd {d_pack.std(ddof=1):.3f}, "
          f"range {d_pack.min():.3f}-{d_pack.max():.3f})")
    print(f"  seed4-to-pack mean {d_star.mean():.3f} (range {d_star.min():.3f}-"
          f"{d_star.max():.3f})")
    print(f"  ratio {d_star.mean() / d_pack.mean():.3f}x  — "
          + ("seed 4 is an outlier in function space"
             if d_star.mean() > 1.25 * d_pack.mean() else
             "seed 4 is NO further from the pack than the pack members are from each "
             "other; all six are mutually far apart"))

    # classical MDS on the distance matrix: is there a shared axis at all?
    n = len(SEEDS)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (Dm ** 2) @ J
    ev, evec = np.linalg.eigh(Bm)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    coords = evec[:, :2] * np.sqrt(np.maximum(ev[:2], 0))
    frac = np.maximum(ev, 0) / np.maximum(ev, 0).sum()
    print(f"\n  classical MDS: first two axes explain {100*frac[0]:.0f}% and "
          f"{100*frac[1]:.0f}% of the spread")
    for i, s in enumerate(SEEDS):
        print(f"    s{s}  ({coords[i, 0]:+.3f}, {coords[i, 1]:+.3f})"
              + ("   <-- outlier" if s == STAR else ""))

    json.dump(dict(gait=stats, geometry={str(s): geo[s] for s in SEEDS},
                   distance_matrix=Dm.tolist(), seeds=list(SEEDS),
                   pack_to_pack_mean=float(d_pack.mean()),
                   star_to_pack_mean=float(d_star.mean()),
                   mds_coords=coords.tolist(), mds_frac=frac[:3].tolist(),
                   episodes=EPISODES, max_steps=MAX_STEPS),
              open(OUT_JSON, "w"), indent=1)
    np.savez_compressed(OUT_NPZ, D=Dm, **arrays)
    print(f"\nwrote {OUT_JSON} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
