"""exp012: steady-state evolution of a 33-neuron spnet with NO plasticity at all.

The loop is deliberately the same shape as the rest of the chapter (steady_state.py,
lut_evolve.py) so the results are comparable: a pool of K, every member re-scored each round
on the SAME fresh batch (so members stay paired within a round), an EWMA over rounds so a
member is not culled for one unlucky batch, the worst M culled once past a grace period, and
refill by tournament-of-2 + mutate.

WHAT IS EVOLVED. Everything, directly: the [27,16] presence mask (topology), the [27,16]
per-synapse delay, and the [27,16] per-synapse weight. No STDP settling step -- the weights
the genome ships are the weights that run. See tiny_snn.py for the substrate and for why
Dale's law is structural rather than policed.

FITNESS = -MSE on exp009's quantised centred target (offsets 0..31, silence reads 32). The
two reference points: the best CONSTANT predictor, recomputed per split, and exp009's 37.52,
which an 800-excitatory STDP reservoir reached on the same target after 300 rounds.

    python tiny_evolve.py --seed 0 --rounds 400 --pool 64 --tag run0 --out-dir …
"""
import argparse
import json
import os
import time

import numpy as np

import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder


jsonable = T.jsonable


def genome_to_json(g):
    return dict(mask=g["mask"].astype(np.int8).tolist(),
                delay=g["delay"].astype(int).tolist(),
                weight=np.round(g["weight"], 6).tolist())


def tournament(rng, fit, k=2):
    i = rng.integers(0, len(fit), k)
    return int(i[np.argmax(fit[i])])


def save_ckpt(path, pool, ewma, age, rnd, hist, best, rng):
    """Resumable state. Written atomically, because a run that dies mid-write loses the lot.

    The pool is 64 x 3 x [27,16] -- about 400 KB -- so this is cheap enough to do often.
    """
    np.savez_compressed(
        path + ".tmp.npz",
        mask=np.stack([g["mask"] for g in pool]).astype(np.uint8),
        delay=np.stack([g["delay"] for g in pool]).astype(np.int16),
        weight=np.stack([g["weight"] for g in pool]).astype(np.float32),
        ewma=ewma, age=age, rnd=np.int64(rnd),
        hist=np.frombuffer(json.dumps(T.jsonable(hist)).encode(), np.uint8),
        best=np.frombuffer(json.dumps(T.jsonable(best)).encode(), np.uint8),
        rng=np.frombuffer(json.dumps(rng.bit_generator.state).encode(), np.uint8))
    os.replace(path + ".tmp.npz", path)


def load_ckpt(path):
    Z = np.load(path, allow_pickle=False)
    pool = [dict(mask=Z["mask"][i].astype(bool), delay=Z["delay"][i].astype(np.int64),
                 weight=Z["weight"][i].astype(np.float64)) for i in range(len(Z["mask"]))]
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(Z["rng"].tobytes().decode())
    return (pool, Z["ewma"], Z["age"], int(Z["rnd"]),
            json.loads(Z["hist"].tobytes().decode()),
            json.loads(Z["best"].tobytes().decode()), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--cull", type=int, default=8, help="members replaced each round")
    ap.add_argument("--grace", type=int, default=3, help="rounds before a member may be culled")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ewma", type=float, default=0.5)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--w-ceiling", type=float, default=200.0)
    ap.add_argument("--p-init", type=float, default=0.5)
    ap.add_argument("--p-add", type=float, default=0.02)
    ap.add_argument("--p-prune", type=float, default=0.02)
    ap.add_argument("--p-delay", type=float, default=0.08)
    ap.add_argument("--p-weight", type=float, default=0.25)
    ap.add_argument("--w-sigma", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--crossover", action="store_true",
                    help="sexual: each offspring gets TWO tournament parents, uniform per-cell "
                         "crossover, then the same mutate(). Off = today's asexual path, "
                         "byte for byte.")
    ap.add_argument("--ckpt-every", type=int, default=50,
                    help="rounds between resumable checkpoints (0 disables)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from ck_<tag>.npz in --out-dir if it exists")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    ck = os.path.join(a.out_dir, f"ck_{a.tag}.npz")
    beat = os.path.join(a.out_dir, f"{a.tag}.progress")

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    base_v = T.constant_baseline(Yv)
    tgt_v = T.target_offsets(Yv)

    start = 0
    if a.resume and os.path.exists(ck):
        pool, ewma, age, start, hist, best, rng = load_ckpt(ck)
        start += 1
        print(f"resumed {ck} at round {start}", flush=True)
    else:
        pool = [T.random_genome(rng, a.w_max, a.p_init) for _ in range(a.pool)]
        ewma = np.full(a.pool, np.nan)
        age = np.zeros(a.pool, np.int64)
        hist, best = [], dict(mse=np.inf)

    n_sex = 0                      # offspring that actually got two distinct parents
    for rnd in range(start, a.rounds):
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, rnd)
        H = T.build(pool, device=a.device, seed=1, w_ceiling=a.w_ceiling)
        s = T.score(H, Xb, Yb, enc)
        mse = s["mse"]
        ewma = np.where(np.isnan(ewma), mse, a.ewma * mse + (1 - a.ewma) * ewma)
        age += 1

        i = int(np.argmin(mse))
        if mse[i] < best["mse"]:
            Hb = T.build([pool[i]], device=a.device, w_ceiling=a.w_ceiling)
            sv = T.score(Hb, Xv, Yv, enc)
            r_b, ceil_b = T.affine_ceiling_and_r(sv["first"][:, 0, :], tgt_v)
            best = dict(mse=float(mse[i]), rnd=rnd, heldout_mse=float(sv["mse"][0]),
                        heldout_tau=float(sv["tau"][0]),
                        heldout_mse_action=float(sv["mse_action"][0]),
                        heldout_silent=float(sv["silent"][0]),
                        heldout_mean_abs_r=r_b, heldout_affine_ceiling=ceil_b,
                        stats=T.genome_stats(pool[i]), genome=genome_to_json(pool[i]))

        # mean|r| and the affine ceiling for THIS round's best, on the batch. Reporting only --
        # neither is in the fitness, which stays the raw offset MSE.
        r_r, ceil_r = T.affine_ceiling_and_r(s["first"][:, i, :], T.target_offsets(Yb))
        rec = dict(rnd=rnd, mse_min=float(mse.min()), mse_mean=float(mse.mean()),
                   mse_std=float(mse.std()), tau_max=float(s["tau"].max()),
                   tau_of_best=float(s["tau"][i]),
                   mean_abs_r=r_r, affine_ceiling=ceil_r,
                   silent=float(s["silent"].mean()), silent_min=float(s["silent"].min()),
                   silent_of_best=float(s["silent"][i]),
                   n_syn_mean=float(np.mean([g["mask"].sum() for g in pool])),
                   best_heldout=best.get("heldout_mse"))
        if a.eval_every and rnd % a.eval_every == 0:
            Hb = T.build([pool[int(np.argmin(ewma))]], device=a.device, w_ceiling=a.w_ceiling)
            sv = T.score(Hb, Xv, Yv, enc)
            r_l, ceil_l = T.affine_ceiling_and_r(sv["first"][:, 0, :], tgt_v)
            rec["ewma_leader_heldout_mse"] = float(sv["mse"][0])
            rec["ewma_leader_heldout_tau"] = float(sv["tau"][0])
            rec["ewma_leader_heldout_mean_abs_r"] = r_l
            rec["ewma_leader_heldout_affine_ceiling"] = ceil_l
            rec["ewma_leader_heldout_silent"] = float(sv["silent"][0])
        hist.append(rec)

        # ---- cull the worst, refill by tournament + mutate
        elig = np.where(age > a.grace)[0]
        if len(elig) > a.cull:
            worst = elig[np.argsort(-ewma[elig])[:a.cull]]
            for j in worst:
                p = tournament(rng, -ewma)
                if a.crossover:
                    q = tournament(rng, -ewma)
                    # two independent draws can land on the same member; recombining a genome
                    # with itself is the identity, so fall through to plain mutation rather
                    # than spend a coin flip per cell to reproduce the parent exactly
                    kid = pool[p] if q == p else T.crossover(pool[p], pool[q], rng)
                    n_sex += (q != p)
                else:
                    kid = pool[p]
                pool[j] = T.mutate(kid, rng, a.w_max, a.p_add, a.p_prune, a.p_delay,
                                   a.p_weight, a.w_sigma, a.w_ceiling)
                ewma[j], age[j] = np.nan, 0

        # HEARTBEAT, not a Slack call. Each seed writes one line; the driver owns the single
        # bar and sums these. Seeds never touch progress.py, so eight of them cannot race on
        # one handle or mint bars of their own.
        with open(beat + ".tmp", "w") as f:
            f.write(f"{rnd + 1} {a.rounds} {best.get('heldout_mse', float('nan')):.4f}\n")
        os.replace(beat + ".tmp", beat)

        if a.ckpt_every and (rnd % a.ckpt_every == 0 or rnd == a.rounds - 1):
            save_ckpt(ck, pool, ewma, age, rnd, hist, best, rng)

        if rnd % 25 == 0 or rnd == a.rounds - 1:
            print(f"r{rnd:4d}  pool best {mse.min():7.3f}  mean {mse.mean():7.3f}  "
                  f"held-out best {best.get('heldout_mse', float('nan')):7.3f}  "
                  f"tau {best.get('heldout_tau', float('nan')):+.4f}  "
                  f"[{time.time() - t0:6.1f}s]", flush=True)

    out = dict(config=vars(a), constant_baseline_val=base_v,
               exp009_reference=dict(constant=39.19, best_stdp_800exc=37.52),
               best=best, n_recombined_offspring=int(n_sex), elapsed_s=time.time() - t0)
    with open(os.path.join(a.out_dir, f"{a.tag}_final.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as f:
        json.dump(jsonable(hist), f)
    print(f"DONE seed {a.seed}: held-out MSE {best['heldout_mse']:.3f} "
          f"vs constant {base_v:.3f}, tau {best['heldout_tau']:+.4f} "
          f"({out['elapsed_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
