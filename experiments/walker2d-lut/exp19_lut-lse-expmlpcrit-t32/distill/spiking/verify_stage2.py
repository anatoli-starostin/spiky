"""Stage-two smoke test: the four checks that must pass before the long run.

(a) STDP moves excitatory weights while inhibitory stay exactly RES_W_INH -- keyed on
    (src,tgt), NEVER elementwise: export order is not stable between calls, so an
    elementwise diff silently compares unrelated synapses (it previously reported weights
    "moving" in a do_train=False control, which is impossible).
(b) Lamarckian inheritance: a newborn's birth excitatory weights equal its parent's CURRENT
    matured weights. Two links have to hold -- readback must pull device weights into the
    parent genome, and clone+mutate_structural must not disturb them.
(c) Newborn maturation does not stall selection: enough matured members remain to cull M
    every round.
(d) Excitatory synapse count stays bounded across rounds -- STDP depresses useless weights
    so remove-weak fires, and nothing ratchets upward.
Plus liveness, Dale's law, and no self-loops / duplicate (src,tgt).
"""
import argparse
import time

import numpy as np
import torch

import steady_state as S


def export_keyed(h):
    """-> {(src,tgt): (weight, meta)} straight off the device."""
    sp, ids, dev = h["spnet"], h["ids"], h["device"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=dev)
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device=dev) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    s, m, w, _, t = (x.cpu().numpy() for x in b)
    return {(int(a_), int(b_)): (float(c_), int(d_)) for a_, b_, c_, d_ in
            zip(s, t, w, m)}, len(s)


def genome_global_keys(h, genomes):
    """Per genome, the global (src,tgt) id of every synapse, in genome array order."""
    ids = h["ids"]
    base = {S.EXC: S.N_EXC, S.INH: S.N_INH, S.INP: S.N_IN, S.OUTP: S.N_OUT}
    out = []
    for c, g in enumerate(genomes):
        gs = np.empty(g["weight"].size, np.int64)
        gt = np.empty_like(gs)
        for p in (S.EXC, S.INH, S.INP):
            m = g["src_pool"] == p
            if m.any():
                gs[m] = ids[p][c * base[p] + g["src_idx"][m]]
        for p in (S.EXC, S.INH, S.OUTP):
            m = g["tgt_pool"] == p
            if m.any():
                gt[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
        out.append((gs, gt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--mature-batches", type=int, default=8)
    ap.add_argument("--stdp-lr", type=float, default=0.1)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--cull", type=float, default=0.25)
    ap.add_argument("--grace", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    X, Y, Xpool, Ypool, Xval, Yval = S.load(a.batch, a.seed, 512)
    enc = S.LatencyEncoder(Xpool)
    M = max(1, int(a.cull * a.pool))
    genomes = [S.seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max)
               for i in range(a.pool)]
    ewma = np.full(a.pool, np.nan)
    age = np.zeros(a.pool, int)
    newborn = np.zeros(a.pool, bool)

    print(f"stage-two smoke: K={a.pool} M={M}/round rounds={a.rounds} "
          f"stdp_lr={a.stdp_lr} mature={a.mature_batches} batches  dev={dev}")
    print(f"group sizes: metas={S.stage2_metas(a.stdp_lr, a.w_max)[0]._forward_group_size} "
          f"engine={S.GROUP_SIZE}  (must match, and must be 2)")

    res = dict(a_dw_exc=[], a_moved=[], a_inh_ok=[], b_readback=[], b_inherit=[],
               c_eligible=[], c_culled=[], d_n_exc=[], live=[], dale=[], loops=[],
               dups=[], clipped=[], corr_batch=[], corr_ewma=[])
    ceiling = 1.5 * a.w_max          # the meta's max_weight; railing here == saturation
    t0 = time.time()

    for rnd in range(a.rounds):
        Xb, Yb, _ = S.sample_batch(Xpool, Ypool, a.batch, a.seed, rnd)
        h = S.build_pool(genomes, dev, seed=1, stdp_lr=a.stdp_lr, w_max=a.w_max)

        # ---------- (a) does STDP move the right weights?
        before, n_syn = export_keyed(h)
        if newborn.any():
            S.stdp_batches(h, Xpool, Ypool, enc, a.batch, a.seed, rnd,
                           a.mature_batches, a.current)
            newborn[:] = False
        else:
            S.stdp_batches(h, Xpool, Ypool, enc, a.batch, a.seed, rnd, 2, a.current)
        after, _ = export_keyed(h)
        common = [k for k in before if k in after]
        exc = [k for k in common if before[k][1] < S.N_DELAY_METAS]
        inh = [k for k in common if before[k][1] >= S.N_DELAY_METAS]
        res["a_dw_exc"].append(max((abs(after[k][0] - before[k][0]) for k in exc),
                                   default=0.0))
        res["a_moved"].append(sum(1 for k in exc if after[k][0] != before[k][0]))
        res["a_inh_ok"].append(all(after[k][0] == S.RES_W_INH for k in inh))
        # what fraction of excitatory synapses is pinned AT the clip ceiling
        res["clipped"].append(
            sum(1 for k in exc if after[k][0] >= ceiling - 1e-4) / max(len(exc), 1))

        f, _ = S.score(h, Xb, Yb, enc, a.current)
        res["corr_batch"].append(float(f.mean()))

        # liveness: fraction of (sample, member, output) that actually spiked
        first, _ = S.run_episode(h, Xb[:64], enc, a.current)
        res["live"].append(float((first < S.N_TICKS).mean()))

        # ---------- (b1) readback fidelity: genome must now equal the DEVICE
        S.readback(h, genomes)
        gk = genome_global_keys(h, genomes)
        worst_rb = 0.0
        for c, g in enumerate(genomes):
            gs, gt = gk[c]
            e = np.nonzero(g["src_pool"] != S.INH)[0]
            for i in e:
                v = after.get((int(gs[i]), int(gt[i])))
                if v is not None:
                    worst_rb = max(worst_rb, abs(v[0] - float(g["weight"][i])))
        res["b_readback"].append(worst_rb)
        del h

        ewma = np.where(np.isnan(ewma), f, (1 - a.alpha) * ewma + a.alpha * f)
        res["corr_ewma"].append(float(np.nanmax(ewma)))
        age += 1

        # ---------- (c) selection, and (b2) inheritance of matured weights
        eligible = np.nonzero(age > a.grace)[0]
        res["c_eligible"].append(int(eligible.size))
        culled = 0
        worst_inh = 0.0
        if eligible.size >= M:
            worst = eligible[np.argsort(ewma[eligible])[:M]]
            surv = np.setdiff1d(np.arange(a.pool), worst)
            for slot in worst:
                c1, c2 = rng.choice(surv, 2, replace=False)
                par = c1 if ewma[c1] >= ewma[c2] else c2
                parent = genomes[par]
                child = S.mutate_structural(S.clone(parent), rng, a.w_max)
                # keyed on the LOCAL genome key: every synapse the child kept must carry
                # the parent's CURRENT (post-readback, matured) weight, bit for bit.
                pk, ck = S._key(parent), S._key(child)
                _, ip, ic = np.intersect1d(pk, ck, return_indices=True)
                pe = parent["src_pool"][ip] != S.INH
                if pe.any():
                    worst_inh = max(worst_inh, float(np.abs(
                        parent["weight"][ip][pe] - child["weight"][ic][pe]).max()))
                genomes[slot] = child
                ewma[slot] = ewma[par]
                age[slot] = 0
                newborn[slot] = True
                culled += 1
        res["c_culled"].append(culled)
        res["b_inherit"].append(worst_inh)

        # ---------- (d) and the invariants
        n_exc = int(sum((g["src_pool"] != S.INH).sum() for g in genomes))
        res["d_n_exc"].append(n_exc)
        dale = all(bool((g["weight"][g["src_pool"] == S.INH] == S.RES_W_INH).all()
                        and (g["weight"][g["src_pool"] != S.INH] >= 0).all())
                   for g in genomes)
        loops = sum(int(((g["src_pool"] == g["tgt_pool"]) &
                         (g["src_idx"] == g["tgt_idx"])).sum()) for g in genomes)
        dups = sum(int(g["weight"].size - np.unique(S._key(g)).size) for g in genomes)
        res["dale"].append(dale)
        res["loops"].append(loops)
        res["dups"].append(dups)

        print(f"  round {rnd:2d}  max|dw| {res['a_dw_exc'][-1]:7.4f}/{ceiling:.0f} "
              f"clipped {res['clipped'][-1]:6.2%}  live {res['live'][-1]:.3f}  "
              f"CORR batch {res['corr_batch'][-1]:+.4f} EWMA {res['corr_ewma'][-1]:+.4f}  "
              f"inh_ok {res['a_inh_ok'][-1]}  rb {worst_rb:.1e} inh {worst_inh:.1e}  "
              f"culled {culled}  n_exc {n_exc:,}  {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------- verdicts
    trained = [i for i, m in enumerate(res["a_moved"]) if m > 0]
    a_ok = bool(trained and all(res["a_inh_ok"]) and
                max(res["a_dw_exc"]) > 1e-4)
    b_ok = bool(max(res["b_readback"]) < 1e-5 and max(res["b_inherit"]) == 0.0)
    post = res["c_culled"][a.grace + 1:]
    c_ok = bool(post and all(c == M for c in post))
    n0, nl = res["d_n_exc"][0], res["d_n_exc"][-1]
    growth = (nl - n0) / max(n0, 1)
    d_ok = bool(abs(growth) < 0.25 and max(res["d_n_exc"]) < 1.5 * n0)
    inv_ok = bool(all(res["dale"]) and not any(res["loops"]) and not any(res["dups"]))
    live_ok = bool(min(res["live"]) > 0.05)

    print(f"\n(a) STDP moves exc, inh pinned : {a_ok}   max|dw| {max(res['a_dw_exc']):.4f}, "
          f"moved up to {max(res['a_moved']):,}, inh always -5: {all(res['a_inh_ok'])}")
    print(f"(b) Lamarckian inheritance     : {b_ok}   readback max|d| "
          f"{max(res['b_readback']):.2e}, newborn-vs-parent max|d| {max(res['b_inherit']):.2e}")
    print(f"(c) maturation not stalling    : {c_ok}   culled/round after grace: "
          f"{sorted(set(post))} (want [{M}])")
    print(f"(d) exc count bounded          : {d_ok}   {n0:,} -> {nl:,} "
          f"({growth:+.1%}), peak {max(res['d_n_exc']):,}")
    print(f"    liveness                   : {live_ok}   fired fraction "
          f"{min(res['live']):.2f}-{max(res['live']):.2f}")
    print(f"    Dale / loops / dups        : {inv_ok}   dale {all(res['dale'])}, "
          f"loops {sum(res['loops'])}, dups {sum(res['dups'])}")
    print(f"\nALL CHECKS: {'PASS' if (a_ok and b_ok and c_ok and d_ok and inv_ok and live_ok) else 'FAIL'}")
    # one-line machine-readable summary for the lr sweep
    print(f"SWEEP lr={a.stdp_lr} batch={a.batch} "
          f"maxdw_med={float(np.median(res['a_dw_exc'])):.4f} "
          f"maxdw_last={res['a_dw_exc'][-1]:.4f} "
          f"clip_med={float(np.median(res['clipped'])):.4f} "
          f"clip_last={res['clipped'][-1]:.4f} "
          f"live_med={float(np.median(res['live'])):.4f} "
          f"corr_batch_last={res['corr_batch'][-1]:+.4f} "
          f"corr_ewma_last={res['corr_ewma'][-1]:+.4f} "
          f"corr_ewma_best={max(res['corr_ewma']):+.4f}")


if __name__ == "__main__":
    main()
