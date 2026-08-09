"""Verify the purely-structural mutation rules hold, by diffing genomes across a mutation."""
import numpy as np

import steady_state as S

W_MAX = 30.0
rng = np.random.default_rng(0)
g = S.seed_genome(np.random.default_rng(0), W_MAX)
add_w, weak = 0.10 * W_MAX, 0.02 * W_MAX

print(f"neuron counts: input {S.N_IN}, exc {S.N_EXC}, inh {S.N_INH}, output {S.N_OUT}")
assert (S.N_IN, S.N_EXC, S.N_INH, S.N_OUT) == (17, 800, 200, 6)
print("  -> 17/800/200/6 CONFIRMED")
print(f"add_weight_exc {add_w}  weak_thresh {weak}  (add is "
      f"{'ABOVE' if add_w > weak else 'BELOW'} threshold)")


def key(gg):
    return ((gg["src_pool"] * 10 + gg["tgt_pool"]) * 1000 + gg["src_idx"]) * 1000 + gg["tgt_idx"]


ok = True
for rnd in range(5):
    h = S.mutate_structural(g, rng, W_MAX, add_weight_exc=add_w, weak_thresh=weak)
    kg, kh = key(g), key(h)
    common, ig, ih = np.intersect1d(kg, kh, return_indices=True)

    # 1. SURVIVORS UNTOUCHED: every synapse present before and after must have an
    #    identical weight and delay. This is the core claim of "purely structural".
    dw = np.abs(g["weight"][ig] - h["weight"][ih]).max() if common.size else 0.0
    dd = np.abs(g["delay"][ig] - h["delay"][ih]).max() if common.size else 0
    # 2. Dale
    inh = h["src_pool"] == S.INH
    dale = bool((h["weight"][inh] == S.RES_W_INH).all() and (h["weight"][~inh] >= 0).all())
    # 3. index ranges
    inrange = all(
        (h[f"{e}_idx"][h[f"{e}_pool"] == p].max(initial=-1) < S.POOL_SIZE[p])
        for e in ("src", "tgt") for p in (S.EXC, S.INH, S.INP, S.OUTP)
        if (h[f"{e}_pool"] == p).any())
    # 4. newly added exc synapses: weight exactly add_w, and NOT prune-eligible
    new = np.setdiff1d(kh, kg)
    ni = np.nonzero(np.isin(kh, new))[0]
    ne = ni[h["src_pool"][ni] != S.INH]
    nin = ni[h["src_pool"][ni] == S.INH]
    new_exc_ok = bool(ne.size == 0 or ((h["weight"][ne] == add_w).all()
                                       and (np.abs(h["weight"][ne]) >= weak).all()))
    # 5. newly added inh synapses: delay 1, weight -5
    new_inh_ok = bool(nin.size == 0 or ((h["delay"][nin] == 1).all()
                                        and (h["weight"][nin] == S.RES_W_INH).all()))
    # 6. removed excitatory must all have been weak
    gone = np.setdiff1d(kg, kh)
    gi = np.nonzero(np.isin(kg, gone))[0]
    ge = gi[g["src_pool"][gi] != S.INH]
    rm_ok = bool(ge.size == 0 or (np.abs(g["weight"][ge]) < weak).all())

    # 7. no self-loops and no duplicate (src,tgt) anywhere in the genome
    loops = int(((h["src_pool"] == h["tgt_pool"]) & (h["src_idx"] == h["tgt_idx"])).sum())
    dups = int(h["weight"].size - np.unique(key(h)).size)
    good = (dw == 0 and dd == 0 and dale and inrange and new_exc_ok and new_inh_ok
            and rm_ok and loops == 0 and dups == 0)
    ok &= good
    print(f"round {rnd}: n {g['weight'].size:,}->{h['weight'].size:,}  "
          f"survivors {common.size:,} max|dw| {dw:.3g} max|dd| {dd}  "
          f"+{ne.size} exc +{nin.size} inh  -{ge.size} exc -{gi.size-ge.size} inh  "
          f"| dale {dale} range {inrange} newexc {new_exc_ok} newinh {new_inh_ok} "
          f"rm-weak-only {rm_ok} loops {loops} dups {dups}  "
          f"{'OK' if good else 'FAIL'}")
    g = h

print(f"\nVERDICT: {'ALL INVARIANTS HOLD' if ok else 'VIOLATION FOUND'}")
