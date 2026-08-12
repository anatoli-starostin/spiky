"""exp012 PART 1 (take 2): the spike-order comparator that the measured primitives allow.

The veto probe settled the design question. Measured on this substrate:

    lat_exc = 2 ticks      lat_inh = 1 tick      VETO WINDOW = 1 TICK WIDE

An inhibitory spike suppresses an excitatory cell only if it lands in the SAME tick as the
excitation. It does not "hold the cell down" -- arriving earlier or later does nothing. So a
single veto path is not a comparator at all, it is a coincidence detector for one exact time
difference:

    veto lands at   t16 + dB + dEI + 4
    drive lands at  t0  + dA
    veto fires  <=>  t0 - t16 == dB + dEI + 4 - dA        ONE value, not a threshold

To veto on the whole range t0 - t16 in [1, 31] -- which is what "x16 spikes first" means --
the inhibition has to arrive on EVERY one of those 31 offsets. That needs a DELAY COMB: 31
excitatory relays, each carrying input 16 to the shared inhibitory cell one tick later than
the last, so the inhibitory cell emits 31 consecutive veto spikes.

    in16 --dB+i--> E_i --dEI--> I --1--> E_a          i = 0 .. 30
    in0  --dA----> E_a --dOut--> OUT

That is the point: the comparator IS representable, but it costs ~32 hidden neurons for ONE
bit, not 8.
"""
import argparse
import json

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

IA, IB = 0, 16
LAT_E, LAT_I = 2, 1


def setup(n_exc, n_inh=1):
    G.set_hidden_capacity(n_exc, n_inh)
    G.set_out_per_target(1, "mean")
    G.set_weight_levels([-1.0] + [round(0.1 * i, 10) for i in range(11)])
    G.set_delay_levels(None)
    G.QUANTIZED = True
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)


def comb_genome(n_comb, dA, dB, dEI, dOut, coeff=1.0, gain=200.0):
    """E_a = exc 0; the comb is exc 1..n_comb; one inhibitory cell."""
    g = G.blank(n_exc=G.N_EXC_MAX, n_inh=G.N_INH_MAX)
    c_out = G.N_EXC_MAX + G.N_INH_MAX
    c_i = G.N_EXC_MAX

    def put(r, c, d, w):
        g["mask"][r, c] = True
        g["delay"][r, c] = int(d)
        g["weight"][r, c] = float(w)

    put(IA, 0, dA, 1.0)                                  # drive
    put(G.N_IN, c_out, dOut, 1.0)                        # E_a -> OUT
    put(G.N_IN + G.N_EXC_MAX, 0, 1, -1.0)                # I -> E_a, delay pinned to 1
    for i in range(n_comb):
        e = 1 + i
        put(IB, e, dB + i, 1.0)                          # in16 -> E_i, staggered by one tick
        put(G.N_IN + e, c_i, dEI, 1.0)                   # E_i -> I
    g["inh_coeff"] = float(coeff)
    g["gain"] = float(gain)
    g["aff_a"] = np.array([1.0])
    g["aff_b"] = np.array([0.0])
    return G.enforce(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    R = dict(lat_exc=LAT_E, lat_inh=LAT_I, veto_window_ticks=1, arms=[])

    for n_comb in (1, 2, 4, 8, 16, 31):
        setup(n_exc=1 + n_comb, n_inh=1)
        T.fit_target_stats(Yp)
        enc = LatencyEncoder(Xp)
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 12345)
        yt = G.task_targets(Yb, Xb).ravel()
        yv = G.task_targets(Yv, Xv).ravel()
        chance = float(yv.var())

        # C = dB + dEI + 4 - dA must be 1 so the covered range starts at t0 - t16 = 1
        dB, dEI = 2, 2
        dA = dB + dEI + 4 - 1                            # -> C = 1
        best = None
        for dOut in (40, 48, 56, 60):
            for coeff in (0.5, 1.0):
                g = comb_genome(n_comb, dA, dB, dEI, dOut, coeff=coeff)
                H = G.build([g], device="cuda")
                st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
                sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                             readout_map=st["readout_map"])
                m = float(sv["mse"][0])
                acc = float(((sv["calibrated"][:, 0, 0] > 0.5) == (yv > 0.5)).mean())
                tt = st["first"][:, 0, 0].astype(int)
                tv = sv["first"][:, 0, 0].astype(int)
                glob = float(yt.mean())
                tab = np.array([yt[tt == k].mean() if (tt == k).any() else glob
                                for k in range(33)])
                pv = tab[np.clip(tv, 0, 32)]
                lut = float(((pv - yv) ** 2).mean())
                lacc = float(((pv > 0.5) == (yv > 0.5)).mean())
                silent = float((tv == 32).mean())
                del H, st, sv
                torch.cuda.empty_cache()
                if best is None or lut < best["lut"]:
                    best = dict(n_comb=n_comb, n_hidden=1 + n_comb + 1, dA=dA, dB=dB,
                                dEI=dEI, dOut=dOut, coeff=coeff, chance=chance,
                                diagls=m, diagls_acc=acc, lut=lut, lut_acc=lacc,
                                silent=silent)
        R["arms"].append(best)
        print(f"comb {n_comb:2d} relays ({best['n_hidden']:2d} hidden)  "
              f"diagls {best['diagls']:.4f} ({100 * best['diagls_acc']:5.2f}%)   "
              f"best-decode {best['lut']:.4f} ({100 * best['lut_acc']:5.2f}%)   "
              f"silent {best['silent']:.3f}   dOut {best['dOut']} coeff {best['coeff']}",
              flush=True)

    print(f"\nchance {R['arms'][0]['chance']:.4f}   encoder floor 0.0273 (95.85%)")
    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
