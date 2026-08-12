"""exp012 PART 1: hand-build a spike-order detector for b* = 1[x_norm[0] > x_norm[16]].

Larger x fires EARLIER, so the bit is "does input 0 spike before input 16". The substrate
constrains the design hard:

  * inputs may only drive EXCITATORY cells (no in->inh, no in->out)
  * inhibitory cells may only target excitatory cells (no inh->out)
  * every inh->exc delay is PINNED to 1 tick

so the veto cannot be wired straight from an input -- it has to go
input 16 -> E_b -> I -> E_a, three hops, against the one hop input 0 -> E_a. The delay budget
has to make the three-hop veto arrive no later than the one-hop drive.

    in0  --dA-->  E_a  --dOut-->  OUT          E_a fires unless vetoed
    in16 --dB-->  E_b  --dEI-->  I  --1-->  E_a      (veto)

bit = 1  ->  E_a fires  ->  OUT fires at t0 + dA + dOut + latency
bit = 0  ->  veto lands first  ->  E_a silent  ->  OUT silent (tick 32)

Everything is set by hand and swept on the real engine; nothing is evolved.
"""
import argparse
import itertools
import json

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder

IA, IB = 0, 16
N_EXC, N_INH = 4, 2


def setup(quantized=True):
    G.set_hidden_capacity(N_EXC, N_INH)
    G.set_out_per_target(1, "mean")
    G.set_weight_levels([-1.0] + [round(0.1 * i, 10) for i in range(11)])
    G.set_delay_levels(None)                 # free integer delays for the hand build
    G.QUANTIZED = quantized
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)


def handbuilt(dA, dB, dEI, dOut, w_drive=1.0, w_b=1.0, w_ei=1.0, w_out=1.0,
              coeff=1.0, gain=200.0):
    """E_a = exc 0, E_b = exc 1, I = inh 0, and the single output neuron."""
    g = G.blank(n_exc=N_EXC, n_inh=N_INH)
    E_A, E_B, I0 = 0, 1, 0
    r_ea, r_eb = G.N_IN + E_A, G.N_IN + E_B
    c_ea, c_eb = E_A, E_B
    c_i = G.N_EXC_MAX + I0
    c_out = G.N_EXC_MAX + G.N_INH_MAX

    def put(r, c, d, w):
        g["mask"][r, c] = True
        g["delay"][r, c] = int(d)
        g["weight"][r, c] = float(w)

    put(IA, c_ea, dA, w_drive)            # the drive
    put(IB, c_eb, dB, w_b)                # the veto's first hop
    put(r_eb, c_i, dEI, w_ei)             # E_b -> I
    put(r_ea, c_out, dOut, w_out)         # E_a -> OUT
    put(G.N_IN + G.N_EXC_MAX + I0, c_ea, 1, -1.0)   # I -> E_a, delay pinned to 1
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
    setup()

    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, 12345)
    yt = G.task_targets(Yb, Xb).ravel()
    yv = G.task_targets(Yv, Xv).ravel()
    chance = float(yv.var())

    def evaluate(g):
        H = G.build([g], device="cuda")
        st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                     readout_map=st["readout_map"])
        m = float(sv["mse"][0])
        acc = float(((sv["calibrated"][:, 0, 0] > 0.5) == (yv > 0.5)).mean())
        tk_t = st["first"][:, 0, 0].astype(int)
        tk_v = sv["first"][:, 0, 0].astype(int)
        # the best ANY decode of that tick could do -- separates 'circuit is wrong' from
        # 'the linear scale+shift cannot read a correct circuit'
        glob = float(yt.mean())
        tab = np.array([yt[tk_t == k].mean() if (tk_t == k).any() else glob
                        for k in range(33)])
        pv = tab[np.clip(tk_v, 0, 32)]
        lut = float(((pv - yv) ** 2).mean())
        lacc = float(((pv > 0.5) == (yv > 0.5)).mean())
        del H, st, sv
        torch.cuda.empty_cache()
        return dict(diagls=m, diagls_acc=acc, lut=lut, lut_acc=lacc,
                    silent=float((tk_v == 32).mean()), n_ticks=int(len(np.unique(tk_v))),
                    tick_hist=np.bincount(tk_v, minlength=33).tolist())

    print(f"chance {chance:.4f}   P(1) held-out {yv.mean():.4f}\n")
    R = dict(chance=chance, sweep=[], best=None)

    # ---- sweep the timing that makes or breaks the veto. dA - (dB + dEI + 1) is the margin
    # the three-hop veto has over the one-hop drive; everything else is secondary.
    best = None
    for dB, dEI, marg, dOut, coeff in itertools.product(
            (2,), (2,), (0, 1, 2, 3, 4, 6, 8), (40, 48, 56), (1.0, 2.0)):
        dA = dB + dEI + 1 + marg
        if dA > 64 or dOut > 64:
            continue
        g = handbuilt(dA, dB, dEI, dOut, coeff=coeff)
        r = evaluate(g)
        row = dict(dA=dA, dB=dB, dEI=dEI, margin=marg, dOut=dOut, coeff=coeff, **r)
        R["sweep"].append({k: v for k, v in row.items() if k != "tick_hist"})
        print(f"  dA {dA:2d} dB {dB} dEI {dEI} margin {marg} dOut {dOut:2d} coeff {coeff:.1f}"
              f" -> diagls {r['diagls']:.4f} ({100 * r['diagls_acc']:.1f}%)  "
              f"best-decode {r['lut']:.4f} ({100 * r['lut_acc']:.1f}%)  "
              f"silent {r['silent']:.3f}  ticks {r['n_ticks']}", flush=True)
        if best is None or r["lut"] < best[1]["lut"]:
            best = (row, r, g)

    row, r, g = best
    R["best"] = {k: v for k, v in row.items()}
    R["best_genome"] = dict(
        mask=np.argwhere(g["mask"]).tolist(),
        delays=[int(g["delay"][i, j]) for i, j in np.argwhere(g["mask"])],
        weights=[float(g["weight"][i, j]) for i, j in np.argwhere(g["mask"])],
        inh_coeff=float(G.inh_coeff_of(g)), gain=float(G.gain_of(g)))
    print(f"\nBEST hand-built circuit: dA {row['dA']} dB {row['dB']} dEI {row['dEI']} "
          f"dOut {row['dOut']} coeff {row['coeff']}")
    print(f"  diagls (what evolution optimises) {r['diagls']:.4f}  acc {100*r['diagls_acc']:.2f}%")
    print(f"  best possible decode of its tick  {r['lut']:.4f}  acc {100*r['lut_acc']:.2f}%")
    print(f"  silent fraction {r['silent']:.4f}   distinct ticks {r['n_ticks']}")
    print(f"  chance {chance:.4f}   encoder floor 0.0273")

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
