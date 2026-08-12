"""exp012 Part B step 1: the cross-inhibition WTA comparator, now on LIF neurons.

Under Izhikevich this circuit was inert -- the loser fired in 92-100% of samples and the
argmin answer was bit-identical across all 18 settings, because a membrane deviation was
erased inside one tick by the regenerative quadratic. LIF removes the quadratic, so an
inhibitory impulse now decays with tau (half-life ~13.7 ticks at tau = 20) and can actually
hold the loser down.

    in0  --dA--> A --dI--> I_A --1--> B
    in16 --dB--> B --dI--> I_B --1--> A
    A --dOut--> OUT

The regime that makes it work: the drive weight sits just ABOVE threshold, so A fires from
its own input alone, but a veto still lingering in its membrane keeps the same input below
threshold. Everything is swept rather than assumed.
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
from tiny_cross_inhibition import hidden_first, wta_genome

IA, IB = 0, 16


def setup():
    G.set_hidden_capacity(2, 2)
    G.set_out_per_target(1, "mean")
    G.QUANTIZED = False              # hand build: free weights, the grid is not the question
    G.set_delay_levels(None)
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)


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
    yv = G.task_targets(Yv, Xv).ravel()
    chance = float(yv.var())
    ev = enc(Xv)
    tie = ev[:, IA] == ev[:, IB]
    raw = (ev[:, IA] < ev[:, IB]).astype(float)
    raw[tie] = float(yv[tie].mean() > 0.5)
    R = dict(chance=chance, encoder_floor=0.0273,
             raw_argmin=dict(mse=float(((raw - yv) ** 2).mean()),
                             acc=float(((raw > 0.5) == (yv > 0.5)).mean())),
             sweep=[])
    print(f"chance {chance:.4f} | encoder floor 0.0273/95.85% | raw argmin "
          f"{R['raw_argmin']['mse']:.4f}/{100 * R['raw_argmin']['acc']:.2f}% | "
          f"Izhikevich evolved 0.112, Izhikevich hand-built WTA 0.0800/92.00%\n")

    best = None
    for tau, w_drive, w_inh, dI, refr in itertools.product(
            (10.0, 20.0, 40.0), (1.05, 1.3, 2.0), (-1.5, -3.0), (1, 2), (0, 3)):
        G.set_lif(tau=tau, threshold=1.0, v_rest=0.0, v_reset=0.0, refractory_ticks=refr)
        g = wta_genome(dA=4, dB=4, dI=dI, dOut=48, w_inh=w_inh, coeff=1.0, gain=1.0,
                       w_drive=w_drive, w_ei=w_drive)
        try:
            H = G.build([g], device="cuda")
            exc = hidden_first(H, Xv, enc, 0)
        except Exception as e:                                   # a regime the engine hates
            print(f"  tau {tau:4.0f} w {w_drive:4.2f} inh {w_inh:5.1f} dI {dI} refr {refr}"
                  f" -> FAILED: {type(e).__name__}")
            continue
        fA, fB = exc[:, 0], exc[:, 1]
        onlyA = (fA < G.N_TICKS) & (fB >= G.N_TICKS)
        onlyB = (fB < G.N_TICKS) & (fA >= G.N_TICKS)
        both = (fA < G.N_TICKS) & (fB < G.N_TICKS)
        clean = float((onlyA | onlyB).mean())
        # argmin readout: the winner is whoever fired; if only one fired that IS the answer
        pred = np.where(onlyA, 1.0, np.where(onlyB, 0.0, (fA < fB).astype(float)))
        am_mse = float(((pred - yv) ** 2).mean())
        am_acc = float(((pred > 0.5) == (yv > 0.5)).mean())
        st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                     readout_map=st["readout_map"])
        d_mse = float(sv["mse"][0])
        d_acc = float(((sv["calibrated"][:, 0, 0] > 0.5) == (yv > 0.5)).mean())
        del H, st, sv
        torch.cuda.empty_cache()
        row = dict(tau=tau, w_drive=w_drive, w_inh=w_inh, dI=dI, refractory=refr,
                   clean_wta=clean, frac_both=float(both.mean()),
                   argmin_mse=am_mse, argmin_acc=am_acc, diagls_mse=d_mse, diagls_acc=d_acc)
        R["sweep"].append(row)
        print(f"  tau {tau:4.0f} w {w_drive:4.2f} inh {w_inh:5.1f} dI {dI} refr {refr} -> "
              f"cleanWTA {100 * clean:5.1f}%  both {100 * both.mean():5.1f}%  | "
              f"argmin {am_mse:.4f}/{100 * am_acc:5.2f}%  diagls {d_mse:.4f}/"
              f"{100 * d_acc:5.2f}%", flush=True)
        if best is None or am_mse < best["argmin_mse"]:
            best = row

    R["best"] = best
    if best:
        print(f"\nBEST: tau {best['tau']:.0f} w_drive {best['w_drive']} w_inh {best['w_inh']} "
              f"dI {best['dI']} refr {best['refractory']}")
        print(f"  clean WTA {100 * best['clean_wta']:.2f}%   "
              f"argmin {best['argmin_mse']:.4f}/{100 * best['argmin_acc']:.2f}%   "
              f"diagls {best['diagls_mse']:.4f}/{100 * best['diagls_acc']:.2f}%")
    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
