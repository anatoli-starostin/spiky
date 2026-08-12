"""exp012: a two-neuron cross-inhibition (winner-take-all) comparator on the REAL engine.

    in0  --dA--> A --dI--> I_A --1--> B          (A vetoes B)
    in16 --dB--> B --dI--> I_B --1--> A          (B vetoes A)
    A --dOut--> OUT

Dale forbids an excitatory cell from inhibiting directly and forbids inputs from driving
inhibitory cells, so each side needs its own interneuron and the veto is a three-hop path
against the loser's one-hop drive.

The timing algebra says this should not work: A fires at t0 + dA + 2, so its veto reaches B
at t0 + dA + dI + 4, while B commits at t16 + dB. With dA = dB the veto lands on B's
commitment tick only when t16 - t0 == dI + 4 -- ONE offset, not a threshold. But that is a
prediction, so it gets measured rather than asserted, including at low Izhikevich-scale gain.

Two readouts are scored:
  argmin  which of A / B emits its first spike earlier -- the WTA answer itself
  diagls  the run's own fitted scale+shift on the output neuron, for comparability
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
N_EXC, N_INH = 2, 2


def setup(gain_levels=True):
    G.set_hidden_capacity(N_EXC, N_INH)
    G.set_out_per_target(1, "mean")
    G.set_weight_levels([-1.0] + [round(0.1 * i, 10) for i in range(11)])
    G.set_delay_levels(None)
    G.QUANTIZED = True
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)


def wta_genome(dA, dB, dI, dOut, w_inh=-1.0, coeff=1.0, gain=200.0, w_drive=1.0, w_ei=1.0):
    g = G.blank(n_exc=N_EXC, n_inh=N_INH)
    A, B = 0, 1
    rA, rB = G.N_IN + A, G.N_IN + B
    cIA, cIB = G.N_EXC_MAX + 0, G.N_EXC_MAX + 1
    rIA, rIB = G.N_IN + G.N_EXC_MAX + 0, G.N_IN + G.N_EXC_MAX + 1
    c_out = G.N_EXC_MAX + G.N_INH_MAX

    def put(r, c, d, w):
        g["mask"][r, c] = True
        g["delay"][r, c] = int(d)
        g["weight"][r, c] = float(w)

    put(IA, A, dA, w_drive)
    put(IB, B, dB, w_drive)
    put(rA, cIA, dI, w_ei)
    put(rB, cIB, dI, w_ei)
    put(rIA, B, 1, w_inh)          # A's interneuron vetoes B
    put(rIB, A, 1, w_inh)          # B's interneuron vetoes A
    put(rA, c_out, dOut, 1.0)
    g["inh_coeff"] = float(coeff)
    g["gain"] = float(gain)
    g["aff_a"] = np.array([1.0])
    g["aff_b"] = np.array([0.0])
    return G.enforce(g)


def hidden_first(H, X, enc, kind, current=200.0):
    """-> [B, n] first spike tick of every neuron of that meta; N_TICKS = never fired."""
    from spiky.spnet.spnet import NeuronDataType
    G.grow_run_episode(H, X, enc, current=current)
    sp = H["spnet"]
    ids = torch.as_tensor(H["ids"][kind], dtype=torch.int32, device=H["device"])
    R = sp.export_neuron_data(ids, X.shape[0], NeuronDataType.Spike, 0, G.N_TICKS - 1)
    R = R.reshape(X.shape[0], -1, G.N_TICKS)
    w = torch.arange(G.N_TICKS, 0, -1, device=R.device, dtype=R.dtype)
    return (G.N_TICKS - (R.ne(0) * w).amax(-1)).cpu().numpy()


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

    # ---- the no-network baseline: argmin of the two RAW input spike times
    ev = enc(Xv)
    raw = (ev[:, IA] < ev[:, IB]).astype(float)
    tie = ev[:, IA] == ev[:, IB]
    raw[tie] = float(yv[tie].mean() > 0.5)          # the best fixed tie-break
    R = dict(chance=chance, n_val=int(len(Xv)),
             raw_argmin=dict(mse=float(((raw - yv) ** 2).mean()),
                             acc=float(((raw > 0.5) == (yv > 0.5)).mean()),
                             tie_fraction=float(tie.mean())),
             sweep=[])
    print(f"chance {chance:.4f}   encoder floor 0.0273 / 95.85%   evolved 0.112")
    print(f"raw argmin of the two input spike times (NO network): "
          f"MSE {R['raw_argmin']['mse']:.4f}  acc {100 * R['raw_argmin']['acc']:.2f}%  "
          f"ties {100 * R['raw_argmin']['tie_fraction']:.2f}%\n")

    best = None
    for gain, dI, coeff, w_inh in itertools.product(
            (200.0, 60.0, 20.0), (1, 2, 3), (0.5, 1.0), (-1.0,)):
        dA = dB = 4
        dOut = 48
        g = wta_genome(dA, dB, dI, dOut, w_inh=w_inh, coeff=coeff, gain=gain)
        H = G.build([g], device="cuda")
        exc = hidden_first(H, Xv, enc, 0)
        fA, fB = exc[:, 0], exc[:, 1]
        both = (fA < G.N_TICKS) & (fB < G.N_TICKS)
        onlyA = (fA < G.N_TICKS) & (fB >= G.N_TICKS)
        onlyB = (fB < G.N_TICKS) & (fA >= G.N_TICKS)
        neither = (fA >= G.N_TICKS) & (fB >= G.N_TICKS)
        pred = (fA < fB).astype(float)
        argmin_acc = float(((pred > 0.5) == (yv > 0.5)).mean())
        argmin_mse = float(((pred - yv) ** 2).mean())

        st = G.score(H, Xb, Yb, enc, genomes=[g], readout="diagls")
        sv = G.score(H, Xv, Yv, enc, genomes=[g], readout="diagls",
                     readout_map=st["readout_map"])
        d_mse = float(sv["mse"][0])
        d_acc = float(((sv["calibrated"][:, 0, 0] > 0.5) == (yv > 0.5)).mean())
        del H, st, sv
        torch.cuda.empty_cache()

        row = dict(gain=gain, dI=dI, coeff=coeff, w_inh=w_inh, dA=dA, dOut=dOut,
                   frac_both_fire=float(both.mean()), frac_only_A=float(onlyA.mean()),
                   frac_only_B=float(onlyB.mean()), frac_neither=float(neither.mean()),
                   argmin_mse=argmin_mse, argmin_acc=argmin_acc,
                   diagls_mse=d_mse, diagls_acc=d_acc)
        R["sweep"].append(row)
        print(f"  gain {gain:5.0f} dI {dI} coeff {coeff:.1f} -> "
              f"both fire {100 * both.mean():5.1f}%  onlyA {100 * onlyA.mean():5.1f}%  "
              f"onlyB {100 * onlyB.mean():5.1f}%  none {100 * neither.mean():5.1f}%  | "
              f"argmin {argmin_mse:.4f}/{100 * argmin_acc:5.2f}%  "
              f"diagls {d_mse:.4f}/{100 * d_acc:5.2f}%", flush=True)
        if best is None or argmin_mse < best["argmin_mse"]:
            best = row

    R["best"] = best
    print(f"\nBEST cross-inhibition setting: gain {best['gain']:.0f} dI {best['dI']} "
          f"coeff {best['coeff']:.1f}")
    print(f"  argmin readout  MSE {best['argmin_mse']:.4f}  acc {100*best['argmin_acc']:.2f}%")
    print(f"  diagls readout  MSE {best['diagls_mse']:.4f}  acc {100*best['diagls_acc']:.2f}%")
    print(f"  clean WTA (exactly one of A/B fires): "
          f"{100 * (best['frac_only_A'] + best['frac_only_B']):.2f}%   "
          f"both fire {100 * best['frac_both_fire']:.2f}%")

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
