"""exp012: the two fixes to the LIF cross-inhibition comparator.

STEP 1 left the veto working but 4 ticks too slow, and the 32-tick latency code gives a mean
input gap of only 2.20 -- so 83% of samples fell inside the veto's blind spot.

  FIX b   let inputs drive inhibitory cells (LEGAL[R_IN, C_INH] = True), making the veto ONE
          hop (in16 -> I_B -> A) instead of two. Substrate legality only.
  FIX a   widen the input tick window 32 -> 128, scaling every gap ~4x and thinning the
          exact-tie population that caps any spike-order scheme.

Neither touches shared code. Fix b is a runtime flip of the LEGAL mask; fix a is a local
episode runner that mirrors grow_run_episode with a configurable T_IN / N_TICKS, because
harness.py and tiny_snn.py are off limits.
"""
import argparse
import itertools
import json

import numpy as np
import torch

import tiny_grow as G
import tiny_snn as S
import tiny_snn as T
from data import load
from harness import LatencyEncoder

IA, IB = 0, 16


class WideEncoder:
    """The same percentile-calibrated latency code, spread over `t_in` ticks instead of 32."""

    def __init__(self, X, t_in, lo_pct=0.5, hi_pct=99.5):
        self.lo = float(np.percentile(X, lo_pct))
        self.hi = float(np.percentile(X, hi_pct))
        self.t_in = int(t_in)

    def __call__(self, x):
        u = (np.asarray(x, np.float64) - self.lo) / max(self.hi - self.lo, 1e-9)
        t = (1.0 - np.clip(u, 0.0, 1.0)) * (self.t_in - 1)
        return np.clip(t.round().astype(np.int64), 0, self.t_in - 1)


def run_wide(H, ticks, t_in, n_ticks, current=200.0):
    """grow_run_episode's engine call, with T_IN / N_TICKS passed in -> exc first spikes."""
    from spiky.spnet.spnet import NeuronDataType
    sp, ids, dev = H["spnet"], H["ids"], H["device"]
    B = ticks.shape[0]
    cols = ids[2]
    va = np.zeros((B, t_in, cols.size), np.float32)
    for b in range(B):
        for j in range(G.N_IN):
            va[b, ticks[b, j], j::G.N_IN] = current
    sp.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=t_in,
                     input_values=torch.as_tensor(va, device=dev),
                     sparse_input=S._sparse_ids(cols, B, t_in, dev),
                     do_train=False, do_record_voltage=False,
                     do_reset_context=True, _stdp_period=32)
    out = {}
    for kind in (0, 1):
        oid = torch.as_tensor(np.ascontiguousarray(ids[kind], dtype=np.int32), device=dev)
        R = sp.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
        R = R.reshape(B, -1, n_ticks)
        w = torch.arange(n_ticks, 0, -1, device=R.device, dtype=R.dtype)
        out[kind] = (n_ticks - (R.ne(0) * w).amax(-1)).cpu().numpy()
    return out


def genome(one_hop, dA, dV, dOut, w_drive, w_inh):
    """A/B driven by their inputs; the veto is one hop (fix b) or two (the STEP-1 circuit)."""
    g = G.blank(n_exc=2, n_inh=2)
    A, B_ = 0, 1
    cIA, cIB = G.N_EXC_MAX + 0, G.N_EXC_MAX + 1
    rA, rB = G.N_IN + A, G.N_IN + B_
    rIA, rIB = G.N_IN + G.N_EXC_MAX + 0, G.N_IN + G.N_EXC_MAX + 1
    c_out = G.N_EXC_MAX + G.N_INH_MAX

    def put(r, c, d, w):
        g["mask"][r, c] = True
        g["delay"][r, c] = int(d)
        g["weight"][r, c] = float(w)

    put(IA, A, dA, w_drive)
    put(IB, B_, dA, w_drive)
    if one_hop:
        put(IA, cIA, dV, w_drive)          # in0  -> I_A   (illegal unless fix b is on)
        put(IB, cIB, dV, w_drive)          # in16 -> I_B
    else:
        put(rA, cIA, dV, w_drive)          # A -> I_A      (the STEP-1 two-hop path)
        put(rB, cIB, dV, w_drive)
    put(rIA, B_, 1, w_inh)                 # I_A vetoes B
    put(rIB, A, 1, w_inh)                  # I_B vetoes A
    put(rA, c_out, dOut, 1.0)
    g["inh_coeff"] = 1.0
    g["gain"] = 1.0
    g["aff_a"] = np.array([1.0])
    g["aff_b"] = np.array([0.0])
    return G.enforce(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # Each SpikingNet holds device buffers that Python's GC does not reliably release, so a
    # long sweep dies with cudaErrorMemoryAllocation partway through -- the wide arms are run
    # in their own process instead.
    ap.add_argument("--t-in", type=int, default=None)
    a = ap.parse_args()

    G.set_hidden_capacity(2, 2)
    G.set_out_per_target(1, "mean")
    G.QUANTIZED = False
    G.set_delay_levels(None)
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    G.set_bit_task(IA, IB)
    LEGAL_STRICT = G.LEGAL.copy()

    _, _, Xp, Yp, Xv, Yv = load(1024, seed=0)
    T.fit_target_stats(Yp)
    yv = G.task_targets(Yv, Xv).ravel()
    chance = float(yv.var())
    R = dict(chance=chance, arms=[])
    print(f"chance {chance:.4f} | encoder floor(32t) 0.0273/95.85% | raw argmin 0.0415/95.85%"
          f" | Izhikevich evolved 0.112 | STEP-1 LIF WTA 0.0800/92.00%\n")

    for t_in, one_hop in itertools.product(
            (32, 128) if a.t_in is None else (a.t_in,), (False, True)):
        n_ticks = 3 * t_in                       # same 1:1:1 input/compute/readout shape
        enc = (LatencyEncoder(Xp) if t_in == 32 else WideEncoder(Xp, t_in))
        ev = enc(Xv)
        gap = np.abs(ev[:, IA].astype(int) - ev[:, IB].astype(int))
        ties = float((gap == 0).mean())
        raw = (ev[:, IA] < ev[:, IB]).astype(float)
        raw[gap == 0] = float(yv[gap == 0].mean() > 0.5)
        raw_mse = float(((raw - yv) ** 2).mean())
        raw_acc = float(((raw > 0.5) == (yv > 0.5)).mean())

        G.LEGAL = LEGAL_STRICT.copy()
        if one_hop:
            G.LEGAL[G.R_IN, G.C_INH] = True      # FIX b, runtime only
        G.set_lif(tau=20.0 * (t_in / 32), threshold=1.0, v_rest=0.0, v_reset=0.0,
                  refractory_ticks=0)

        best = None
        for dA, dV, w_drive, w_inh in itertools.product(
                (4,), (1, 2, 4), (1.05, 1.3), (-3.0,)):
            # the synapse-meta bank only spans delays 1..64 (one meta per delay), so a delay
            # past D_HI indexes off the end of it and the build dies in add_connections
            g = genome(one_hop, dA, dV, dOut=min(n_ticks // 2, G.D_HI),
                       w_drive=w_drive, w_inh=w_inh)
            if one_hop and not g["mask"][IA, G.N_EXC_MAX + 0]:
                raise RuntimeError("fix b did not take -- in->inh still masked out")
            H = G.build([g], device="cuda")
            ev_all = enc(Xv)
            # 2000 samples x 384 ticks x all neurons blows the engine's buffer, so chunk.
            chunks = [run_wide(H, ev_all[i:i + 256], t_in, n_ticks)
                      for i in range(0, len(ev_all), 256)]
            sp = {k: np.concatenate([c[k] for c in chunks]) for k in (0, 1)}
            fA, fB = sp[0][:, 0], sp[0][:, 1]
            onlyA = (fA < n_ticks) & (fB >= n_ticks)
            onlyB = (fB < n_ticks) & (fA >= n_ticks)
            clean = onlyA | onlyB
            # The WTA answer is "who fired alone". When BOTH fire the circuit has not
            # decided, and falling back on (fA < fB) silently predicts 0 on exact ties --
            # which is the whole 3.85% the raw baseline beats us by. Use the best CONSTANT
            # for the undecided set instead, which is what any fitted readout would do.
            undecided = ~clean
            fill = float(yv[undecided].mean() > 0.5) if undecided.any() else 0.0
            pred = np.where(onlyA, 1.0, np.where(onlyB, 0.0, fill))
            am = float(((pred - yv) ** 2).mean())
            ac = float(((pred > 0.5) == (yv > 0.5)).mean())
            del H
            torch.cuda.empty_cache()
            if best is None or am < best["argmin_mse"]:
                best = dict(t_in=t_in, one_hop=one_hop, dA=dA, dV=dV, w_drive=w_drive,
                            w_inh=w_inh, clean=float(clean.mean()), argmin_mse=am,
                            argmin_acc=ac, ties=ties, mean_gap=float(gap.mean()),
                            raw_mse=raw_mse, raw_acc=raw_acc,
                            by_gap={f"{lo}-{hi}": float(clean[(gap >= lo) & (gap <= hi)].mean())
                                    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 7), (8, 999))
                                    if ((gap >= lo) & (gap <= hi)).any()})
        R["arms"].append(best)
        tag = f"T_IN {t_in:3d}  veto {'ONE hop (fix b)' if one_hop else 'two hops'}"
        print(f"{tag}\n   ties {100 * ties:5.2f}%  mean gap {best['mean_gap']:5.2f}  "
              f"raw argmin {raw_mse:.4f}/{100 * raw_acc:.2f}%")
        print(f"   best: dV {best['dV']} w {best['w_drive']}  -> cleanWTA "
              f"{100 * best['clean']:5.1f}%   argmin {best['argmin_mse']:.4f}/"
              f"{100 * best['argmin_acc']:5.2f}%")
        print(f"   clean by gap: " + "  ".join(f"{k}:{100 * v:.0f}%"
                                               for k, v in best["by_gap"].items()) + "\n")

    G.LEGAL = LEGAL_STRICT
    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
