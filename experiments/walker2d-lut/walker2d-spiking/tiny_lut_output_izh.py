"""exp012: can a STANDARD Izhikevich cell serve as the LUT output-stage neuron?

The anti-leaky LIF (cf_1 = +1/tau_m) reaches R2 0.979-0.994. Izhikevich is interesting here
because its own dynamics are anti-leaky in part of the range: with

    dV/dt = 0.04V^2 + 5V + 140 - U

the fixed points at the resting U are -70.44 (stable) and -54.56 (unstable). BELOW -54.56 the
cell is leaky and pulls back to rest; ABOVE it the quadratic runs away. So the regenerative
upstroke is a growing membrane -- but its rate is VOLTAGE-dependent (quadratic), not a fixed
exponential, so the effective logsumexp temperature drifts during the accumulation. That is
the distortion this measures.

Standard params only, exactly as meta type 0 in the chapter: cf_2=0.04, cf_1=5.0, cf_0=140.0,
a=0.02, b=0.2, c=-65.0, d=8.0, threshold=30.0. No re-parameterisation (that would be the LIF
again).
"""
import argparse
import itertools
import json

import numpy as np
import torch

from tiny_lut_output_stage import TAU, affine_fit, lut_targets, run_first_spike

IZH = dict(cf_2=0.04, cf_1=5.0, cf_0=140.0, a=0.02, b=0.2, c=-65.0, d=8.0,
           spike_threshold=30.0)


def build(n_src, delays, srcs, amp, out_meta, tau_drv=10.0, device="cuda"):
    from spiky.spnet.spnet import LIFNeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = int(delays.max())
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_drv, threshold=1.0), out_meta]
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=[n_src, 1],
                     initial_synapse_capacity=1 << 22, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(2)]
    tri = np.stack([delays - 1, ids[0][srcs], np.full(len(srcs), ids[1][0])], 1)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2,
                             max_groups_in_buffer=max(8192, 8 * (len(tri) + n_src + 8)))
    for i in range(2):
        ge.register_neuron_type(max_synapses=4 * (len(tri) + 4), growth_command_list=[])
    for i in range(2):
        t = torch.tensor(ids[i], dtype=torch.int32)
        ge.add_neurons(neuron_type_index=i, identifiers=t,
                       coordinates=torch.stack([torch.arange(t.numel()).float(),
                                                torch.zeros(t.numel()),
                                                torch.full((t.numel(),), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device=device), 1,
                              weights=torch.full((len(tri),), float(amp),
                                                 dtype=torch.float32, device=device))
    net.add_connections(chunk, 1)
    chunk.recycle()
    net.compile(shuffle_synapses_random_seed=None)
    return net, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--dims", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from spiky.spnet.spnet import NeuronMeta
    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    ntr = len(x) - 4000
    w_sel, out_true, _ = lut_targets(Z, x[ntr:ntr + a.n])
    roots = np.roots([0.04, 5.0, 140.0 - 0.2 * -65.0])
    print(f"Izhikevich fixed points at rest U: {np.round(roots, 2).tolist()}  "
          f"-> leaky below {roots.max():.2f}, anti-leaky above\n")

    R = {"note": "standard Izhikevich output neuron", "izh": IZH, "dims": {}}
    for o in [int(v) for v in a.dims.split(",")]:
        ws = w_sel[:, :, o]
        best = None
        # sweep the arrival spread (tight = inside the regenerative window, wide = LIF-like)
        # and the drive amplitude, keeping I in the Izhikevich-natural range
        for scale, amp in itertools.product((5.0, 20.0, 60.0, 113.4), (2.0, 6.0, 15.0, 40.0)):
            raw = -scale * ws
            C = float(np.ceil(-raw.min() + 1.0))
            arr = np.rint(raw + C).astype(np.int64)
            if arr.max() > 120:
                continue
            n_ticks = int(arr.max() + 80)
            net, ids = build(32, np.ones(32, np.int64), np.arange(32), amp,
                             NeuronMeta(neuron_type=1, **IZH))
            T = run_first_spike(net, ids, arr, 32, n_ticks)
            del net
            torch.cuda.empty_cache()
            fired = T < n_ticks
            if fired.mean() < 0.5:
                continue
            early = fired & (T <= arr.max(1))
            y = out_true[:, o]
            half = fired & (np.arange(len(y)) < len(y) // 2)
            if half.sum() < 20:
                continue
            aa, bb = affine_fit(T.astype(float), y, half)
            ev = fired & (np.arange(len(y)) >= len(y) // 2)
            pred = aa * T.astype(float) + bb
            mse = float(((pred[ev] - y[ev]) ** 2).mean())
            var = float(y[ev].var())
            row = dict(scale=scale, amp=amp, spread=float((arr.max(1) - arr.min(1)).mean()),
                       frac_fired=float(fired.mean()),
                       frac_early=float(early.mean()), mse=mse,
                       max_err=float(np.abs(pred[ev] - y[ev]).max()),
                       target_var=var, r2=1 - mse / var)
            print(f"  dim {o} scale {scale:6.1f} amp {amp:5.1f} spread "
                  f"{row['spread']:5.1f}  fired {100 * row['frac_fired']:5.1f}%  early "
                  f"{100 * row['frac_early']:5.1f}%  R2 {row['r2']:8.4f}")
            if best is None or row["r2"] > best["r2"]:
                best = row
        R["dims"][str(o)] = best
        if best:
            print(f"  -> dim {o} BEST R2 {best['r2']:.4f} at scale {best['scale']} "
                  f"amp {best['amp']} (MSE {best['mse']:.6f}, max|err| {best['max_err']:.4f})\n")
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
