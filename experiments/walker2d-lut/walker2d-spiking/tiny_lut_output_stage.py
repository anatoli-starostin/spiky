"""exp012: the handcrafted OUTPUT stage of the walker2d LUT teacher, on the REAL engine.

Target, per output dim o, over the 32 tables:

    out[o] = tph*tau * log( (1/tph) * sum_t exp( w_sel[t,o] / tau ) )      tph=32, tau=0.09036568

A decaying LIF membrane integrating impulses A at arrival times a_t crosses threshold at

    T = tau_eff * log( sum_t A * exp( a_t / tau_eff ) ) - tau_eff*log(theta)

so if a_t = (tau_eff/tau) * w_sel[t,o] + c then T is affine in the target. Two ways to
produce those arrival times:

  VERSION A  spike-time encoding -- 32 driver neurons FIRE at a_t; one uniform synapse delay.
  VERSION B  conduction-delay encoding -- ONE NEURON PER LUT CELL (32*64 = 2048), all firing
             simultaneously when selected, the value carried by a FIXED per-synapse delay
             d[t,k] = (tau_eff/tau)*weights[t,k,o] + c0. A per-sample delay is impossible --
             a delay belongs to a synapse, not to an episode -- so B only exists in the
             one-neuron-per-cell form. That IS the user's stated assumption, made concrete.

tau_eff is CALIBRATED against the engine's actual per-tick decay rather than assumed: the
kernel takes 2 Euler half-steps, so the decay is (1 - 0.5/tau_m)^2 per tick, not exp(-1/tau_m).
"""
import argparse
import json

import numpy as np
import torch

TPH, TAU, CLAMP = 32, 0.09036568, 60.0


def lut_targets(Z, x):
    """-> w_sel [B,32,6] and the true LUT output [B,6]."""
    A, B_ = Z["anchor_a"], Z["anchor_b"]
    W = Z["weights"].astype(np.float64)
    d = x[:, A] - x[:, B_]
    idx = ((d > 0) * (2 ** np.arange(5, -1, -1))).sum(-1)          # [B,32]
    w_sel = W.reshape(32 * 64, 6)[idx + (np.arange(32) * 64)[None, :]]
    out = TPH * TAU * (np.log(np.exp(np.clip(w_sel / TAU, -CLAMP, CLAMP)).sum(1)) - np.log(TPH))
    return w_sel, out, idx


def calibrate_tau(tau_m, n_euler=2, dt=0.5):
    """The tick-to-tick decay the kernel actually applies -> the effective LSE temperature."""
    per_tick = (1.0 - dt / tau_m) ** n_euler
    return -1.0 / np.log(per_tick), per_tick


def build_net(n_src, n_out, delays, srcs, amp, tau_m, threshold, device="cuda", grow=False):
    """One LIF output neuron per column; `delays[i]`/`srcs[i]` describe synapse i."""
    from spiky.spnet.spnet import LIFNeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    dmax = int(delays.max())
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=2, _backward_group_size=2)
              for d in range(1, dmax + 1)]
    # The drivers must FIRE on command, so their threshold has to be low and the injected
    # current large. A high threshold to "stop spontaneous firing" also blocks the forced
    # spike -- that is why nothing fired on the first attempt.
    from spiky.spnet.spnet import NeuronMeta
    if grow:
        # ANTI-LEAKY output: cf_1 = +1/tau instead of -1/tau, so v' = +v/tau + I and the
        # membrane GROWS. This is only a parameter choice on the existing kernel -- nothing
        # in the engine is modified. With growth, early arrivals are AMPLIFIED rather than
        # decayed, so all 32 terms stay live and V is monotone, which makes the threshold
        # crossing unique and necessarily after the last arrival for a large enough theta.
        out_meta = NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / tau_m, cf_0=0.0,
                              a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=threshold)
    else:
        out_meta = LIFNeuronMeta(neuron_type=1, tau=tau_m, threshold=threshold)
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_m, threshold=1.0), out_meta]
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas,
                     neuron_counts=[n_src, n_out], initial_synapse_capacity=1 << 22,
                     summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(2)]
    tri = np.stack([delays - 1, ids[0][srcs], np.full(len(srcs), ids[1][0])], 1)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=2,
                             max_groups_in_buffer=max(8192, 8 * (len(tri) + n_src + n_out)))
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


def run_first_spike(net, ids, drive_ticks, n_src, n_ticks, device="cuda"):
    """drive_ticks [B, n_src] (-1 = this driver does not fire) -> output first spike [B]."""
    from spiky.spnet.spnet import NeuronDataType
    B = drive_ticks.shape[0]
    va = np.zeros((B, n_ticks, n_src), np.float32)
    for b in range(B):
        m = drive_ticks[b] >= 0
        va[b, drive_ticks[b][m], np.where(m)[0]] = 1e6          # force the driver to spike
    cols = ids[0]
    sid = torch.as_tensor(np.ascontiguousarray(cols, dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    oid = torch.as_tensor(np.ascontiguousarray(ids[1], dtype=np.int32), device=device)
    R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
    R = R.reshape(B, -1, n_ticks)[:, 0, :]
    w = torch.arange(n_ticks, 0, -1, device=R.device, dtype=R.dtype)
    return (n_ticks - (R.ne(0) * w).amax(-1)).cpu().numpy().astype(np.int64)


def affine_fit(T, y, ok):
    v = T[ok].var()
    a = float(np.cov(T[ok], y[ok], bias=True)[0, 1] / v) if v > 1e-12 else 0.0
    return a, float(y[ok].mean() - a * T[ok].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-m", type=float, default=10.0)
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--c0", type=float, default=60.0)
    ap.add_argument("--amp", type=float, default=0.02)
    ap.add_argument("--version", default="A", choices=("A", "B"))
    ap.add_argument("--amp-safe", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    ntr = len(x) - 4000
    tr = slice(ntr, ntr + a.n)                 # held-out block
    w_sel, out_true, idx = lut_targets(Z, x[tr])
    o = a.dim
    tau_eff, per_tick = calibrate_tau(a.tau_m)
    scale = tau_eff / TAU
    print(f"tau_m {a.tau_m}  ->  per-tick decay {per_tick:.6f}  CALIBRATED tau_eff "
          f"{tau_eff:.4f}  (naive exp(-1/tau_m) would give {a.tau_m:.4f})")
    print(f"scale tau_eff/tau = {scale:.3f} ticks per unit of w")

    W = Z["weights"].astype(np.float64)[:, :, o]
    lo, hi = W.min(), W.max()
    print(f"w[:, :, {o}] range [{lo:.4f}, {hi:.4f}] -> tick spread "
          f"{scale * (hi - lo):.1f}, offset c0 {a.c0}")

    if a.version == "A":
        n_src = 32
        arr = np.rint(scale * w_sel[:, :, o] + a.c0).astype(np.int64)   # firing ticks
        delays = np.ones(32, np.int64)
        srcs = np.arange(32)
        drive = arr
    else:
        n_src = 32 * 64
        d = np.rint(scale * W + a.c0).astype(np.int64).ravel()          # per-CELL fixed delay
        delays, srcs = d, np.arange(n_src)
        drive = np.full((len(w_sel), n_src), -1, np.int64)
        rows = np.arange(len(w_sel))[:, None]
        drive[rows, (np.arange(32) * 64)[None, :] + idx] = 1            # all fire at tick 1
    assert delays.min() >= 1, f"delay {delays.min()} below the engine minimum of 1"

    # The arrival spread is (tau_eff/tau)*range(w) ticks = range(w)/tau time-constants = 11.88
    # WHATEVER tau_m is -- the ratio is a property of the teacher, not of the circuit. So
    # exp(a_t/tau_eff) spans e^11.88 = 1.4e5 and the sum is dominated by the latest arrival.
    # That is faithful (the true LUT is equally max-dominated), but it means the amplitude has
    # to be calibrated: too small and nothing ever fires, too large and a single early spike
    # crosses threshold and the output reports a max instead of a sum.
    arrivals = (drive + delays[None, :]) if a.version == "A" else \
        (1 + delays[None, :] * np.ones((len(w_sel), 1), np.int64))
    if a.version == "B":
        arrivals = np.where(drive >= 0, 1 + delays[None, :], -1)
    S = np.array([np.exp(-(r[r >= 0].max() - r[r >= 0]) / tau_eff).sum() for r in arrivals])
    amp = 1.0 / float(np.median(S))
    part = np.array([np.sort(np.exp(-(r[r >= 0].max() - r[r >= 0]) / tau_eff))[:-1].sum()
                     for r in arrivals])
    print(f"sum-at-last-arrival S: median {np.median(S):.4f} "
          f"[{S.min():.4f},{S.max():.4f}]  -> amplitude {amp:.6f}")
    print(f"  MARGIN: without the final spike the partial sum reaches "
          f"{amp * np.percentile(part, 99):.4f} of threshold 1.0 (99th pct) -- "
          f"{'SAFE' if amp * np.percentile(part, 99) < 1.0 else 'UNSAFE: fires early'}")
    # Firing early truncates the sum, so the amplitude must be small enough that the partial
    # sum BEFORE the final arrival stays under threshold. Scaling by 1/max-partial does that.
    if a.amp_safe:
        amp = 1.0 / float(np.percentile(part, 99.5)) * 0.98
        print(f"  amp-safe: rescaled to {amp:.6f} so the pre-final partial sum stays under 1.0")
    a.amp = amp
    n_ticks = int(max(delays.max(), drive.max()) + 4 * a.tau_m + 20)
    print(f"version {a.version}: {n_src} source neurons, delays [{delays.min()},"
          f"{delays.max()}], episode {n_ticks} ticks")

    net, ids = build_net(n_src, 1, delays, srcs, a.amp, a.tau_m, threshold=1.0)
    T = run_first_spike(net, ids, drive, n_src, n_ticks)
    fired = T < n_ticks
    print(f"output fired on {100 * fired.mean():.2f}% of samples; "
          f"T range [{T[fired].min() if fired.any() else -1}, "
          f"{T[fired].max() if fired.any() else -1}]")

    y = out_true[:, o]
    half = fired & (np.arange(len(y)) < len(y) // 2)
    aa, bb = affine_fit(T.astype(float), y, half)
    ev = fired & (np.arange(len(y)) >= len(y) // 2)
    pred = aa * T.astype(float) + bb
    mse = float(((pred[ev] - y[ev]) ** 2).mean())
    mx = float(np.abs(pred[ev] - y[ev]).max())
    var = float(y[ev].var())
    print(f"\naffine a {aa:.6f} b {bb:.6f}")
    print(f"held-out MSE {mse:.6f}   max|err| {mx:.6f}   target var {var:.6f}   "
          f"ratio {mse / var:.6f}   R2 {1 - mse / var:.6f}")
    R = dict(version=a.version, tau_m=a.tau_m, tau_eff=tau_eff, per_tick_decay=per_tick,
             scale=scale, dim=o, n=int(a.n), amp=a.amp, c0=a.c0, n_ticks=n_ticks,
             frac_fired=float(fired.mean()), mse=mse, max_err=mx, target_var=var,
             ratio=mse / var, r2=1 - mse / var, affine=[aa, bb])
    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
